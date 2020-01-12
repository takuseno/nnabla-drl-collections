import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.solvers as S
import random
import argparse
import gym
import os
import cv2

from datetime import datetime
from collections import deque
from nnabla.monitor import Monitor, MonitorSeries
from nnabla.ext_utils import get_extension_context


#------------------------------- neural network ------------------------------#
def cnn_network(obs, num_actions, num_heads, scope):
    with nn.parameter_scope(scope):
        with nn.parameter_scope('shared'):
            out = PF.convolution(obs, 32, (8, 8), stride=(4, 4), name='conv1')
            out = F.relu(out)
            out = PF.convolution(out, 64, (4, 4), stride=(2, 2), name='conv2')
            out = F.relu(out)
            out = PF.convolution(out, 64, (3, 3), stride=(1, 1), name='conv3')
            out = F.relu(out)
        q_values = []
        for i in range(num_heads):
            with nn.parameter_scope('head%d' % i):
                h = F.relu(PF.affine(out, 512, name='fc1'))
                q_values.append(PF.affine(h, num_actions, name='output'))
        return q_values

class BootstrappedDQN:
    def __init__(self, num_actions, num_heads, batch_size, gamma, lr):
        self.num_actions = num_actions
        self.num_heads = num_heads

        # infer variable
        self.infer_obs_t = nn.Variable((1, 4, 84, 84))
        # inference output
        self.infer_qs_t = cnn_network(self.infer_obs_t, num_actions,
                                      num_heads, 'q_func')
        self.infer_all = F.sink(*self.infer_qs_t)

        # train variables
        self.obss_t = nn.Variable((batch_size, 4, 84, 84))
        self.acts_t = nn.Variable((batch_size, 1))
        self.rews_tp1 = nn.Variable((batch_size, 1))
        self.obss_tp1 = nn.Variable((batch_size, 4, 84, 84))
        self.ters_tp1 = nn.Variable((batch_size, 1))
        self.weights = nn.Variable((batch_size, num_heads))

        # training output
        qs_t = cnn_network(self.obss_t, num_actions, num_heads, 'q_func')
        qs_tp1 = cnn_network(self.obss_tp1, num_actions, num_heads, 'target')
        stacked_qs_t = F.transpose(F.stack(*qs_t), [1, 0, 2])
        stacked_qs_tp1 = F.transpose(F.stack(*qs_tp1), [1, 0, 2])

        # select one dimension
        a_one_hot = F.reshape(
            F.one_hot(self.acts_t, (num_actions,)), (-1, 1, num_actions))
        # mask output
        q_t_selected = F.sum(stacked_qs_t * a_one_hot, axis=2)
        q_tp1_best = F.max(stacked_qs_tp1, axis=2)
        q_tp1_best.need_grad = False

        # loss calculation
        y = self.rews_tp1 + gamma * q_tp1_best * (1.0 - self.ters_tp1)
        td = F.huber_loss(q_t_selected, y)
        self.loss = F.mean(F.sum(td * self.weights, axis=1))

        # optimizer
        self.solver = S.RMSprop(lr, 0.95, 1e-2)

        # weights and biases
        with nn.parameter_scope('q_func'):
            self.params = nn.get_parameters()
            self.head_params = []
            for i in range(self.num_heads):
                with nn.parameter_scope('head%d' % i):
                    self.head_params.append(nn.get_parameters())
            with nn.parameter_scope('shared'):
                self.shared_params = nn.get_parameters()
        with nn.parameter_scope('target'):
            self.target_params = nn.get_parameters()

        # set q function parameters to solver
        self.solver.set_parameters(self.params)

    def infer(self, obs_t, head):
        self.infer_obs_t.d = np.array(obs_t)
        self.infer_qs_t[head].forward(clear_buffer=True)
        return self.infer_qs_t[head].d

    def ensemble(self, obs_t):
        self.infer_obs_t.d = np.array(obs_t)
        self.infer_all.forward(clear_buffer=True)
        votes = np.zeros(self.num_actions)
        for q_value in self.infer_qs_t:
            votes[np.argmax(q_value.d[0])] += 1
        return np.argmax(votes)

    def train(self, obss_t, acts_t, rews_tp1, obss_tp1, ters_tp1, weights):
        self.obss_t.d = np.array(obss_t)
        self.acts_t.d = np.array(acts_t)
        self.rews_tp1.d = np.array(rews_tp1)
        self.obss_tp1.d = np.array(obss_tp1)
        self.ters_tp1.d = np.array(ters_tp1)
        self.weights.d = np.array(weights)
        self.loss.forward()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # gradient normalization
        for variable in self.shared_params.values():
            variable.grad /= self.num_heads
        self.solver.update()
        return self.loss.d

    def update_target(self):
        for key in self.target_params.keys():
            self.target_params[key].data.copy_from(self.params[key].data)
#-----------------------------------------------------------------------------#

#---------------------------- replay buffer ----------------------------------#
class Buffer:
    def __init__(self, maxlen=10 ** 5, batch_size=32):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=maxlen)

    def add(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1, weight):
        ter_tp1 = 1.0 if ter_tp1 else 0.0
        experience = dict(obs_t=obs_t, act_t=[act_t],
                          rew_tp1=[rew_tp1], obs_tp1=obs_tp1,
                          ter_tp1=[ter_tp1], weight=weight)
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)
#-----------------------------------------------------------------------------#

#----------------------- epsilon-greedy exploration --------------------------#
class EpsilonGreedy:
    def __init__(self, num_actions, init_value, final_value, duration):
        self.num_actions = num_actions
        self.base= init_value - final_value
        self.init_value = init_value
        self.final_value = final_value
        self.duration = duration

    def get(self, t, greedy_action):
        decay = t / self.duration
        if decay > 1.0:
            decay = 1.0
        epsilon = (1.0 - decay) * self.base + self.final_value
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        return greedy_action
#-----------------------------------------------------------------------------#

#------------------------ environment wrapper --------------------------------#
def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(gray, (210, 160))
    state = cv2.resize(state, (84, 110))
    state = state[18:102, :]
    return state

def get_deque():
    return deque(list(np.zeros((4, 84, 84), dtype=np.uint8)), maxlen=4)

class AtariWrapper:
    def __init__(self, env, episodic_life=False, render=False):
        self.env = env
        self.render = render
        self.queue = get_deque()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.episodic_life = episodic_life
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.queue.append(preprocess(obs))
        if self.render:
            self.env.render()
        if self.episodic_life:
            self.was_real_done = done
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                done = True
            self.lives = lives
        return np.array(list(self.queue)), reward, done, {}

    def reset(self):
        if not self.episodic_life or self.was_real_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        self.queue = get_deque()
        self.queue.append(preprocess(obs))
        return np.array(list(self.queue))
#-----------------------------------------------------------------------------#

#-------------------------- training loop ------------------------------------#
def pixel_to_float(obs):
    return np.array(obs, dtype=np.float32) / 255.0

def train(model, buffer):
    experiences = buffer.sample()
    obss_t = []
    acts_t = []
    rews_tp1 = []
    obss_tp1 = []
    ters_tp1 = []
    weights = []
    for experience in experiences:
        obss_t.append(experience['obs_t'])
        acts_t.append(experience['act_t'])
        rews_tp1.append(experience['rew_tp1'])
        obss_tp1.append(experience['obs_tp1'])
        ters_tp1.append(experience['ter_tp1'])
        weights.append(experience['weight'])
    return model.train(pixel_to_float(obss_t), acts_t, rews_tp1,
                       pixel_to_float(obss_tp1), ters_tp1, weights)

def train_loop(env, model, buffer, exploration, evaluate, logdir):
    monitor = Monitor(logdir)
    reward_monitor = MonitorSeries('reward', monitor, interval=1)
    loss_monitor = MonitorSeries('loss', monitor, interval=10000)
    eval_reward_monitor = MonitorSeries('eval_reward', monitor, interval=1)
    # copy parameters to target network
    model.update_target()

    step = 0
    while step <= 5 * 10 ** 7:
        obs_t = env.reset()
        ter_tp1 = False
        cumulative_reward = 0.0
        head = np.random.randint(model.num_heads)
        while not ter_tp1:
            # infer q values
            q_t = model.infer(pixel_to_float([obs_t]), head)[0]
            # epsilon-greedy exploration
            act_t = exploration.get(step, np.argmax(q_t))
            # move environment
            obs_tp1, rew_tp1, ter_tp1, _ = env.step(act_t)
            # clip reward between [-1.0, 1.0]
            clipped_rew_tp1 = np.clip(rew_tp1, -1.0, 1.0)
            # sample weights from Ber(0.5)
            weight = np.random.randint(2, size=model.num_heads)
            # store transition
            buffer.add(obs_t, act_t, clipped_rew_tp1, obs_tp1, ter_tp1, weight)

            # update parameters
            if step > 50000 and step % 4 == 0:
                loss = train(model, buffer)
                loss_monitor.add(step, loss)

            # synchronize target parameters with the latest parameters
            if step % 10000 == 0:
                model.update_target()

            # save parameters
            if step % 10 ** 6 == 0:
                path = os.path.join(logdir, 'model_{}.h5'.format(step))
                nn.save_parameters(path)
                eval_reward_monitor.add(step, evaluate(step))

            step += 1
            cumulative_reward += rew_tp1
            obs_t = obs_tp1

        # record metrics
        reward_monitor.add(step, cumulative_reward)
#-----------------------------------------------------------------------------#

def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    # atari environment
    env = AtariWrapper(gym.make(args.env), True, args.render)
    eval_env = AtariWrapper(gym.make(args.env), False, args.render)
    num_actions = env.action_space.n

    # action-value function built with neural network
    model = BootstrappedDQN(num_actions, args.num_heads, args.batch_size,
                            args.gamma, args.lr)
    if args.load is not None:
        nn.load_parameters(args.load)

    # replay buffer for experience replay
    buffer = Buffer(args.buffer_size, args.batch_size)

    # epsilon-greedy exploration
    exploration = EpsilonGreedy(num_actions, args.epsilon, 0.01,
                                args.schedule_duration)

    # evaluation loop
    def evaluate(step):
        episode = 0
        episode_rewards = []
        while episode < 10:
            obs = eval_env.reset()
            ter = False
            cumulative_reward = 0.0
            while not ter:
                act = model.ensemble(pixel_to_float([obs]))
                if np.random.random() < 0.05:
                    act = np.random.randint(num_actions)
                obs, rew, ter, _ = eval_env.step(act)
                cumulative_reward += rew
            episode_rewards.append(cumulative_reward)
            episode += 1
        return np.mean(episode_rewards)

    # prepare log directory
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = os.path.join('logs', args.logdir + '_' + date)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # start training loop
    train_loop(env, model, buffer, exploration, evaluate, logdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--num-heads', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 6)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--schedule-duration', type=int, default=10 ** 6)
    parser.add_argument('--logdir', type=str, default='dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
