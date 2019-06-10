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
from nnabla.initializer import ConstantInitializer, UniformInitializer
from nnabla.monitor import Monitor, MonitorSeries
from nnabla.ext_utils import get_extension_context
from nnabla.parameter import get_parameter_or_create


#------------------------------- neural network ------------------------------#
def noisy_layer(x, out_size, eps_w, eps_b, name):
    inpt_size = x.shape[1]
    root_p = np.sqrt(inpt_size)
    mu_init = UniformInitializer((-1.0 / root_p, 1.0 / root_p))
    sig_init = ConstantInitializer(0.5 / root_p)
    with nn.parameter_scope(name):
        mu_w = get_parameter_or_create('mu_w', (inpt_size, out_size), mu_init)
        sig_w = get_parameter_or_create('sig_w', (inpt_size, out_size), sig_init)
        mu_b = get_parameter_or_create('mu_b', (out_size,), mu_init)
        sig_b = get_parameter_or_create('sig_b', (out_size,), sig_init)
    return F.affine(x, mu_w + sig_w * eps_w, mu_b + sig_b * eps_b)


def sample_noise(inpt_size, out_size):
    _f = lambda x: np.sign(x) * np.sqrt(np.abs(x))
    noise = _f(np.random.normal(0.0, 1.0, size=inpt_size + out_size))
    eps_w = np.matmul(np.reshape(noise[:inpt_size], (-1, 1)),
                      np.reshape(noise[inpt_size:], (1, -1)))
    eps_b = noise[inpt_size:]
    return eps_w, eps_b


def cnn_network(obs, num_actions, epsilon_w, epsilon_b, scope):
    with nn.parameter_scope(scope):
        out = PF.convolution(obs, 32, (8, 8), stride=(4, 4), name='conv1')
        out = F.relu(out)
        out = PF.convolution(out, 64, (4, 4), stride=(2, 2), name='conv2')
        out = F.relu(out)
        out = PF.convolution(out, 64, (3, 3), stride=(1, 1), name='conv3')
        out = F.relu(out)
        out = PF.affine(out, 512, name='fc1')
        out = F.relu(out)
        return noisy_layer(out, num_actions, epsilon_w, epsilon_b, 'output')


class NoisyNetDQN:
    def __init__(self, num_actions, batch_size, gamma, lr):
        self.eps_w = nn.Variable((512, num_actions))
        self.eps_b = nn.Variable((num_actions,))
        self.t_eps_w = nn.Variable((512, num_actions))
        self.t_eps_b = nn.Variable((num_actions,))

        # inference
        self.infer_obs_t = infer_obs_t = nn.Variable((1, 4, 84, 84))
        self.infer_q_t = cnn_network(infer_obs_t, num_actions, self.eps_w,
                                     self.eps_b, 'q_func')

        # training
        self.obs_t = obs_t = nn.Variable((batch_size, 4, 84, 84))
        self.actions_t = actions_t = nn.Variable((batch_size, 1))
        self.rewards_tp1 = rewards_tp1 = nn.Variable((batch_size, 1))
        self.obs_tp1 = obs_tp1 = nn.Variable((batch_size, 4, 84, 84))
        self.dones_tp1 = dones_tp1 = nn.Variable((batch_size, 1))

        # training output
        q_t = cnn_network(obs_t, num_actions, self.eps_w, self.eps_b, 'q_func')
        q_tp1 = cnn_network(obs_tp1, num_actions, self.t_eps_w, self.t_eps_b,
                            'target_q_func')

        # select one dimension
        a_one_hot = F.one_hot(actions_t, (num_actions,))
        q_t_selected = F.sum(q_t * a_one_hot, axis=1, keepdims=True)
        q_tp1_best = F.max(q_tp1, axis=1, keepdims=True)

        # loss calculation
        y = self.rewards_tp1 + gamma * q_tp1_best * (1.0 - self.dones_tp1)
        # prevent unnecessary gradient calculation
        unlinked_y = y.get_unlinked_variable(need_grad=False)
        self.loss = F.mean(F.huber_loss(q_t_selected, unlinked_y))

        # weights and biases
        with nn.parameter_scope('q_func'):
            self.params = nn.get_parameters()
        with nn.parameter_scope('target_q_func'):
            self.target_params = nn.get_parameters()

        # optimizer
        self.solver = S.RMSprop(lr, 0.95, 1e-2)
        self.solver.set_parameters(self.params)

    def infer(self, obs_t):
        self.eps_w.d, self.eps_b.d = sample_noise(*self.eps_w.shape)
        self.infer_obs_t.d = np.array(obs_t)
        self.infer_q_t.forward(clear_buffer=True)
        return self.infer_q_t.d

    def train(self, obs_t, actions_t, rewards_tp1, obs_tp1, dones_tp1):
        self.eps_w.d, self.eps_b.d = sample_noise(*self.eps_w.shape)
        self.t_eps_w.d, self.t_eps_b.d = sample_noise(*self.eps_w.shape)
        self.obs_t.d = np.array(obs_t)
        self.actions_t.d = np.array(actions_t)
        self.rewards_tp1.d = np.array(rewards_tp1)
        self.obs_tp1.d = np.array(obs_tp1)
        self.dones_tp1.d = np.array(dones_tp1)
        self.loss.forward()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # gradient clipping by norm
        for name, variable in self.params.items():
            g = 10.0 * variable.g / np.sqrt(np.sum(variable.g ** 2))
            variable.g = g
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

    def add(self, obs_t, action_t, reward_tp1, obs_tp1, done_tp1):
        experience = dict(obs_t=obs_t, action_t=[action_t],
                          reward_tp1=[reward_tp1], obs_tp1=obs_tp1,
                          done_tp1=[done_tp1])
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)
#-----------------------------------------------------------------------------#

#------------------------ environment wrapper --------------------------------#
def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(cv2.resize(gray, (210, 160)), (84, 110))
    state = state[18:102, :]
    return state


def get_deque():
    return deque(list(np.zeros((4, 84, 84), dtype=np.uint8)), maxlen=4)


class AtariWrapper:
    def __init__(self, env, render=False):
        self.env = env
        self.render = render
        self.queue = get_deque()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.queue.append(preprocess(obs))
        if self.render:
            self.env.render()
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return np.array(list(self.queue)), reward, done, {}

    def reset(self):
        if self.was_real_done:
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
    obs_t = []
    actions_t = []
    rewards_tp1 = []
    obs_tp1 = []
    dones_tp1 = []
    for experience in experiences:
        obs_t.append(experience['obs_t'])
        actions_t.append(experience['action_t'])
        rewards_tp1.append(experience['reward_tp1'])
        obs_tp1.append(experience['obs_tp1'])
        dones_tp1.append(experience['done_tp1'])
    return model.train(pixel_to_float(obs_t), actions_t, rewards_tp1,
                       pixel_to_float(obs_tp1), dones_tp1)


def train_loop(env, model, buffer, logdir):
    monitor = Monitor(logdir)
    reward_monitor = MonitorSeries('reward', monitor, interval=1)
    loss_monitor = MonitorSeries('loss', monitor, interval=10000)
    # copy parameters to target network
    model.update_target()

    step = 0
    while step <= 10 ** 7:
        obs_t = env.reset()
        reward_t = 0.0
        done_tp1 = False
        cumulative_reward = 0.0

        while not done_tp1:
            # determine action
            action_t = np.argmax(model.infer(pixel_to_float([obs_t]))[0])
            # move environment
            obs_tp1, reward_tp1, done_tp1, _ = env.step(action_t)
            # clip reward between [-1.0, 1.0]
            clipped_reward_tp1 = np.clip(reward_tp1, -1.0, 1.0)
            # store transition
            buffer.add(obs_t, action_t, clipped_reward_tp1, obs_tp1, done_tp1)

            # update parameters
            if step > 10000 and step % 4 == 0:
                loss = train(model, buffer)
                loss_monitor.add(step, loss)

            # synchronize target parameters with the latest parameters
            if step % 10000 == 0:
                model.update_target()

            # save parameters
            if step % 10 ** 6 == 0:
                path = os.path.join(logdir, 'model_{}.h5'.format(step))
                nn.save_parameters(path)

            step += 1
            cumulative_reward += reward_tp1
            obs_t = obs_tp1

        # record metrics
        reward_monitor.add(step, cumulative_reward)
#-----------------------------------------------------------------------------#

def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    # atari environment
    env = AtariWrapper(gym.make(args.env), args.render)
    num_actions = env.action_space.n

    # action-value function built with neural network
    model = NoisyNetDQN(num_actions, args.batch_size, args.gamma, args.lr)
    if args.load is not None:
        nn.load_parameters(args.load)

    # replay buffer for experience replay
    buffer = Buffer(args.buffer_size, args.batch_size)

    # prepare log directory
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = os.path.join('logs', args.logdir + '_' + date)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # start training loop
    train_loop(env, model, buffer, logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 5)
    parser.add_argument('--logdir', type=str, default='noisynet_dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
