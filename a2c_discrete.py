import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.solvers as S
import argparse
import gym
import os
import cv2

from datetime import datetime
from collections import deque
from nnabla.monitor import Monitor, MonitorSeries
from nnabla.ext_utils import get_extension_context


#------------------------------- neural network ------------------------------#
def cnn_network(obs, num_actions, scope):
    with nn.parameter_scope(scope):
        out = PF.convolution(obs, 32, (8, 8), stride=(4, 4), name='conv1')
        out = F.relu(out)
        out = PF.convolution(out, 64, (4, 4), stride=(2, 2), name='conv2')
        out = F.relu(out)
        out = PF.convolution(out, 64, (3, 3), stride=(1, 1), name='conv3')
        out = F.relu(out)
        out = PF.affine(out, 512, name='fc1')
        out = F.relu(out)
        policy = F.softmax(PF.affine(out, num_actions, name='policy'))
        value = PF.affine(out, 1, name='value')
    return policy, value


def learning_rate_scheduler(init_lr, max_iter):
    def _scheduler(step):
        lr = init_lr * (1.0 - step / max_iter)
        return max(lr, 0.0)
    return _scheduler


class A2C:
    def __init__(self, num_actions, num_envs, batch_size, v_coeff, ent_coeff,
                 lr_scheduler):
        self.infer_obs_t = nn.Variable((num_envs, 4, 84, 84))
        self.infer_pi_t, self.infer_value_t = cnn_network(
            self.infer_obs_t, num_actions, 'network')
        self.infer_t = F.sink(self.infer_pi_t, self.infer_value_t)

        self.obs_t = nn.Variable((batch_size, 4, 84, 84))
        self.actions_t = nn.Variable((batch_size, 1))
        self.returns_t = nn.Variable((batch_size, 1))
        self.advantages_t = nn.Variable((batch_size, 1))

        pi_t, value_t = cnn_network(self.obs_t, num_actions, 'network')

        # value loss
        l2loss = F.squared_error(value_t, self.returns_t)
        self.value_loss = v_coeff * F.mean(l2loss)

        # policy loss
        log_pi_t = F.log(pi_t + 1e-20)
        a_one_hot = F.one_hot(self.actions_t, (num_actions,))
        log_probs_t = F.sum(log_pi_t * a_one_hot, axis=1, keepdims=True)
        self.pi_loss = F.mean(log_probs_t * self.advantages_t)

        # KL loss
        entropy = -ent_coeff * F.mean(F.sum(pi_t * log_pi_t, axis=1))

        self.loss = self.value_loss - self.pi_loss - entropy

        self.params = nn.get_parameters()
        self.solver = S.RMSprop(lr_scheduler(0.0), 0.99, 1e-5)
        self.solver.set_parameters(self.params)
        self.lr_scheduler = lr_scheduler

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array(obs_t)
        self.infer_t.forward(clear_buffer=True)
        return self.infer_pi_t.d, np.reshape(self.infer_value_t.d, (-1,))

    def train(self, obs_t, actions_t, returns_t, advantages_t, step):
        self.obs_t.d = np.array(obs_t).reshape((-1, 4, 84, 84))
        self.actions_t.d = np.array(actions_t).reshape((-1, 1))
        self.returns_t.d = np.array(returns_t).reshape((-1, 1))
        self.advantages_t.d = np.array(advantages_t).reshape(-1, 1)
        self.loss.forward()
        pi_loss, v_loss = self.pi_loss.d.copy(), self.value_loss.d.copy()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # gradient clipping
        sum_grad = 0
        for variable in self.params.values():
            sum_grad += np.sum(variable.g ** 2)
        if np.sqrt(sum_grad) > 0.5:
            for variable in self.params.values():
                variable.g = 0.5 * variable.g / np.sqrt(sum_grad)
        self.solver.set_learning_rate(self.lr_scheduler(step))
        self.solver.update()
        return pi_loss, v_loss
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


class BatchEnv:
    def __init__(self, envs):
        self.envs = envs

    def step(self, actions):
        obs_list = []
        reward_list = []
        done_list = []
        for env, action in zip(self.envs, actions):
            obs, reward, done, _ = env.step(action)
            if done:
                obs = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(1.0 if done else 0.0)
        return np.array(obs_list), np.array(reward_list), np.array(done_list), _

    def reset(self):
        return np.array([env.reset() for env in self.envs])
#-----------------------------------------------------------------------------#

#-------------------------- training loop ------------------------------------#
def pixel_to_float(obs):
    return np.array(obs, dtype=np.float32) / 255.0


def compute_returns(gamma):
    def _compute(values_t, rewards_tp1, dones_tp1):
        V = values_t[-1]
        returns_t = []
        for reward_t, done_tp1 in reversed(list(zip(rewards_tp1, dones_tp1))):
            V = reward_t + gamma * V * (1.0 - done_tp1)
            returns_t.append(V)
        return np.array(list(reversed(returns_t)))
    return _compute


def train_loop(env, model, num_actions, update_interval, return_fn, logdir):
    monitor = Monitor(logdir)
    reward_monitor = MonitorSeries('reward', monitor, interval=1)
    policy_loss_monitor = MonitorSeries('policy_loss', monitor, interval=10000)
    value_loss_monitor = MonitorSeries('value_loss', monitor, interval=10000)
    sample_action = lambda x: np.random.choice(num_actions, p=x)

    step = 0
    obs_t = env.reset()
    cumulative_reward = np.zeros(len(env.envs), dtype=np.float32)
    obss_t, actions_t, values_t, rewards_tp1, dones_tp1 = [], [], [], [], []
    while step <= 10 ** 7:
        # inference
        probs_t, value_t = model.infer(pixel_to_float(obs_t))
        # sample actions
        action_t = list(map(sample_action, probs_t))
        # move environment
        obs_tp1, reward_tp1, done_tp1, _ = env.step(action_t)
        # clip reward between [-1.0, 1.0]
        clipped_reward_tp1 = np.clip(reward_tp1, -1.0, 1.0)

        obss_t.append(obs_t)
        actions_t.append(action_t)
        values_t.append(value_t)
        rewards_tp1.append(clipped_reward_tp1)
        dones_tp1.append(done_tp1)

        # update parameters
        if len(obss_t) == update_interval:
            values_t.append(value_t)
            returns_t = return_fn(values_t, rewards_tp1, dones_tp1)
            advantages_t = returns_t - values_t[:-1]
            policy_loss, value_loss = model.train(pixel_to_float(obss_t),
                                                  actions_t, returns_t,
                                                  advantages_t, step)
            policy_loss_monitor.add(step, policy_loss)
            value_loss_monitor.add(step, value_loss)
            obss_t.clear()
            actions_t.clear()
            values_t.clear()
            rewards_tp1.clear()
            dones_tp1.clear()

        # save parameters
        cumulative_reward += reward_tp1
        obs_t = obs_tp1

        for i, done in enumerate(done_tp1):
            step += 1
            if done:
                reward_monitor.add(step, cumulative_reward[i])
                cumulative_reward[i] = 0.0
            if step % 10 ** 6 == 0:
                path = os.path.join(logdir, 'model_{}.h5'.format(step))
                nn.save_parameters(path)
#-----------------------------------------------------------------------------#

def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    # atari environment
    num_envs = args.num_envs
    envs = [gym.make(args.env) for _ in range(num_envs)]
    batch_env = BatchEnv([AtariWrapper(env, args.render) for env in envs])
    num_actions = envs[0].action_space.n

    # action-value function built with neural network
    time_horizon = args.time_horizon
    lr_scheduler = learning_rate_scheduler(args.lr, 10 ** 7)
    model = A2C(num_actions, num_envs, num_envs * time_horizon, args.v_coeff,
                args.ent_coeff, lr_scheduler)
    if args.load is not None:
        nn.load_parameters(args.load)

    # prepare log directory
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = os.path.join('logs', args.logdir + '_' + date)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # start training loop
    return_fn = compute_returns(args.gamma)
    train_loop(batch_env, model, num_actions, time_horizon, return_fn, logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--time-horizon', type=int, default=5)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--v-coeff', type=float, default=0.5)
    parser.add_argument('--ent-coeff', type=float, default=0.01)
    parser.add_argument('--logdir', type=str, default='a2c')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
