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
from common.buffer import ReplayBuffer
from common.log import prepare_directory
from common.experiment import evaluate
from common.exploration import LinearlyDecayEpsilonGreedy
from common.env import AtariWrapper
from common.helper import clip_by_value


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
        return PF.affine(out, num_actions, name='output')

class DQN:
    def __init__(self, num_actions, batch_size, gamma, lr):
        self.num_actions = num_actions

        # infer variable
        self.infer_obs_t = infer_obs_t = nn.Variable((1, 4, 84, 84))
        # inference output
        self.infer_q_t = self.cnn_network(infer_obs_t, num_actions,
                                          scope='q_func')

        # train variables
        self.obss_t = nn.Variable((batch_size, 4, 84, 84))
        self.acts_t = nn.Variable((batch_size, 1))
        self.rews_tp1 = nn.Variable((batch_size, 1))
        self.obss_tp1 = nn.Variable((batch_size, 4, 84, 84))
        self.ters_tp1 = nn.Variable((batch_size, 1))

        # training output
        q_t = self.cnn_network(self.obss_t, num_actions, scope='q_func')
        q_tp1 = self.cnn_network(self.obss_tp1, num_actions,
                                 scope='target_q_func')

        # select one dimension
        a_one_hot = F.one_hot(self.acts_t, (num_actions,))
        q_t_selected = F.sum(q_t * a_one_hot, axis=1, keepdims=True)
        q_tp1_best = F.max(q_tp1, axis=1, keepdims=True)

        # reward clipping
        clipped_rews_tp1 = clip_by_value(self.rews_tp1, -1.0, 1.0)

        # loss calculation
        y = clipped_rews_tp1 + gamma * q_tp1_best * (1.0 - self.ters_tp1)
        self.loss = F.mean(F.huber_loss(q_t_selected, y))

        # optimizer
        self.solver = S.RMSprop(lr, 0.95, 1e-2)

        # weights and biases
        with nn.parameter_scope('q_func'):
            self.params = nn.get_parameters()
        with nn.parameter_scope('target_q_func'):
            self.target_params = nn.get_parameters()

        # set q function parameters to solver
        self.solver.set_parameters(self.params)

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array([obs_t])
        self.infer_q_t.forward(clear_buffer=True)
        return np.argmax(self.infer_q_t.d[0])

    def evaluate(self, obs_t):
        if np.random.random() < 0.05:
            return np.random.randint(self.num_actions)
        return self.infer(pixel_to_float(obs_t))

    def train(self, obss_t, acts_t, rews_tp1, obss_tp1, ters_tp1):
        self.obss_t.d = np.array(obss_t)
        self.acts_t.d = np.array(acts_t)
        self.rews_tp1.d = np.array(rews_tp1)
        self.obss_tp1.d = np.array(obss_tp1)
        self.ters_tp1.d = np.array(ters_tp1)
        self.loss.forward()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # gradient clipping by norm
        for name, variable in self.params.items():
            g = 10.0 * variable.g / max(np.sqrt(np.sum(variable.g ** 2)), 10.0)
            variable.g = g
        self.solver.update()
        return self.loss.d

    def update_target(self):
        for key in self.target_params.keys():
            self.target_params[key].data.copy_from(self.params[key].data)

    def cnn_network(self, *args, **kwargs):
        return cnn_network(*args, **kwargs)
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
    for experience in experiences:
        obss_t.append(experience['obs_t'])
        acts_t.append(experience['act_t'])
        rews_tp1.append(experience['rew_tp1'])
        obss_tp1.append(experience['obs_tp1'])
        ters_tp1.append(experience['ter_tp1'])
    return model.train(pixel_to_float(obss_t), acts_t, rews_tp1,
                       pixel_to_float(obss_tp1), ters_tp1)

def train_loop(env, model, buffer, exploration, logdir, eval_fn, args):
    monitor = Monitor(logdir)
    reward_monitor = MonitorSeries('reward', monitor, interval=1)
    eval_reward_monitor = MonitorSeries('eval_reward', monitor, interval=1)
    loss_monitor = MonitorSeries('loss', monitor, interval=10000)
    # copy parameters to target network
    model.update_target()

    step = 0
    while step <= args.final_step:
        obs_t = env.reset()
        rew_t = 0.0
        ter_tp1 = False
        cumulative_reward = 0.0
        while not ter_tp1:
            # infer q values
            act_t = model.infer(pixel_to_float(obs_t))
            # epsilon-greedy exploration
            act_t = exploration.get(step, act_t)
            # move environment
            obs_tp1, rew_tp1, ter_tp1, _ = env.step(act_t)
            # store transition
            buffer.append(obs_t, [act_t], rew_tp1, obs_tp1, ter_tp1)

            # update parameters
            if step > args.learning_start and step % 4 == 0:
                loss = train(model, buffer)
                loss_monitor.add(step, loss)

            # synchronize target parameters with the latest parameters
            if step % args.target_update == 0:
                model.update_target()

            # save parameters
            if step % 10 ** 6 == 0:
                path = os.path.join(logdir, 'model_{}.h5'.format(step))
                nn.save_parameters(path)
                eval_reward_monitor.add(step, np.mean(eval_fn()))

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
    env = AtariWrapper(gym.make(args.env), args.seed, episodic=True)
    eval_env = AtariWrapper(gym.make(args.env), 50, episodic=False)
    num_actions = env.action_space.n

    # action-value function built with neural network
    model = DQN(num_actions, args.batch_size, args.gamma, args.lr)
    if args.load is not None:
        nn.load_parameters(args.load)

    buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    exploration = LinearlyDecayEpsilonGreedy(num_actions, args.epsilon, 0.1,
                                             args.schedule_duration)

    logdir = prepare_directory(args.logdir)

    eval_fn = evaluate(eval_env, model, render=args.render)

    # start training loop
    train_loop(env, model, buffer, exploration, logdir, eval_fn, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 5)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--schedule-duration', type=int, default=10 ** 6)
    parser.add_argument('--target-update', type=int, default=10 ** 4)
    parser.add_argument('--learning-start', type=int, default=5 * 10 ** 4)
    parser.add_argument('--final-step', type=int, default=10 ** 7)
    parser.add_argument('--logdir', type=str, default='dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
