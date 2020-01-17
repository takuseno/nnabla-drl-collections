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
from common.helper import clip_by_value
from common.env import AtariWrapper
from dqn import DQN, train_loop


#------------------------------- neural network ------------------------------#
class DoubleDQN(DQN):
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
        q_tp1 = self.cnn_network(self.obss_tp1, num_actions, scope='q_func')
        q_tp1_target = self.cnn_network(self.obss_tp1, num_actions,
                                        scope='target_q_func')

        # select one dimension
        a_t_one_hot = F.one_hot(self.acts_t, (num_actions,))
        q_t_selected = F.sum(q_t * a_t_one_hot, axis=1, keepdims=True)
        _, a_tp1 = F.max(q_tp1, axis=1, keepdims=True, with_index=True)
        a_tp1_one_hot = F.one_hot(a_tp1, (num_actions,))
        q_tp1_best = F.max(q_tp1_target * a_tp1_one_hot, axis=1, keepdims=True)

        # reward clipping
        clipped_rews_tp1 = clip_by_value(self.rews_tp1, -1.0, 1.0)

        # loss calculation
        y = clipped_rews_tp1 + gamma * q_tp1_best * (1.0 - self.ters_tp1)
        y.need_grad = False
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
    model = DoubleDQN(num_actions, args.batch_size, args.gamma, args.lr)
    if args.load is not None:
        nn.load_parameters(args.load)

    buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    exploration = LinearlyDecayEpsilonGreedy(num_actions, args.epsilon, 0.1,
                                             args.schedule_duration)

    logdir = prepare_directory(args.logdir)

    eval_fn = evaluate(eval_env, model, render=args.render)

    train_loop(env, model, buffer, exploration, logdir, eval_fn, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 5)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--schedule-duration', type=int, default=10 ** 6)
    parser.add_argument('--final-step', type=int, default=10 ** 7)
    parser.add_argument('--target-update', type=int, default=10 ** 4)
    parser.add_argument('--learning-start', type=int, default=5 * 10 ** 4)
    parser.add_argument('--logdir', type=str, default='double_dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
