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
def cnn_network(obs, num_actions, min_v, max_v, num_bins, scope):
    with nn.parameter_scope(scope):
        out = PF.convolution(obs, 32, (8, 8), stride=(4, 4), name='conv1')
        out = F.relu(out)
        out = PF.convolution(out, 64, (4, 4), stride=(2, 2), name='conv2')
        out = F.relu(out)
        out = PF.convolution(out, 64, (3, 3), stride=(1, 1), name='conv3')
        out = F.relu(out)
        out = PF.affine(out, 512, name='fc1')
        out = F.relu(out)
        out = PF.affine(out, num_actions * num_bins, name='output')
        out = F.reshape(out, (-1, num_actions, num_bins))
    probs = F.exp(out) / F.sum(F.exp(out), axis=2, keepdims=True)
    dists = F.arange(0, num_bins) * (max_v - min_v) / (num_bins - 1) + min_v
    values = F.sum(probs * F.reshape(dists, (1, 1, num_bins)), axis=2)
    return values, probs, F.reshape(dists, (-1, 1))


class CategoricalDQN(DQN):
    def __init__(self, num_actions, min_v, max_v, num_bins, batch_size,
                 gamma, lr):
        self.num_actions = num_actions

        # infer variable
        self.infer_obs_t = infer_obs_t = nn.Variable((1, 4, 84, 84))
        # inference output
        self.infer_q_t,\
        self.infer_probs_t, _ = self.cnn_network(infer_obs_t, num_actions,
                                                 min_v, max_v, num_bins,
                                                 'q_func')
        self.infer_t = F.sink(self.infer_q_t, self.infer_probs_t)

        # train variables
        self.obss_t = nn.Variable((batch_size, 4, 84, 84))
        self.acts_t = nn.Variable((batch_size, 1))
        self.rews_tp1 = nn.Variable((batch_size, 1))
        self.obss_tp1 = nn.Variable((batch_size, 4, 84, 84))
        self.ters_tp1 = nn.Variable((batch_size, 1))

        # training output
        q_t, probs_t, dists = self.cnn_network(self.obss_t, num_actions, min_v,
                                               max_v, num_bins, 'q_func')
        q_tp1, probs_tp1, _ = self.cnn_network(self.obss_tp1, num_actions,
                                               min_v, max_v, num_bins,
                                               'target_q_func')

        expand_last = lambda x: F.reshape(x, x.shape + (1,))
        flat = lambda x: F.reshape(x, (-1, 1))

        # extract selected dimension
        a_t_one_hot = expand_last(F.one_hot(self.acts_t, (num_actions,)))
        probs_t_selected = F.max(probs_t * a_t_one_hot, axis=1)
        # extract max dimension
        _, indices = F.max(q_tp1, axis=1, keepdims=True, with_index=True)
        a_tp1_one_hot = expand_last(F.one_hot(indices, (num_actions,)))
        probs_tp1_best = F.max(probs_tp1 * a_tp1_one_hot, axis=1)

        # clipping reward
        clipped_rews_tp1 = clip_by_value(self.rews_tp1, -1.0, 1.0)

        disc_q_tp1 = gamma * F.reshape(dists, (1, -1)) * (1.0 - self.ters_tp1)
        t_z = clip_by_value(clipped_rews_tp1 + disc_q_tp1, min_v, max_v)

        # update indices
        b = (t_z - min_v) / ((max_v - min_v) / (num_bins - 1))
        l = F.floor(b)
        l_mask = F.reshape(F.one_hot(flat(l), (num_bins,)),
                           (-1, num_bins, num_bins))
        u = F.ceil(b)
        u_mask = F.reshape(F.one_hot(flat(u), (num_bins,)),
                           (-1, num_bins, num_bins))

        m_l = expand_last(probs_tp1_best * (1 - (b - l)))
        m_u = expand_last(probs_tp1_best * (b - l))
        m = F.sum(m_l * l_mask + m_u * u_mask, axis=1)
        m.need_grad = False

        self.loss = -F.mean(F.sum(m * F.log(probs_t_selected + 1e-20), axis=1))

        # optimizer
        self.solver = S.RMSprop(lr, 0.95, 1e-2)

        # weights and biases
        with nn.parameter_scope('q_func'):
            self.params = nn.get_parameters()
        with nn.parameter_scope('target_q_func'):
            self.target_params = nn.get_parameters()

        # set q function parameters to solver
        self.solver.set_parameters(self.params)

    def cnn_network(self, *args, **kwargs):
        return cnn_network(*args, **kwargs)
#-----------------------------------------------------------------------------#

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
    model = CategoricalDQN(num_actions, args.min_v, args.max_v, args.num_bins,
                           args.batch_size, args.gamma, args.lr)
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
    parser.add_argument('--min-v', type=float, default=-10.0)
    parser.add_argument('--max-v', type=float, default=10.0)
    parser.add_argument('--num-bins', type=int, default=51)
    parser.add_argument('--logdir', type=str, default='categorical_dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
