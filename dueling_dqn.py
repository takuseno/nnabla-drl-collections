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
from dqn import DQN, train_loop


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
        advantages = PF.affine(out, num_actions, name='advantage')
        value = PF.affine(out, 1, name='value')
        baseline = F.mean(advantages, axis=1, keepdims=True)
        return value + advantages - baseline


class DuelingDQN(DQN):
    def cnn_network(self, *args, **kwargs):
        return cnn_network(*args, **kwargs)
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
    model = DuelingDQN(num_actions, args.batch_size, args.gamma, args.lr)
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
    parser.add_argument('--logdir', type=str, default='dueling_dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
