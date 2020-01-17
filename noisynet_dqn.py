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
from common.buffer import ReplayBuffer
from common.log import prepare_directory
from common.experiment import evaluate
from common.helper import clip_by_value
from common.env import AtariWrapper
from common.exploration import ConstantEpsilonGreedy
from dqn import DQN, train_loop, pixel_to_float


#------------------------------- neural network ------------------------------#
def sample_noise(inpt_size, out_size):
    _f = lambda x: F.sign(x) * F.pow_scalar(F.abs(x), 0.5)
    noise = _f(F.randn(shape=(inpt_size + out_size,)))
    eps_w = F.batch_matmul(F.reshape(noise[:inpt_size], (1, -1)),
                           F.reshape(noise[inpt_size:], (1, -1)), True)
    eps_b = noise[inpt_size:]
    return eps_w, eps_b


def noisy_layer(x, out_size, name):
    inpt_size = x.shape[1]
    root_p = np.sqrt(inpt_size)
    mu_init = UniformInitializer((-1.0 / root_p, 1.0 / root_p))
    sig_init = ConstantInitializer(0.5 / root_p)
    eps_w, eps_b = sample_noise(inpt_size, out_size)
    with nn.parameter_scope(name):
        mu_w = get_parameter_or_create('mu_w', (inpt_size, out_size), mu_init)
        sig_w = get_parameter_or_create('sig_w', (inpt_size, out_size), sig_init)
        mu_b = get_parameter_or_create('mu_b', (out_size,), mu_init)
        sig_b = get_parameter_or_create('sig_b', (out_size,), sig_init)
    return F.affine(x, mu_w + sig_w * eps_w, mu_b + sig_b * eps_b)


def cnn_network(obs, num_actions, scope):
    with nn.parameter_scope(scope):
        out = PF.convolution(obs, 32, (8, 8), stride=(4, 4), name='conv1')
        out = F.relu(out)
        out = PF.convolution(out, 64, (4, 4), stride=(2, 2), name='conv2')
        out = F.relu(out)
        out = PF.convolution(out, 64, (3, 3), stride=(1, 1), name='conv3')
        out = F.reshape(F.relu(out), (obs.shape[0], -1))
        out = noisy_layer(out, 512, 'fc1')
        out = F.relu(out)
        return noisy_layer(out, num_actions, 'output')


class NoisyNetDQN(DQN):
    def evaluate(self, obs_t):
        return self.infer(pixel_to_float(obs_t))

    def cnn_network(self, *args, **kwargs):
        return cnn_network(*args, **kwargs)
#-----------------------------------------------------------------------------#

def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    # atari environment
    env = AtariWrapper(gym.make(args.env), args.seed, episodic=True)
    eval_env = AtariWrapper(gym.make(args.env), args.seed, episodic=False)
    num_actions = env.action_space.n

    # action-value function built with neural network
    model = NoisyNetDQN(num_actions, args.batch_size, args.gamma, args.lr)
    if args.load is not None:
        nn.load_parameters(args.load)

    buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    exploration = ConstantEpsilonGreedy(num_actions, 0.0)

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
    parser.add_argument('--logdir', type=str, default='noisynet_dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--final-step', type=int, default=10 ** 7)
    parser.add_argument('--target-update', type=int, default=10 ** 4)
    parser.add_argument('--learning-start', type=int, default=5 * 10 ** 4)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
