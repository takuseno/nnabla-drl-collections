import nnabla as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gym

from nnabla.ext_utils import get_extension_context
from categorical_dqn import CategoricalDQN
from categorical_dqn import AtariWrapper
from categorical_dqn import pixel_to_float


def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    env = AtariWrapper(gym.make(args.env), True)
    num_actions = env.action_space.n

    model = CategoricalDQN(
        num_actions, args.min_v, args.max_v, args.num_bins, 1, 0.99, 2.5e-4)

    if args.load is not None:
        nn.load_parameters(args.load)

    delta_z = (args.max_v - args.min_v) / (args.num_bins - 1)
    dist = np.arange(args.num_bins) * delta_z + args.min_v

    # visualize categorical probabilities of returns
    def visualize(probs, action):
        fig.clf()
        plt.bar(dist, probs[action])
        plt.xlabel('return')
        plt.ylabel('probability')
        fig.canvas.draw()

    fig = plt.figure()
    fig.show()
    while True:
        obs = env.reset()
        done = False
        cumulative_reward = 0
        while not done:
            q, probs = model.infer(pixel_to_float([obs]))
            if np.random.random() > 0.05:
                action = np.argmax(q)
            else:
                action = np.random.randint(num_actions)
            visualize(probs, action)
            obs, reward, done, _ = env.step(action)
            cumulative_reward += reward
        print(cumulative_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--min-v', type=float, default=-10.0)
    parser.add_argument('--max-v', type=float, default=10.0)
    parser.add_argument('--num-bins', type=int, default=51)
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    main(args)
