import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.solvers as S
import argparse
import gym

from nnabla.ext_utils import get_extension_context
from common.buffer import ReplayBuffer
from common.log import prepare_monitor
from common.experiment import evaluate, train
from common.exploration import LinearlyDecayEpsilonGreedy
from common.env import AtariWrapper
from common.helper import clip_by_value
from common.network import nature_head


def pixel_to_float(obs):
    return np.array(obs, dtype=np.float32) / 255.0


def q_function(obs, num_actions, scope):
    with nn.parameter_scope(scope):
        out = nature_head(obs)
        return PF.affine(out, num_actions, name='output')


class DQN:
    def __init__(self, q_function, num_actions, batch_size, gamma, lr):
        self.q_function = q_function
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self._build()

    def _build(self):
        # infer variable
        self.infer_obs_t = infer_obs_t = nn.Variable((1, 4, 84, 84))
        # inference output
        self.infer_q_t = self.q_function(infer_obs_t, self.num_actions,
                                         scope='q_func')

        # train variables
        self.obss_t = nn.Variable((self.batch_size, 4, 84, 84))
        self.acts_t = nn.Variable((self.batch_size, 1))
        self.rews_tp1 = nn.Variable((self.batch_size, 1))
        self.obss_tp1 = nn.Variable((self.batch_size, 4, 84, 84))
        self.ters_tp1 = nn.Variable((self.batch_size, 1))

        # training output
        q_t = self.q_function(self.obss_t, self.num_actions, scope='q_func')
        q_tp1 = self.q_function(self.obss_tp1, self.num_actions,
                                scope='target_q_func')

        # select one dimension
        a_one_hot = F.one_hot(self.acts_t, (self.num_actions,))
        q_t_selected = F.sum(q_t * a_one_hot, axis=1, keepdims=True)
        q_tp1_best = F.max(q_tp1, axis=1, keepdims=True)

        # reward clipping
        clipped_rews_tp1 = clip_by_value(self.rews_tp1, -1.0, 1.0)

        # loss calculation
        y = clipped_rews_tp1 + self.gamma * q_tp1_best * (1.0 - self.ters_tp1)
        self.loss = F.mean(F.huber_loss(q_t_selected, y))

        # optimizer
        self.solver = S.RMSprop(self.lr, 0.95, 1e-2)

        # weights and biases
        with nn.parameter_scope('q_func'):
            self.params = nn.get_parameters()
        with nn.parameter_scope('target_q_func'):
            self.target_params = nn.get_parameters()

        # set q function parameters to solver
        self.solver.set_parameters(self.params)

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array(pixel_to_float([obs_t]))
        self.infer_q_t.forward(clear_buffer=True)
        return np.argmax(self.infer_q_t.d[0])

    def evaluate(self, obs_t):
        if np.random.random() < 0.05:
            return np.random.randint(self.num_actions)
        return self.infer(obs_t)

    def train(self, obss_t, acts_t, rews_tp1, obss_tp1, ters_tp1):
        self.obss_t.d = np.array(obss_t)
        self.acts_t.d = np.array(acts_t)
        self.rews_tp1.d = np.array(rews_tp1)
        self.obss_tp1.d = np.array(obss_tp1)
        self.ters_tp1.d = np.array(ters_tp1)
        self.loss.forward()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        self.solver.clip_grad_by_norm(10.0)
        self.solver.update()
        return self.loss.d

    def update_target(self):
        for key in self.target_params.keys():
            self.target_params[key].data.copy_from(self.params[key].data)

    def reset(self, step):
        pass


def update(model, buffer, target_update_inteval):
    def _func(step):
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
        loss = model.train(pixel_to_float(obss_t), acts_t, rews_tp1,
                           pixel_to_float(obss_tp1), ters_tp1)

        if step % target_update_inteval == 0:
            model.update_target()

        return [loss]
    return _func


def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    # atari environment
    env = AtariWrapper(gym.make(args.env), args.seed, episodic=True)
    eval_env = AtariWrapper(gym.make(args.env), 50, episodic=False)
    num_actions = env.action_space.n

    # action-value function built with neural network
    model = DQN(q_function, num_actions, args.batch_size, args.gamma, args.lr)
    if args.load is not None:
        nn.load_parameters(args.load)
    model.update_target()

    buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    exploration = LinearlyDecayEpsilonGreedy(num_actions, args.epsilon, 0.1,
                                             args.schedule_duration)

    monitor = prepare_monitor(args.logdir)

    update_fn = update(model, buffer, args.target_update_interval)

    eval_fn = evaluate(eval_env, model, render=args.render)

    train(env, model, buffer, exploration, monitor, update_fn, eval_fn,
          args.final_step, args.update_start, args.update_interval,
          args.save_interval, args.evaluate_interval, ['loss'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 5)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--schedule-duration', type=int, default=10 ** 6)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 4)
    parser.add_argument('--update-start', type=int, default=5 * 10 ** 4)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--evaluate-interval', type=int, default=10 ** 6)
    parser.add_argument('--save-interval', type=int, default=10 ** 6)
    parser.add_argument('--final-step', type=int, default=10 ** 7)
    parser.add_argument('--logdir', type=str, default='dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
