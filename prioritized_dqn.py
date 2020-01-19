import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.solvers as S
import random
import argparse
import gym

from collections import deque
from nnabla.ext_utils import get_extension_context
from common.log import prepare_monitor
from common.experiment import evaluate, train
from common.exploration import LinearlyDecayEpsilonGreedy
from common.env import AtariWrapper
from dqn import DQN, q_function, pixel_to_float


#------------------------------- neural network ------------------------------#
class PrioritizedDQN(DQN):
    def _build(self):
        # infer variable
        self.infer_obs_t = nn.Variable((1, 4, 84, 84))
        # inference output
        self.infer_q_t = self.q_function(self.infer_obs_t, self.num_actions,
                                         scope='q_func')

        # train variables
        self.obss_t = nn.Variable((self.batch_size, 4, 84, 84))
        self.acts_t = nn.Variable((self.batch_size, 1))
        self.rews_tp1 = nn.Variable((self.batch_size, 1))
        self.obss_tp1 = nn.Variable((self.batch_size, 4, 84, 84))
        self.ters_tp1 = nn.Variable((self.batch_size, 1))
        self.weights = nn.Variable((self.batch_size, 1))

        # training output
        q_t = self.q_function(self.obss_t, self.num_actions, scope='q_func')
        q_tp1 = self.q_function(self.obss_tp1, self.num_actions,
                                scope='target_q_func')

        # select one dimension
        a_t_one_hot = F.one_hot(self.acts_t, (self.num_actions,))
        q_t_selected = F.sum(q_t * a_t_one_hot, axis=1, keepdims=True)
        q_tp1_best = F.max(q_tp1, axis=1, keepdims=True)

        # loss calculation
        y = self.rews_tp1 + self.gamma * q_tp1_best * (1.0 - self.ters_tp1)
        self.td = q_t_selected - y
        self.loss = F.sum(F.huber_loss(q_t_selected, y) * self.weights)
        self.loss_sink = F.sink(self.td, self.loss)

        # optimizer
        self.solver = S.RMSprop(self.lr, 0.95, 1e-2)

        # weights and biases
        with nn.parameter_scope('q_func'):
            self.params = nn.get_parameters()
        with nn.parameter_scope('target_q_func'):
            self.target_params = nn.get_parameters()

        # set q function parameters to solver
        self.solver.set_parameters(self.params)

    def train(self, obss_t, acts_t, rews_tp1, obss_tp1, ters_tp1, weights):
        self.obss_t.d = np.array(obss_t)
        self.acts_t.d = np.array(acts_t)
        self.rews_tp1.d = np.array(rews_tp1)
        self.obss_tp1.d = np.array(obss_tp1)
        self.ters_tp1.d = np.array(ters_tp1)
        self.weights.d = np.array(weights)
        self.loss_sink.forward()
        td, loss = self.td.d.copy(), self.loss.d.copy()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # gradient clipping by norm
        for name, variable in self.params.items():
            g = 10.0 * variable.g / max(np.sqrt(np.sum(variable.g ** 2)), 10.0)
            variable.g = g
        self.solver.update()
        return np.reshape(td, (-1,)), loss


class PrioritizedReplayBuffer:
    def __init__(self, maxlen=10 ** 5, batch_size=32, alpha=0.6, beta=0.4):
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.priorities = np.zeros(maxlen, dtype=np.float32)
        self.index = 0

    def append(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        # add priority
        if self.size() == 0:
            priority = 1.0
        else:
            priority = np.max(self.priorities)
        self.priorities[self.index] = priority
        self.index = (self.index + 1) % self.maxlen

        # add transition
        ter_tp1 = 1.0 if ter_tp1 else 0.0
        experience = dict(obs_t=obs_t, act_t=act_t, rew_tp1=[rew_tp1],
                          obs_tp1=obs_tp1, ter_tp1=[ter_tp1])
        self.buffer.append(experience)

    def sample(self):
        priorities = self.priorities[:self.size()]
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(self.size(), self.batch_size, p=probs)
        weights = (self.batch_size * probs[indices]) ** -self.beta
        reshaped_weights = np.expand_dims(weights, axis=-1)
        normalized_weights = reshaped_weights / np.max(weights)
        experiences = []
        for index in indices:
            experiences.append(self.buffer[index])
        return indices, experiences, normalized_weights

    def update_priorities(self, indices, tds):
        self.priorities[indices] = np.abs(tds) ** self.alpha

    def size(self):
        return len(self.buffer)


def update(model, buffer, target_update_interval):
    def _func(step):
        indices, experiences, weights = buffer.sample()
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
        td, loss = model.train(pixel_to_float(obss_t), acts_t, rews_tp1,
                               pixel_to_float(obss_tp1), ters_tp1, weights)
        buffer.update_priorities(indices, td)

        if step % target_update_interval == 0:
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
    model = PrioritizedDQN(q_function, num_actions, args.batch_size,
                           args.gamma, args.lr)
    if args.load is not None:
        nn.load_parameters(args.load)
    model.update_target()

    # replay buffer for experience replay
    buffer = PrioritizedReplayBuffer(args.buffer_size, args.batch_size,
                                     args.alpha, args.beta)

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
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 5)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--schedule-duration', type=int, default=10 ** 6)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 4)
    parser.add_argument('--update-start', type=int, default=5 * 10 ** 4)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--evaluate-interval', type=int, default=10 ** 6)
    parser.add_argument('--save-interval', type=int, default=10 ** 6)
    parser.add_argument('--final-step', type=int, default=10 ** 7)
    parser.add_argument('--logdir', type=str, default='prioritized_dqn')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
