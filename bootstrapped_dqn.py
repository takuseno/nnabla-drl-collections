import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.solvers as S
import argparse
import gym

from collections import deque
from nnabla.ext_utils import get_extension_context
from common.log import prepare_monitor
from common.experiment import evaluate, train
from common.exploration import LinearlyDecayEpsilonGreedy
from common.helper import clip_by_value
from common.env import AtariWrapper
from dqn import pixel_to_float


def q_function(obs, num_actions, num_heads, scope):
    with nn.parameter_scope(scope):
        with nn.parameter_scope('shared'):
            out = PF.convolution(obs, 32, (8, 8), stride=(4, 4), name='conv1')
            out = F.relu(out)
            out = PF.convolution(out, 64, (4, 4), stride=(2, 2), name='conv2')
            out = F.relu(out)
            out = PF.convolution(out, 64, (3, 3), stride=(1, 1), name='conv3')
            out = F.relu(out)
        q_values = []
        for i in range(num_heads):
            with nn.parameter_scope('head%d' % i):
                h = F.relu(PF.affine(out, 512, name='fc1'))
                q_values.append(PF.affine(h, num_actions, name='output'))
        return q_values


class BootstrappedDQN:
    def __init__(self,
                 q_function,
                 num_actions,
                 num_heads,
                 batch_size,
                 gamma,
                 lr):
        self.q_function = q_function
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self._build()

    def _build(self):
        # infer variable
        self.infer_obs_t = nn.Variable((1, 4, 84, 84))
        # inference output
        self.infer_qs_t = self.q_function(self.infer_obs_t, self.num_actions,
                                          self.num_heads, 'q_func')
        self.infer_all = F.sink(*self.infer_qs_t)

        # train variables
        self.obss_t = nn.Variable((self.batch_size, 4, 84, 84))
        self.acts_t = nn.Variable((self.batch_size, 1))
        self.rews_tp1 = nn.Variable((self.batch_size, 1))
        self.obss_tp1 = nn.Variable((self.batch_size, 4, 84, 84))
        self.ters_tp1 = nn.Variable((self.batch_size, 1))
        self.weights = nn.Variable((self.batch_size, self.num_heads))

        # training output
        qs_t = self.q_function(self.obss_t, self.num_actions, self.num_heads,
                               'q_func')
        qs_tp1 = q_function(self.obss_tp1, self.num_actions, self.num_heads,
                            'target')
        stacked_qs_t = F.transpose(F.stack(*qs_t), [1, 0, 2])
        stacked_qs_tp1 = F.transpose(F.stack(*qs_tp1), [1, 0, 2])

        # select one dimension
        a_one_hot = F.reshape(F.one_hot(self.acts_t, (self.num_actions,)),
                              (-1, 1, self.num_actions))
        # mask output
        q_t_selected = F.sum(stacked_qs_t * a_one_hot, axis=2)
        q_tp1_best = F.max(stacked_qs_tp1, axis=2)
        q_tp1_best.need_grad = False

        # reward clipping
        clipped_rews_tp1 = clip_by_value(self.rews_tp1, -1.0, 1.0)

        # loss calculation
        y = clipped_rews_tp1 + self.gamma * q_tp1_best * (1.0 - self.ters_tp1)
        td = F.huber_loss(q_t_selected, y)
        self.loss = F.mean(F.sum(td * self.weights, axis=1))

        # optimizer
        self.solver = S.RMSprop(self.lr, 0.95, 1e-2)

        # weights and biases
        with nn.parameter_scope('q_func'):
            self.params = nn.get_parameters()
            self.head_params = []
            for i in range(self.num_heads):
                with nn.parameter_scope('head%d' % i):
                    self.head_params.append(nn.get_parameters())
            with nn.parameter_scope('shared'):
                self.shared_params = nn.get_parameters()
        with nn.parameter_scope('target'):
            self.target_params = nn.get_parameters()

        # set q function parameters to solver
        self.solver.set_parameters(self.params)

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array(pixel_to_float([obs_t]))
        self.infer_qs_t[self.current_head].forward(clear_buffer=True)
        return np.argmax(self.infer_qs_t[self.current_head].d[0])

    def evaluate(self, obs_t):
        if np.random.random() < 0.05:
            return np.random.randint(self.num_actions)
        self.infer_obs_t.d = np.array(pixel_to_float([obs_t]))
        self.infer_all.forward(clear_buffer=True)
        votes = np.zeros(self.num_actions)
        for q_value in self.infer_qs_t:
            votes[np.argmax(q_value.d[0])] += 1
        return np.argmax(votes)

    def train(self, obss_t, acts_t, rews_tp1, obss_tp1, ters_tp1, weights):
        self.obss_t.d = np.array(obss_t)
        self.acts_t.d = np.array(acts_t)
        self.rews_tp1.d = np.array(rews_tp1)
        self.obss_tp1.d = np.array(obss_tp1)
        self.ters_tp1.d = np.array(ters_tp1)
        self.weights.d = np.array(weights)
        self.loss.forward()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # gradient normalization
        for variable in self.shared_params.values():
            variable.grad /= self.num_heads
        self.solver.update()
        return self.loss.d

    def update_target(self):
        for key in self.target_params.keys():
            self.target_params[key].data.copy_from(self.params[key].data)

    def reset(self, step):
        self.current_head = np.random.randint(self.num_heads)


class BootstrapReplayBuffer:
    def __init__(self, maxlen=10 ** 5, batch_size=32, num_heads=10):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=maxlen)
        self.num_heads = num_heads

    def append(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        ter_tp1 = 1.0 if ter_tp1 else 0.0
        weight = np.random.randint(2, size=self.num_heads)
        experience = dict(obs_t=obs_t, act_t=act_t,
                          rew_tp1=[rew_tp1], obs_tp1=obs_tp1,
                          ter_tp1=[ter_tp1], weight=weight)
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)


def update(model, buffer):
    def _func(step):
        experiences = buffer.sample()
        obss_t = []
        acts_t = []
        rews_tp1 = []
        obss_tp1 = []
        ters_tp1 = []
        weights = []
        for experience in experiences:
            obss_t.append(experience['obs_t'])
            acts_t.append(experience['act_t'])
            rews_tp1.append(experience['rew_tp1'])
            obss_tp1.append(experience['obs_tp1'])
            ters_tp1.append(experience['ter_tp1'])
            weights.append(experience['weight'])
        loss = model.train(pixel_to_float(obss_t), acts_t, rews_tp1,
                           pixel_to_float(obss_tp1), ters_tp1, weights)
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
    model = BootstrappedDQN(q_function, num_actions, args.num_heads,
                            args.batch_size, args.gamma, args.lr)
    if args.load is not None:
        nn.load_parameters(args.load)
    model.update_target()

    # replay buffer for experience replay
    buffer = BootstrapReplayBuffer(args.buffer_size, args.batch_size,
                                   args.num_heads)

    # epsilon-greedy exploration
    exploration = LinearlyDecayEpsilonGreedy(num_actions, args.epsilon, 0.01,
                                             args.schedule_duration)

    monitor = prepare_monitor(args.logdir)

    update_fn = update(model, buffer)

    eval_fn = evaluate(eval_env, model, render=args.render)

    train(env, model, buffer, exploration, monitor, update_fn, eval_fn,
          args.final_step, args.update_start, args.update_interval,
          args.target_update_interval, args.save_interval,
          args.evaluate_interval, ['loss'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-heads', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 6)
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
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
