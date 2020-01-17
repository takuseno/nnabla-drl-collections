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
from common.log import prepare_directory
from common.experiment import evaluate
from common.env import AtariWrapper


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
        # inference graph
        self.infer_obs_t = nn.Variable((num_envs, 4, 84, 84))
        self.infer_pi_t,\
        self.infer_value_t = cnn_network(self.infer_obs_t, num_actions,
                                         'network')
        self.infer_t = F.sink(self.infer_pi_t, self.infer_value_t)

        # evaluation graph
        self.eval_obs_t = nn.Variable((1, 4, 84, 84))
        self.eval_pi_t, _ = cnn_network(self.eval_obs_t, num_actions, 'network')

        # training graph
        self.obss_t = nn.Variable((batch_size, 4, 84, 84))
        self.acts_t = nn.Variable((batch_size, 1))
        self.rets_t = nn.Variable((batch_size, 1))
        self.advs_t = nn.Variable((batch_size, 1))

        pi_t, value_t = cnn_network(self.obss_t, num_actions, 'network')

        # value loss
        l2loss = F.squared_error(value_t, self.rets_t)
        self.value_loss = v_coeff * F.mean(l2loss)

        # policy loss
        log_pi_t = F.log(pi_t + 1e-20)
        a_one_hot = F.one_hot(self.acts_t, (num_actions,))
        log_probs_t = F.sum(log_pi_t * a_one_hot, axis=1, keepdims=True)
        self.pi_loss = F.mean(log_probs_t * self.advs_t)

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

    def evaluate(self, obs_t):
        self.eval_obs_t.d = np.array([obs_t])
        self.eval_pi_t.forward(clear_buffer=True)
        pi = self.eval_pi_t.d[0]
        return np.random.choice(pi.shape[0], p=pi)

    def train(self, obss_t, acts_t, rets_t, advs_t, step):
        self.obss_t.d = np.array(obss_t).reshape((-1, 4, 84, 84))
        self.acts_t.d = np.array(acts_t).reshape((-1, 1))
        self.rets_t.d = np.array(rets_t).reshape((-1, 1))
        self.advs_t.d = np.array(advs_t).reshape(-1, 1)
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
class BatchEnv:
    def __init__(self, envs):
        self.envs = envs

    def step(self, actions):
        obs_list = []
        rew_list = []
        ter_list = []
        for env, action in zip(self.envs, actions):
            obs, rew, done, _ = env.step(action)
            if done:
                obs = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            ter_list.append(1.0 if done else 0.0)
        return np.array(obs_list), np.array(rew_list), np.array(ter_list), {}

    def reset(self):
        return np.array([env.reset() for env in self.envs])
#-----------------------------------------------------------------------------#

#-------------------------- training loop ------------------------------------#
def pixel_to_float(obs):
    return np.array(obs, dtype=np.float32) / 255.0


def compute_returns(gamma):
    def _compute(vals_t, rews_tp1, ters_tp1):
        V = vals_t[-1]
        rets_t = []
        for rew_t, ter_tp1 in reversed(list(zip(rews_tp1, ters_tp1))):
            V = rew_t + gamma * V * (1.0 - ter_tp1)
            rets_t.append(V)
        return np.array(list(reversed(rets_t)))
    return _compute


def train_loop(env, model, num_actions, return_fn, logdir, eval_fn, args):
    monitor = Monitor(logdir)
    reward_monitor = MonitorSeries('reward', monitor, interval=1)
    eval_reward_monitor = MonitorSeries('eval_reward', monitor, interval=1)
    policy_loss_monitor = MonitorSeries('policy_loss', monitor, interval=10000)
    value_loss_monitor = MonitorSeries('value_loss', monitor, interval=10000)
    sample_action = lambda x: np.random.choice(num_actions, p=x)

    step = 0
    obs_t = env.reset()
    cumulative_reward = np.zeros(len(env.envs), dtype=np.float32)
    obss_t, acts_t, vals_t, rews_tp1, ters_tp1 = [], [], [], [], []
    while step <= args.final_step:
        # inference
        probs_t, val_t = model.infer(pixel_to_float(obs_t))
        # sample actions
        act_t = list(map(sample_action, probs_t))
        # move environment
        obs_tp1, rew_tp1, ter_tp1, _ = env.step(act_t)
        # clip reward between [-1.0, 1.0]
        clipped_rew_tp1 = np.clip(rew_tp1, -1.0, 1.0)

        obss_t.append(obs_t)
        acts_t.append(act_t)
        vals_t.append(val_t)
        rews_tp1.append(clipped_rew_tp1)
        ters_tp1.append(ter_tp1)

        # update parameters
        if len(obss_t) == args.time_horizon:
            vals_t.append(val_t)
            rets_t = return_fn(vals_t, rews_tp1, ters_tp1)
            advs_t = rets_t - vals_t[:-1]
            policy_loss, value_loss = model.train(pixel_to_float(obss_t),
                                                  acts_t, rets_t, advs_t, step)
            policy_loss_monitor.add(step, policy_loss)
            value_loss_monitor.add(step, value_loss)
            obss_t, acts_t, vals_t, rews_tp1, ters_tp1 = [], [], [], [], []

        # save parameters
        cumulative_reward += rew_tp1
        obs_t = obs_tp1

        for i, ter in enumerate(ter_tp1):
            step += 1
            if ter:
                reward_monitor.add(step, cumulative_reward[i])
                cumulative_reward[i] = 0.0
            if step % 10 ** 6 == 0:
                path = os.path.join(logdir, 'model_{}.h5'.format(step))
                nn.save_parameters(path)
                eval_reward_monitor.add(step, eval_fn())
#-----------------------------------------------------------------------------#

def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    # atari environment
    num_envs = args.num_envs
    envs = [gym.make(args.env) for _ in range(num_envs)]
    batch_env = BatchEnv([AtariWrapper(env, args.seed) for env in envs])
    eval_env = AtariWrapper(gym.make(args.env), 50, episodic=False)
    num_actions = envs[0].action_space.n

    # action-value function built with neural network
    lr_scheduler = learning_rate_scheduler(args.lr, 10 ** 7)
    model = A2C(num_actions, num_envs, num_envs * args.time_horizon,
                args.v_coeff, args.ent_coeff, lr_scheduler)
    if args.load is not None:
        nn.load_parameters(args.load)

    logdir = prepare_directory(args.logdir)

    eval_fn = evaluate(eval_env, model, args.render)

    # start training loop
    return_fn = compute_returns(args.gamma)
    train_loop(batch_env, model, num_actions, return_fn, logdir, eval_fn, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--time-horizon', type=int, default=5)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--v-coeff', type=float, default=0.5)
    parser.add_argument('--ent-coeff', type=float, default=0.01)
    parser.add_argument('--logdir', type=str, default='a2c')
    parser.add_argument('--load', type=str)
    parser.add_argument('--final-step', type=int, default=10 ** 7)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
