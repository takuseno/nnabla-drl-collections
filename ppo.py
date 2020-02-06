import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import argparse
import gym

from nnabla.ext_utils import get_extension_context
from nnabla.parameter import get_parameter_or_create
from nnabla.initializer import ConstantInitializer
from common.distribution import Normal
from common.log import prepare_monitor
from common.experiment import evaluate, train
from common.exploration import EmptyNoise


def value_network(obs, name):
    with nn.parameter_scope(name):
        out = PF.affine(obs, 64, name='fc1')
        out = F.tanh(out)
        out = PF.affine(out, 64, name='fc2')
        out = F.tanh(out)
        out = PF.affine(out, 1, name='fc3')
    return out


def policy_network(obs, action_size, name):
    with nn.parameter_scope(name):
        out = PF.affine(obs, 64, name='fc1')
        out = F.tanh(out)
        out = PF.affine(out, 64, name='fc2')
        out = F.tanh(out)
        mean = PF.affine(out, action_size, name='mean')
        logstd = get_parameter_or_create('logstd', [1, action_size],
                                         ConstantInitializer(0.0))
    return Normal(mean, F.exp(2 * logstd))


class PPO:
    def __init__(self,
                 obs_shape,
                 action_size,
                 num_envs,
                 batch_size,
                 lr,
                 eps,
                 value_factor,
                 entropy_factor):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.value_factor = value_factor
        self.entropy_factor = entropy_factor

    def _build(self):
        # inference graph
        self.infer_obs_t = nn.Variable((self.num_envs,) + self.obs_shape)
        with nn.parameter_scope('trainable'):
            infer_dist_t = policy_network(self.infer_obs_t, self.action_size,
                                          'actor')
            self.infer_val_t = value_network(self.infer_obs_t, 'critic')
        self.infer_act_t = infer_dist_t.sample()
        self.infer_t = F.sink(self.infer_val_t, self.infer_act_t)

        # evaluation graph
        self.eval_obs_t = nn.Variable((1,) + self.obs_shape)
        with nn.parameter_scope('trainable'):
            eval_dist_t = policy_network(self.infer_obs_t, self.action_size,
                                         'actor')
        self.eval_act_t = eval_dist_t.mean()

        # training graph
        self.obss_t = nn.Variable((self.batch_size,) + self.obs_shape)
        self.acts_t = nn.Variable((self.batch_size, self.action_size))
        self.old_vals_t = nn.Variable((self.batch_size, 1))
        self.rets_t = nn.Variable((self.batch_size, 1))
        self.advs_t = nn.Variable((self.batch_size, 1))
        self.old_logprobs_t = nn.Variable((self.batch_size, self.action_size))

        with nn.parameter_scope('trainable'):
            dist_t = policy_network(self.obss_t, self.action_size, 'actor')
            vals_t = value_network(self.obss_t, 'critic')
            logprobs_t = dist_t.log_prob(self.acts_t)

        # value training
        clipped_diff = F.clip_by_value(vals_t - self.old_vals_t, -self.eps,
                                       self.eps)
        value_loss_clipped = (self.old_vals_t + clipped_diff - self.rets_t) ** 2
        value_loss_non_clipped = (self.rets_t - vals_t) ** 2
        value_loss_min = F.maximum2(value_loss_clipped, value_loss_non_clipped)
        self.value_loss = self.value_factor * F.mean(value_loss_min)

        # policy training
        sum_logprobs_t = F.mean(logprobs_t, axis=1, keepdims=True)
        sum_old_logprobs_t = F.mean(self.old_logprobs_t, axis=1, keepdims=True)
        ratio = F.exp(sum_logprobs_t - sum_old_logprobs_t)
        surr1 = ratio * self.advs_t
        surr2 = F.clip_by_value(ratio, 1.0 - self.eps, 1.0 + self.eps) * self.advs
        surr = F.minimum2(surr1, surr2)
        self.policy_loss = -F.mean(surr)

        # entropy loss
        self.entropy_loss = self.entropy_factor * -F.mean(dist_t.entropy())

        self.loss = self.policy_loss + self.value_loss + self.entropy_loss

        with nn.parameter_scope('trainable'):
            self.params = nn.get_parameters()

        self.solver = S.Adam(self.lr)
        self.solver.set_parameters(self.params)

    def infer(self, obs_t):
        self.infer_obs_t.d = np.array(obs_t)
        self.infer_t.forward(clear_buffer=True)
        return self.infer_act_t.d, self.infer_val_t.d

    def evaluate(self, obs_t):
        self.eval_obs_t.d = np.array([obs_t])
        self.eval_act_t.forward(clear_buffer=True)
        return self.eval_act_t.d[0]

    def train(self, obss_t, acts_t, old_vals_t, rets_t, advs_t, old_logprobs_t):
        self.obss_t.d = np.array(obss_t)
        self.acts_t.d = np.array(acts_t)
        self.old_vals_t.d = np.array(old_vals_t)
        self.rets_t.d = np.array(rets_t)
        self.advs_t.d = np.array(advs_t)
        self.old_logprobs_t.d = np.array(old_logprobs_t)
        self.loss.forward()
        policy_loss = self.policy_loss.d.copy()
        value_loss = self.value_loss.d.copy()
        entropy_loss = self.entropy_loss.d.copy()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)
        # global gradient clipping
        sum_grad = 0
        for variable in self.params.values():
            sum_grad += np.sum(variable.g ** 2)
        if np.sqrt(sum_grad) > 0.5:
            for variable in self.params.values():
                variable.g = 0.5 * variable.g / np.sqrt(sum_grad)
        self.solver.update()
        return policy_loss, value_loss, entropy_loss

    def reset(self, step):
        pass


def update(model, buffer, num_epochs):
    def _func(step, last_val):
        policy_loss_hist = []
        value_loss_hist = []
        entropy_loss_hist = []
        for epoch in range(num_epochs):
            for batch in buffer.sample(last_val):
                loss = model.train(batch['obss_t'], batch['acts_t'],
                                   batch['vals_t'], batch['rets_t'],
                                   batch['advs_t'], batch['logprobs_t'])
                policy_loss_hist.append(loss[0])
                value_loss_hist.append(loss[1])
                entropy_loss_hist.append(loss[2])
        policy_loss = np.mean(policy_loss_hist)
        value_loss = np.mean(value_loss_hist)
        entropy_loss = np.mean(entropy_loss_hist)
        return policy_loss, value_loss, entropy_loss
    return _func
