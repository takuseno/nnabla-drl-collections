import numpy as np
import random

from collections import deque


class ReplayBuffer:
    def __init__(self, maxlen, batch_size):
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def append(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        ter_tp1 = 1.0 if ter_tp1 else 0.0
        experience = dict(obs_t=obs_t, act_t=act_t, rew_tp1=[rew_tp1],
                          obs_tp1=obs_tp1, ter_tp1=[ter_tp1])
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def size(self):
        return len(self.buffer)


def _compute_returns(bootstrap_val, rews, ters, gamma):
    rets = []
    ret_tp1 = bootstrap_val
    for i in reversed(range(rews.shape[0])):
        ret_t = rews[i] + (1.0 - ters[i]) * gamma * ret_tp1
        rets.append(ret_t)
        ret_tp1 = ret_t
    rets = reversed(rets)
    return np.array(list(rets))


def _compute_gae(bootstrap_val, rews, vals, ters, gamma, lam):
    vals = np.concatenate((vals, [bootstrap_val]), axis=0)
    # compute delta
    deltas = []
    for i in reversed(range(rews.shape[0])):
        ret_t = rews[i] + (1.0 - ters[i]) * gamma * vals[i + 1]
        delta = ret_t - vals[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    adv_tp1 = deltas[-1]
    advs = [adv_tp1]
    for i in reversed(range(deltas.shape[0] - 1)):
        adv_t = deltas[i] + (1.0 - ters[i]) * gamma * lam * adv_tp1
        advs.append(adv_t)
        adv_tp1 = adv_t
    advs = reversed(advs)
    return np.array(list(advs))


class OnPolicyBuffer:
    def __init__(self, batch_size, gamma, lam, standardize_advantage=True):
        self.obss_t = []
        self.acts_t = []
        self.rews_t = []
        self.ters_t = []
        self.vals_t = []
        self.logprobs_t = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.standardize_advantage = standardize_advantage

    def append(self, obs_t, act_t, rew_t, ter_t, val_t, logprob_t):
        self.obss_t.append(obs_t)
        self.acts_t.append(act_t)
        self.rews_t.append(rew_t)
        self.ters_t.append(ter_t)
        self.vals_t.append(val_t)
        self.logprobs_t.append(logprob_t)

    def sample(self, last_val):
        # prepare data
        obss_t = np.array(self.obss_t)
        acts_t = np.array(self.acts_t)
        rews_t = np.array(self.rews_t)
        ters_t = np.array(self.ters_t)
        vals_t = np.array(self.vals_t)
        logprobs_t = np.array(self.logprobs_t)
        rets_t = _compute_returns(last_val, rews_t, ters_t, self.gamma)
        advs_t = _compute_gae(last_val, rews_t, vals_t, ters_t, self.gamma,
                              self.lam)

        # flatten shape
        flat_obss_t = np.reshape(obss_t, [-1] + obss_t.shape[2:])
        flat_acts_t = np.reshape(acts_t, [-1] + acts_t.shape[2:])
        flat_vals_t = np.reshape(vals_t, [-1, 1])
        flat_rets_t = np.reshape(rets_t, [-1, 1])
        flat_advs_t = np.reshape(advs_t, [-1, 1])
        flat_logprobs_t = np.reshape(logprobs_t, [-1] + logprobs_t.shape[2:])

        if self.standardize_advantage:
            advs_mean = np.mean(flat_advs_t)
            advs_std = np.std(flat_advs_t)
            flat_advs_t = (flat_advs_t - advs_mean) / advs_std

        num_batches = flat_obss_t.shape[0] // self.batch_size

        indices = np.random.arange(self.size())
        np.random.shuffle(indices)

        for i in range(num_batches):
            cursor = i * self.batch_size
            cur_indices = indices[cursor:cursor + self.batch_size]
            cur_obss_t = flat_obss_t[cur_indices]
            cur_acts_t = flat_acts_t[cur_indices]
            cur_vals_t = flat_vals_t[cur_indices]
            cur_rets_t = flat_rets_t[cur_indices]
            cur_advs_t = flat_advs_t[cur_indices]
            cur_logprobs_t = flat_logprobs_t[cur_indices]
            batch = dict(obss_t=cur_obss_t, acts_t=cur_acts_t,
                         vals_t=cur_vals_t, rets_t=cur_rets_t,
                         advs_t=cur_advs_t, logprobs_t=cur_logprobs_t)
            yield batch

    def size(self):
        return len(self.obss_t)

    def clear(self):
        self.obss_t.clear()
        self.acts_t.clear()
        self.rews_t.clear()
        self.ters_t.clear()
        self.vals_t.clear()
        self.logprobs_t.clear()
