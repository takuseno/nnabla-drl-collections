import numpy as np
import random

from collections import deque


# https://github.com/chainer/chainerrl/blob/master/chainerrl/misc/random.py
def sample_n_k(n, k):
    """Sample k distinct elements uniformly from range(n)"""

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return np.empty((0,), dtype=np.int64)
    elif 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        result = np.random.choice(n, 2 * k)
        selected = set()
        selected_add = selected.add
        j = k
        for i in range(k):
            x = result[i]
            while x in selected:
                x = result[i] = result[j]
                j += 1
                if j == 2 * k:
                    # This is slow, but it rarely happens.
                    result[k:] = np.random.choice(n, k)
                    j = k
            selected_add(x)
        return result[:k]


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
        samples = []
        for index in sample_n_k(self.size(), self.batch_size):
            samples.append(self.buffer[index])
        return samples

    def size(self):
        return len(self.buffer)
