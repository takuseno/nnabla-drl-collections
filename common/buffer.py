import random

from collections import deque


class ReplayBuffer:
    def __init__(self, maxlen, batch_size):
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def append(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        experience = dict(obs_t=obs_t, act_t=act_t, rew_tp1=[rew_tp1],
                          obs_tp1=obs_tp1, ter_tp1=[ter_tp1])
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def size(self):
        return len(self.buffer)
