import numpy as np


class LinearlyDecayEpsilonGreedy:
    def __init__(self, num_actions, init_value, final_value, duration):
        self.num_actions = num_actions
        self.base= init_value - final_value
        self.init_value = init_value
        self.final_value = final_value
        self.duration = duration

    def get(self, t, greedy_action):
        decay = t / self.duration
        if decay > 1.0:
            decay = 1.0
        epsilon = (1.0 - decay) * self.base + self.final_value
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        return greedy_action

    def reset(self):
        pass


class ConstantEpsilonGreedy:
    def __init__(self, num_actions, epsilon):
        self.num_actions = num_actions
        self.epsilon = epsilon

    def get(self, t, greedy_action):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return greedy_action

    def reset(self):
        pass


# from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, sigma, theta=.15, time=1e-2, init_x=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.time = time
        self.init_x = init_x
        self.prev_x = None
        self.reset()

    def get(self, t, greedy_action):
        normal = np.random.normal(size=self.mean.shape)
        new_x = self.prev_x + self.theta * (self.mean - self.prev_x) \
            * self.time + self.sigma * np.sqrt(self.time) * normal
        self.prev_x = new_x
        return greedy_action + new_x

    def reset(self):
        if self.init_x is not None:
            self.prev_x = self.init_x
        else:
            self.prev_x = np.zeros_like(self.mean)


class NormalNoise:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def get(self, t, greedy_action):
        return greedy_action + np.random.normal(self.mean, self.sigma)

    def reset(self):
        pass


class EmptyNoise:
    def get(self, t, greedy_action):
        return greedy_action

    def reset(self):
        pass
