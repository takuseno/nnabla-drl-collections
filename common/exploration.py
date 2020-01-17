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


class ConstantEpsilonGreedy:
    def __init__(self, num_actions, epsilon):
        self.num_actions = num_actions
        self.epsilon = epsilon

    def get(self, t, greedy_action):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return greedy_action
