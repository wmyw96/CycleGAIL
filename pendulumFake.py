import gym
import numpy as np


class FakePendulumEnv(gym.Env):
    def __init__(self, alpha, beta):
        super.__init__()
        self.alpha = max(np.abs(alpha), 1e-3)
        self.beta = beta

    def step(self, u):
        fixed_u = self.alpha * u + self.beta
        return super.step(fixed)
