import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class SpinDrive3D(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    # we should follow the 3-D line
    #  x = r cos(t)
    #  y = r sin(t)
    #  z = a t
    # action is the velocity, state is the position
    def __init__(self, radius=2, alpha=1, rg=0.2):
        self.alpha = alpha
        self.radius = radius
        self.dt = 0.05
        self.range = rg
        self.viewer = None
        self.state_list = []
        self.figure = None

        high = np.array([2 * radius, 2 * radius, 3 * radius])
        low = np.array([-2 * radius, -2 * radius, -radius])

        mxv = radius
        self.max_velocity = mxv
        self.action_space = \
            spaces.Box(low=np.array([-mxv, -mxv, -mxv]),
                       high=np.array([mxv, mxv, mxv]))
        self.observation_space = spaces.Box(low=low, high=high)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        pos = self.np_random.uniform(low=-self.range / 2,
                                     high=self.range / 2,
                                     size=(3,))
        pos[0] += self.radius
        pos[2] = 0
        self.state = pos
        self.state_list = []
        return self.state

    # suppose u is a 3-dim vector
    def step(self, u):
        u = np.clip(u, -self.max_velocity, self.max_velocity)
        pos = self.state + u * self.dt

        t = pos[2] / self.alpha
        z = pos[2]
        vec = pos - np.array([self.alpha * np.cos(t),
                              self.alpha * np.sin(t), z])
        dist = np.sqrt(np.sum(vec * vec))
        reward = (dist < self.range) * (1 + z)
        self.state_list.append(pos)
        self.state = pos

        return pos, reward, False, {}

    def render(self, mode='human'):
        xdata = [pos[0] for pos in self.state_list]
        ydata = [pos[1] for pos in self.state_list]
        zdata = [pos[2] for pos in self.state_list]

        if mode == 'human':
            self.figure = plt.figure(figsize=(7, 7))
            ax = self.figure.add_subplot(111, projection='3d')
            ax.plot(xdata, ydata, zdata)
            plt.xlim(-2 * self.radius, 2 * self.radius)
            plt.ylim(-2 * self.radius, 2 * self.radius)
            plt.show()

    def close(self):
        pass


if __name__ == '__main__':
    env = SpinDrive3D()
    env.reset()
    env.render()

    for i in range(1000):
        env.step(np.random.uniform(-2, 2, size=(3,)))
        env.render()
