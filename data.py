import numpy as np
import os


class Demonstrations(object):
    def __init__(self, seed, a, b, mod):
        self.seed = seed
        self.seed_a = a
        self.seed_b = b
        self.seed_mod = mod
        self.demos = []
        self.indexs = []
        self.pointer = 0

    def add_demo(self, state, action):
        self.demos.append((state, action))
        self.pointer = len(self.demos)

    def next_demo(self):
        if self.pointer == len(self.demos):
            self.seed = (self.seed * self.seed_a + self.seed_b) % self.seed_mod
            arr = np.arange(self.pointer)
            np.random.shuffle(arr)
            self.indexs = [int(i) for i in arr]
            self.pointer = 1
            return self.demos[self.indexs[0]]
        self.pointer += 1
        return self.demos[self.indexs[self.pointer] - 1]

    def load(self, file_name, nitems):
        if os.path.isdir(file_name):
            pass
        else:
            os.mkdir(file_name)
        if file_name[-1] != '/':
            file_name += '/'
        for i in range(nitems):
            state = np.load(file_name + 'traj%d_obs.npy' % i)
            action = np.load(file_name + 'traj%d_act.npy' % i)
            self.add_demo(state, action)

    def save(self, file_name):
        if os.path.isdir(file_name):
            pass
        else:
            os.mkdir(file_name)
        if file_name[-1] != '/':
            file_name += '/'
        for i in range(len(self.demos)):
            state, action = self.demos[i]
            np.save(file_name + 'traj%d_obs.npy' % i, state)
            np.load(file_name + 'traj%d_act.npy' % i, action)

