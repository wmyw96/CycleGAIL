import numpy as np
import os


class Demonstrations(object):
    def __init__(self, seed, a, b, mod):
        self.seed = seed
        self.seed_a = a
        self.seed_b = b
        self.seed_mod = mod
        self.demos = []
        self.train_indexs = []
        self.test_indexs = []
        self.pointer = 0
        self.test_demos = []
        self.test_pointer = 0
        self.train_demos = []
        self.train_pointer = 0

    def add_demo(self, state, action):
        self.demos.append((state, action))
        self.pointer = len(self.demos)

    def _next_demo(self, train=True):
        if train:
            if self.train_pointer == len(self.train_demos):
                self.seed = (self.seed * self.seed_a + self.seed_b) \
                            % self.seed_mod
                arr = np.arange(self.train_pointer)
                np.random.shuffle(arr)
                self.train_indexs = [int(i) for i in arr]
                self.train_pointer = 1
                return self.train_demos[self.train_indexs[0]]
            self.train_pointer += 1
            return self.train_demos[self.train_indexs[self.train_pointer - 1]]
        else:
            if self.test_pointer == len(self.test_demos):
                self.seed = (self.seed * self.seed_a + self.seed_b) \
                            % self.seed_mod
                arr = np.arange(self.test_pointer)
                np.random.shuffle(arr)
                self.test_indexs = [int(i) for i in arr]
                self.test_pointer = 1
                return self.test_demos[self.test_indexs[0]]
            self.test_pointer += 1
            return self.test_demos[self.test_indexs[self.test_pointer - 1]]

    def set(self, num_trains):
        if num_trains >= self.pointer:
            raise ValueError('Number of train trajectories is larger than '
                             'number of total trajectories\n')
        self.train_demos = []
        self.test_demos = []
        for i in range(num_trains):
            self.train_demos.append(self.demos[i])
        self.train_pointer = num_trains
        for i in range(self.pointer - num_trains):
            self.test_demos.append(self.demos[i + num_trains])
        self.test_pointer = self.pointer - num_trains

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
            np.save(file_name + 'traj%d_act.npy' % i, action)

    def next_demo(self, train=True):
        ff, gg = self._next_demo(train)
        return ff[:100, :], gg[:100, :]





















































###
