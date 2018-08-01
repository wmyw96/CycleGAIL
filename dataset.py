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
        self.batch_size = 0
        self.batch_pointer = 0
        self.cur_obs = np.zeros((1, 1))
        self.cur_act = np.zeros((1, 1))
        self.act_scalar = None
        self.act_bias = None
        self.obs_scalar = None
        self.obs_bias = None

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

    def set_bz(self, batch_size):
        self.batch_size = batch_size
        self.batch_pointer = -1

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
        self.normalize()

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

    def next_demo(self, train=True, normalize=True):
        obs, act = self._next_demo(train)
        #return obs, act
        if normalize:
            return (obs - self.obs_bias) / self.obs_scalar, \
                   (act - self.act_bias) / self.act_scalar, \
                   (np.array(range(obs.shape[0]), dtype=np.float32) /
                    float(obs.shape[0])).reshape((-1, 1))
        else:
            return obs, act, np.array(range(obs.shape[0]))
        #return obs[:100, :], act[:100, :]
        #return obs[900:1000, :], act[900:1000, :]

    def next_batch(self):
        if self.batch_pointer == -1 or \
           self.batch_pointer >= self.cur_obs.shape[0]:
            self.cur_obs, self.cur_act, _ = self.next_demo(train=True)
            # arr = np.arange(self.cur_obs.shape[0])
            # np.random.shuffle(arr)
            # indexs = [int(i) for i in arr]
            # self.cur_obs = self.cur_obs[indexs, :]
            # self.cur_act = self.cur_act[indexs, :]
            self.batch_pointer = 0
        ed = min(self.batch_pointer + self.batch_size,
                 self.cur_obs.shape[0])
        rt_obs = self.cur_obs[self.batch_pointer:ed, :]
        rt_act = self.cur_act[self.batch_pointer:ed, :]
        horizon = self.cur_obs.shape[0]
        ts = np.zeros((rt_obs.shape[0], 1))
        for i in range(ed - self.batch_pointer):
            ts[i, 0] = (i + self.batch_pointer + 0.0) / horizon
        self.batch_pointer = ed
        return rt_obs, rt_act, ts

    def normalize(self):
        obs_full = []
        act_full = []
        for oa_pair in self.demos:
            obs, act = oa_pair
            obs_full.append(obs)
            act_full.append(act)
        obs_full = np.concatenate(obs_full, 0)
        act_full = np.concatenate(act_full, 0)
        self.obs_bias = np.mean(obs_full, 0, keepdims=True)
        self.act_bias = np.mean(act_full, 0, keepdims=True)
        self.obs_scalar = np.sqrt(np.var(obs_full, 0, keepdims=True))
        self.act_scalar = np.sqrt(np.var(act_full, 0, keepdims=True))

    def act_r(self, acts):
        return acts * self.act_scalar + self.act_bias

    def obs_r(self, acts):
        return acts * self.obs_scalar + self.obs_bias
