import numpy as np
import os
from scipy.stats import ortho_group


class Demonstrations(object):
    def __init__(self, seed, a, b, mod, trans_obs=None, trans_act=None):
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
        self.act_dim = 0
        self.obs_dim = 0
        if trans_obs is None:
            self.trans_obs = IdentityTransform()
        else:
            self.trans_obs = trans_obs
        if trans_act is None:
            self.trans_act = IdentityTransform()
        else:
            self.trans_act = trans_act

    def add_demo(self, state, action):
        state = np.array(state)
        action = np.array(action)
        #print(state.shape, action.shape)
        self.obs_dim = state.shape[1]
        self.act_dim = action.shape[1]
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

    def set_bz(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.batch_pointer = -1
        self.seq_len = seq_len

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

        for i in range(len(self.demos)):
            #print(self.demos[i][0].shape)
            self.demos[i] = (self.trans_obs.run(self.demos[i][0]),
                             self.trans_act.run(self.demos[i][1]))
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
        obs = obs[:self.seq_len, :]
        act = act[:self.seq_len, :]
        if normalize:
            return (obs - self.obs_bias) / self.obs_scalar, \
                   (act - self.act_bias) / self.act_scalar, \
                   (np.array(range(obs.shape[0]), dtype=np.float32) /
                    float(obs.shape[0])).reshape((-1, 1))
        else:
            return obs, act, np.array(range(obs.shape[0]))

    def next_batch(self):
        obss, acts, tss = [], [], []
        for i in range(self.batch_size):
            obs, act, ts = self.next_demo(train=True)
            obss.append(np.expand_dims(obs, 0))
            acts.append(np.expand_dims(act, 0))
            tss.append(np.expand_dims(ts, 0))
        obss = np.concatenate(obss, 0)
        acts = np.concatenate(acts, 0)
        tss = np.concatenate(tss, 0)
        return obss, acts, tss

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
        self.obs_scalar = np.sqrt(np.var(obs_full, 0, keepdims=True)) + 1e-6
        self.act_scalar = np.sqrt(np.var(act_full, 0, keepdims=True)) + 1e-6

        #self.obs_scalar = np.ones_like(self.obs_scalar)
        #self.act_scalar = np.ones_like(self.act_scalar)
        #self.obs_bias = np.zeros_like(self.obs_bias)
        #self.act_bias = np.zeros_like(self.act_bias)
        print(self.obs_scalar)
        print(self.act_scalar)
        print(self.obs_bias)
        print(self.act_bias)

    def act_r(self, acts):
        return acts * self.act_scalar + self.act_bias

    def obs_r(self, acts):
        return acts * self.obs_scalar + self.obs_bias

    def act_n(self, acts):
        return (acts - self.act_bias) / self.act_scalar

    def obs_n(self, obs):
        return (obs - self.obs_bias) / self.obs_scalar


class LinearTransform(object):
    def __init__(self, n):
        self.n = n
        self.kernel = ortho_group.rvs(dim=self.n)
        self.bias = np.random.uniform(-1.0, 1.0, (n,))

    def run(self, inp):
        return np.dot(inp, self.kernel) + self.bias


class NonLinearTransform(object):
    def __init__(self, n, nlayers):
        self.n = n
        self.nlayers = nlayers
        self.kernels = []
        self.biases = []
        for i in range(nlayers):
            if n > 1:
                self.kernels.append(ortho_group.rvs(dim=n))
            else:
                self.kernels.append(np.random.uniform(-1.0, 1.0, (1, 1)))
            self.biases.append(np.random.uniform(-1.0, 1.0, (n,)))

    def run(self, inp):
        out = np.dot(inp, self.kernels[0]) + self.biases[0]
        for i in range(self.nlayers - 1):
            out = np.tanh(out / 10) * 10
            out = np.dot(out, self.kernels[i + 1]) + self.biases[i + 1]
        return out


class IdentityTransform(object):
    def __init__(self):
        pass

    def run(self, inp):
        return inp


class ConcatTransform(object):
    def __init__(self, trans_obs, trans_act, obs_dim):
        self.trans_obs = trans_obs
        self.trans_act = trans_act
        self.obs_dim = obs_dim

    def run(self, inp):
        obs = inp[:, :self.obs_dim]
        act = inp[:, self.obs_dim:]
        res = np.concatenate([self.trans_obs.run(obs),
                              self.trans_act.run(act)], 1)
        return res
