import gym
from dataset import Demonstrations
import numpy as np


def set_state(env, state):
    qpos = np.concatenate([np.ones(1) * 2.34, state[:8]])
    qv = state[8:-1]
    env.env.set_state(qpos, qv)


def verify(env_name, file_name):
    demos = Demonstrations(1, 34, 23, 1000000007)
    demos.load(file_name, 25)
    env = gym.make(env_name)
    demos.set(2)
    state, action = demos.next_demo()
    #state, action = demos.next_demo()

    k = 0
    for k in range(1000):
        env.reset()
        set_state(env, state[k, :])
        for j in range(1000 - k):
            s, _, __, ___ = env.step(action[k + j, :])
        err = 0
        for i in range(10):
            env.reset()
            set_state(env, state[k, :])
            for j in range(1000 - k):
                o, _, __, ___ = env.step(action[k + j, :])
            if (np.sum((o - s) * (o - s)) > 1e-6):
                print('%d %d: ' % (k, i) + str(o - s))
            err += np.sum((o - s) * (o - s))
        print('%d %.8f' % (k, err))


def load_data(env_name, file_name):
    demos = Demonstrations(1, 34, 23, 1000000007)
    demos.load(file_name, 25)
    env = gym.make(env_name)
    demos.set(24)
    for i in range(6):
        state, action = demos.next_demo()
    env.reset()
    set_state(env, state[0, :])
    k = 0
    for k in range(1000):
        s, _, __, ___ = env.step(action[k, :])
        error = np.sum((s - state[k + 1, 0:-1]) * (s - state[k + 1, 0:-1]))
        print('timestep %d, error = %.10f' % (k, error))
        env.render()


if __name__ == '__main__':
    load_data("HalfCheetah-v1", "data/HalfCheetah-v1")
