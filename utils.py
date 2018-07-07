import gym
import numpy as np


def greedy_reject_sampling(state, alpha, radius, rg, stepsize):
    dpos = np.random.uniform(-stepsize, stepsize, size=(3,))
    dpos[2] = np.abs(dpos[2])
    #dpos[2] = stepsize / 3 + dpos[2] / 10
    while True:
        new_state = state + dpos
        z = new_state[2]
        t = z / alpha
        target = np.array([np.cos(t) * radius, np.sin(t) * radius, z])
        bias = new_state - target
        dist = np.sqrt(np.sum(bias * bias))
        if dist < rg:
            return dpos * 20
        print('error: dist = %.3f' % dist)
        dpos = np.random.uniform(-stepsize, stepsize, size=(3,))
        dpos[2] = np.abs(dpos[2])
