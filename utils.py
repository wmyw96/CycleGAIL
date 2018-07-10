import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        #print('error: dist = %.3f' % dist)
        dpos = np.random.uniform(-stepsize, stepsize, size=(3,))
        dpos[2] = np.abs(dpos[2])


def spindrive3d_generate_random_trajectories(nsteps, radius=5, alpha=1,
                                             dt=0.05, rg=0.05):
    state = np.zeros((nsteps, 3))
    action = np.zeros((nsteps, 3))
    pre_state = np.zeros(3)
    for i in range(nsteps):
        # x = r cos (t)
        # y = r sin (t)
        # z = alpha t
        t = i * dt
        cur_state = np.array([radius * np.cos(t), radius * np.sin(t),
                              alpha * t])
        cur_state = cur_state + np.random.uniform(-rg, rg, (3,))
        state[i, :] = cur_state
        if i > 0:
            action[i - 1, :] = cur_state - pre_state
        pre_state = cur_state
    t = nsteps * dt
    cur_state = np.array([radius * np.cos(t), radius * np.sin(t),
                          alpha * t])
    cur_state = cur_state + np.random.uniform(-rg, rg, (3,))
    action[nsteps - 1] = cur_state - pre_state
    action /= dt
    return state, action


def show_trajectory(env, state, action):
    obs = np.zeros_like(state)
    obs[0, :] = env.reset(state[0, :])
    for i in range(len(action) - 1):
        obs[i + 1, :], _, _, _ = env.step(action[i, :])
    #env.render()

    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(221, projection='3d')
    ax.plot(state[:, 0], state[:, 1], state[:, 2])
    ax2 = figure.add_subplot(222, projection='3d')
    ax2.plot(obs[:, 0], obs[:, 1], obs[:, 2])
    ax3 = figure.add_subplot(223)
    ax3.plot(state[:, 0], state[:, 1])
    ax4 = figure.add_subplot(224)
    ax4.plot(obs[:, 0], obs[:, 1])
    plt.show()