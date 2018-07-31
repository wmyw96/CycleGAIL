import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    print('Cannot import matplotlib')
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import warnings
from sn import spectral_normed_weight
import os


def lrelu(x, alpha=0.2):
    # return tf.nn.tanh(x)
    return tf.nn.relu(x)
    # return tf.maximum(x, x * alpha)


def cycle_loss(origin, reconstructed, metric, weight=None):
    if weight is None:
        weight = np.ones((1, origin.get_shape()[1]))
    if metric == 'L1':
        return tf.reduce_mean(tf.abs(origin - reconstructed) * weight)
    if metric == 'L2':
        return tf.reduce_mean(tf.square(origin - reconstructed) * weight)
    raise NotImplementedError


def dense_layer(inp, size, name, weight_init, bias_init=0, bias=True,
                reuse=None, spectral_weight=False, update_collection=None):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [inp.get_shape()[1], size],
                            initializer=weight_init)
        if spectral_weight:
            warnings.warn('You are now using spectral weight trick')
            w = spectral_normed_weight(w, update_collection=update_collection)
        if bias:
            b = tf.get_variable('b', [size],
                                initializer=tf.constant_initializer(bias_init))
            return tf.nn.bias_add(tf.matmul(inp, w), b)
        else:
            return tf.matmul(inp, w)


def get_flatten_dim(shape):
    dim = 1
    for sub_dim in shape:
        dim *= int(sub_dim)
    return dim


# copy from openai/baselines/common/tf_util
def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):
        # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


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


def show_trajectory(env, state, action, init_state=None, filename=None):
    obs = np.zeros_like(state)
    if init_state is None:
        init_state = state[0, :]
    obs[0, :] = env.reset(init_state)
    #print(obs[0, :])
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
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def trajectory_diff(obs1, obs2, filename=None):
    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(221, projection='3d')
    ax.plot(obs1[:, 0], obs1[:, 1], obs1[:, 2])
    ax2 = figure.add_subplot(222, projection='3d')
    ax2.plot(obs2[:, 0], obs2[:, 1], obs2[:, 2])
    ax3 = figure.add_subplot(223)
    ax3.plot(obs1[:, 0], obs1[:, 1])
    ax4 = figure.add_subplot(224)
    ax4.plot(obs2[:, 0], obs2[:, 1])
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def distribution_diff(state0, action0, state1, action1, state2, action2,
                      filename=None):
    figure = plt.figure(figsize=(8, 8))
    ax1 = figure.add_subplot(221)
    ax1.scatter(state0[:, 0], action0[:, 0], color='b', s=0.5)
    ax1.scatter(state1[:, 0], action1[:, 0], color='r', s=0.5)
    ax1.scatter(state2[:, 0], action2[:, 0], color='y', s=0.5)
    ax2 = figure.add_subplot(222)
    ax2.scatter(state0[:, 1], action0[:, 1], color='b', s=0.5)
    ax2.scatter(state1[:, 1], action1[:, 1], color='r', s=0.5)
    ax2.scatter(state2[:, 1], action2[:, 1], color='y', s=0.5)
    ax3 = figure.add_subplot(223)
    ax3.scatter(state0[:, 0], action0[:, 1], color='b', s=0.5)
    ax3.scatter(state1[:, 0], action1[:, 1], color='r', s=0.5)
    ax3.scatter(state2[:, 0], action2[:, 1], color='y', s=0.5)
    ax4 = figure.add_subplot(224)
    ax4.scatter(state0[:, 1], action0[:, 0], color='b', s=0.5)
    ax4.scatter(state1[:, 1], action1[:, 0], color='r', s=0.5)
    ax4.scatter(state2[:, 1], action2[:, 0], color='y', s=0.5)
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def generate_dir(prefix):
    if os.path.isdir(prefix):
        pass
    else:
        os.mkdir(prefix)


def distribution_pdiff(state0, action0, state1, action1, state2, action2,
                       dir_name):
    print('Generating distribution diff images')
    generate_dir(dir_name)
    print('State-action distribution pairs')
    for i in range(state0.shape[1]):
        for j in range(action0.shape[1]):
            print('%d %d' % (i, j))
            plt.figure()
            print('fgp')
            plt.scatter(state0[:, i], action0[:, j], color='b', s=0.5)
            plt.scatter(state1[:, i], action1[:, j], color='r', s=0.5)
            plt.scatter(state2[:, i], action2[:, j], color='y', s=0.5)
            plt.savefig(dir_name + '/s%da%d' % (i, j))
            plt.close()
    print('State-state distribution pairs')
    for i in range(state0.shape[1]):
        for j in range(state0.shape[1]):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.scatter(state0[:, i], state0[:, j], color='b', s=0.5)
            ax.scatter(state1[:, i], state1[:, j], color='r', s=0.5)
            ax.scatter(state2[:, i], state2[:, j], color='y', s=0.5)
            plt.savefig(dir_name + '/s%ds%d' % (i, j))
            plt.close()
    print('Action-action distribution pairs')
    for i in range(action0.shape[1]):
        for j in range(action0.shape[1]):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.scatter(action0[:, i], action0[:, j], color='b', s=0.5)
            ax.scatter(action1[:, i], action1[:, j], color='r', s=0.5)
            ax.scatter(action2[:, i], action2[:, j], color='y', s=0.5)
            plt.savefig(dir_name + '/a%da%d' % (i, j))
            plt.close()
