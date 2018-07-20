import gym
import numpy as np
from dataset import Demonstrations
import matplotlib.pyplot as plt
import matplotlib
import cv2


def show_trajectory(env, obs, acts, animate=False):
    env.reset()
    qpos_dim = env.env.model.nq
    qvel_dim = env.env.model.nv
    qpos = obs[0, :qpos_dim]
    qvel = obs[0, qpos_dim:]
    env.env.set_state(qpos, qvel)
    horizon = obs.shape[0]
    state = obs[0, 1:]
    err = 0.0
    reward_sum = 0.0
    for i in range(horizon):
        clip = np.concatenate([obs[i, 1:qpos_dim],
                          np.clip(obs[i, qpos_dim:], -10, 10)])
        clip = obs[i, 1:]
        err += np.sum((state - clip) * (state - clip))
        print(np.sum((state - clip) * (state - clip)))
        if animate:
            env.render()
        nxt_state, rd, __, ___ = env.step(acts[i])
        reward_sum += rd
        state = nxt_state
    print('state L2 error: %.3f' % err)
    print('reward: %.3f' % reward_sum)


def save_trajectory_images(env, obs, acts, file_path):
    env.reset()
    qpos_dim = env.env.model.nq
    qvel_dim = env.env.model.nv
    qpos = obs[0, :qpos_dim]
    qvel = obs[0, qpos_dim:]
    env.env.set_state(qpos, qvel)
    horizon = obs.shape[0]
    for i in range(horizon):
        img = env.render(mode='rgb_array')
        matplotlib.image.imsave(file_path + '/real%d.jpg' % i,
                                img, format='jpg')
        env.step(acts[i])
    for i in range(horizon):
        qpos = obs[i, :qpos_dim]
        qvel = obs[i, qpos_dim:]
        env.reset()
        env.env.set_state(qpos, qvel)
        img = env.render(mode='rgb_array')
        matplotlib.image.imsave(file_path + '/imag%d.jpg' % i,
                                img, format='jpg')


def save_video(file_prefix, num_items):
    fps = 24  # frequency
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(file_prefix + '.avi', fourcc, fps, (500,500))
    for i in range(num_items):
        img12 = cv2.imread(file_prefix + str(i) +'.jpg')
        video_writer.write(img12)
    video_writer.release()


if __name__ == '__main__':
    np.random.seed(1234)
    demos = Demonstrations(1, 34, 23, 1000000007)
    demos.load('data/HalfCheetah-v1-3', 100)
    demos.set(50)
    obss, acts = demos.next_demo()
    #obss, acts = demos.next_demo()
    env = gym.make("HalfCheetah-v1")
    show_trajectory(env, obss, acts, True)
    #save_trajectory_images('HalfCheetah-v1/t1', env, obss, acts)
    #save_video('HalfCheetah-v1/t1/real', 1000)