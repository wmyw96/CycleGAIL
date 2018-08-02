import gym
import numpy as np
from dataset import Demonstrations
try:
    import matplotlib.pyplot as plt
    import matplotlib
except:
    print('Cannot import matplotlib')
import cv2
from gym_extensions.continuous import mujoco


def show_animate_trajectory(env, obs, acts, animate=False):
    env.reset()
    qpos_dim = env.env.model.nq
    qvel_dim = env.env.model.nv
    qpos = obs[0, :qpos_dim]
    qvel = obs[0, qpos_dim:]
    env.env.set_state(qpos, qvel)
    horizon = obs.shape[0]
    state = obs[0, 1:]
    err = 0.0
    print(np.mean(np.var(acts, 0)))
    print(np.mean(np.var(obs, 0)))
    print('horizon = %d\n' % horizon)
    reward_sum = 0.0
    for i in range(horizon):
        #clip = np.concatenate([obs[i, 1:qpos_dim],
        #                  np.clip(obs[i, qpos_dim:], -10, 10)])
        # print(state)
        clip =  obs[i, 1:]
        err += np.sum((state - clip) * (state - clip))
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
    print('Save Trajectory Images in \"%s\": %d timesteps' % (
        file_path, horizon))
    true_obs = np.zeros_like(obs)
    true_obs[0, :] = obs[0, :]
    reward_sum = 0
    for i in range(horizon):
        img = env.render(mode='rgb_array')
        matplotlib.image.imsave(file_path + '/real%d.jpg' % i,
                                img, format='jpg')
        _1, rd, _2, _3 = env.step(acts[i])
        reward_sum += rd
        nxt_obs = \
            np.concatenate([env.env.model.data.qpos,
                            env.env.model.data.qvel]).reshape(-1)
        if i + 1 < horizon:
            true_obs[i + 1, :] = nxt_obs
    print('Total reward: %3f\n' % reward_sum)
    env.reset()
    for i in range(horizon):
        qpos = obs[i, :qpos_dim]
        qvel = obs[i, qpos_dim:]
        env.env.set_state(qpos, qvel)
        img = env.render(mode='rgb_array')
        matplotlib.image.imsave(file_path + '/imag%d.jpg' % i,
                                img, format='jpg')
    return true_obs


def save_video(file_prefix, num_items):
    print('Save Trajectory Video as \"%s\"' % (file_prefix + '.avi'))
    fps = 24  # frequency
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(file_prefix + '.avi', fourcc, fps,
                                   (500, 500))
    for i in range(num_items):
        img12 = cv2.imread(file_prefix + str(i) +'.jpg')
        video_writer.write(img12)
    video_writer.release()


if __name__ == '__main__':
    np.random.seed(1234)
    #env_name = 'Swimmer-v1'
    env_name = 'HalfCheetahSmallFoot-v0'
    #env_name = 'Walker2d-v1'
    #env_name = 'Ant-v1'
    #env_name = 'Humanoid-v1'
    demos = Demonstrations(1, 34, 23, 1000000007)
    demos.load('data/' + env_name, 25)
    demos.set(20)
    obss, acts, ts = demos.next_demo(normalize=False)
    #obss, acts = demos.next_demo()
    env = gym.make(env_name)
    show_animate_trajectory(env, obss, acts, True)
    #save_trajectory_images(env, obss, acts, 'HalfCheetah-v1/t1')
    #save_video('HalfCheetah-v1/t1/real', 1000)
