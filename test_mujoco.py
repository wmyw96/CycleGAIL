import gym
import numpy as np

env_name = 'Walker2d-v1'

env = gym.make(env_name)

err = 0

for i in range(100):
    env.reset()
    print('Rollout %d' % i)
    obs = None
    for k in range(10):
        act = env.action_space.sample()
        obs, _, _, _ = env.step(act)
    act = env.action_space.sample()

    obss = []
    for j in range(100):
        qpos = np.concatenate([np.random.uniform(-1.0, 1.0, (1,)), obs[0:8]])
        qvel = obs[8:]
        env.reset()
        env.env.set_state(qpos, qvel)

        nobs, _, _, _ = env.step(act)
        obss.append(nobs.reshape(1, -1))

    obs_concat = np.concatenate(obss, 0)
    err += (np.sum(np.var(obs_concat, axis=0)))

print('Mean error = {}\n'.format(err))