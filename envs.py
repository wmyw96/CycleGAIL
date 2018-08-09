import numpy as np


class EnvWrapper(object):
    def __init__(self, env_name, env, trans_obs, trans_act):
        self.trans_obs = trans_obs
        self.trans_act = trans_act
        self.env = env
        self.env_name = env_name

    def predict(self, gobs, gact):
        # obs: [N x od]
        # act: [N x ad]
        true_obs = self.trans_obs.inv_run(gobs)
        true_act = self.trans_act.inv_run(gact)
        next_obs = np.zeros_like(true_obs)
        for i in range(true_obs.shape[0]):
            self.env.reset()
            obs = true_obs[i, :]
            act = true_act[i, :]
            if self.env_name == 'HalfCheetah-v1' or \
                self.env_name == 'Walker2d-v1':
                qpos = np.concatenate(
                    [np.random.uniform(-1.0, 1.0, (1,)), obs[0:8]])
                qvel = obs[8:]
                self.env.env.set_state(qpos, qvel)
            else:
                raise NotImplementedError
            nobs, _, _, _ = self.env.step(act)
            next_obs[i, :] = self.trans_obs(nobs)
        return next_obs
