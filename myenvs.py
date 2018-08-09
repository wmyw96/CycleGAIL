import numpy as np


class EnvWrapper(object):
    def __init__(self, env_name, env, trans_obs, trans_act):
        self.trans_obs = trans_obs
        self.trans_act = trans_act
        self.env = env
        self.env_name = env_name
        if env_name == 'HalfCheetah-v1' or \
            env_name == 'Walker2d-v1':
            self.init_qpos = env.env.init_qpos
            self.init_qvel = env.env.init_qvel

    def predict(self, gobs, gact):
        # obs: [N x od]
        # act: [N x ad]
        true_obs = self.trans_obs.inv_run(gobs)
        if true_obs is None:
            return None
        true_act = self.trans_act.inv_run(gact)
        if true_act is None:
            return None
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
            next_obs[i, :] = self.trans_obs.run(nobs)
        return next_obs

    def reset(self):
        return self.env.reset()

    def step(self, act):
        act = self.trans_act.inv_run(act.reshape(1, -1))
        obs, _2, _3, _4 = self.env.step(act.reshape(-1))
        _1 = self.trans_obs.run(obs.reshape(1, -1))
        return _1, _2, _3, _4
