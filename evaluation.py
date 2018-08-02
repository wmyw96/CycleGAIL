import numpy as np


def run_policy_evaluation(ntrajs, envb, model, expert_a_loc):
    from policy_net import MlpPolicy

    rds = []
    for i in range(ntrajs):
        obs_b = envb.reset()
        #obs_b = np.concatenate([envb.env.model.data.qpos,
        #                        envb.env.model.data.qvel]).reshape(-1)
        done = False
        total_rd = 0.0
        while not done:
            obs_a, _ = model.run_trans('b2a', obs=obs_b)
            policy_obs_a = obs_a.reshape(-1)

            policy = MlpPolicy(expert_a_loc)
            act_a = policy.run(policy_obs_a)

            _, act_b = model.run_trans('a2b', act=act_a)
            obs_b, rd, done, _ = envb.step(act_b)
            #obs_b = np.concatenate([envb.env.model.data.qpos,
            #                        envb.env.model.data.qvel]).reshape(-1)
            total_rd += rd
        rds.append(total_rd)
        print('Rollout #%d: total_reward=%.3f\n' % (i, total_rd))

    print('Summary: reward = %.3f +/- %.3f' % (float(np.mean(rds)),
                                               float(np.std(rds))))
