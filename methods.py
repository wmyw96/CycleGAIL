import scipy.optimize
import numpy as np


def greedy_search_refactor(env, obs_array, act_array):
    horizon = obs_array.shape[0]

    env.reset()
    cur_obs = obs_array[0, :]
    env.env.set_state(cur_obs[: 9], cur_obs[9:])

    robs = np.zeros_like(obs_array)
    ract = np.zeros_like(act_array)

    reward_sum = 0

    for i in range(horizon - 1):
        nxt_obs = obs_array[i + 1, :]
        cur_act = act_array[i, :]
        robs[i] = cur_obs

        def loss_func(x, debug=False):
            loss_act = np.sum((x - cur_act) * (x - cur_act))

            env.reset()
            env.env.set_state(obs_array[0, :9], obs_array[0, 9:])
            for j in range(i):
                env.step(ract[j, :])

            env.step(x)
            nobs = np.concatenate([env.env.model.data.qpos,
                                   env.env.model.data.qvel]).reshape(-1)
            loss_obs = np.sum((nobs - nxt_obs) * (nobs - nxt_obs))
            if debug:
                print('Debug', loss_obs)
            return loss_obs

        opt_result = scipy.optimize.minimize(loss_func, cur_act,
                                             method='BFGS')
        act = opt_result.x
        print(loss_func(act))
        ract[i, :] = act

        # recover
        env.reset()
        env.env.set_state(obs_array[0, :9], obs_array[0, 9:])
        for j in range(i):
            env.step(act_array[j, :])

        _1, rd, _2, _3 = env.step(act)
        reward_sum += rd
        cur_obs = np.concatenate([env.env.model.data.qpos,
                                 env.env.model.data.qvel]).reshape(-1)
        if (i + 1) % 10 == 0:
            print('Refactoring Demonstrations: Step %d finished' % (i + 1))

    robs[horizon - 1, :] = cur_obs

    print('Refactored Reward sum: %.5f\n' % reward_sum)
    return robs, ract


def get_std(x):
    x_m1 = np.mean(x, axis=0)
    x_m2 = np.mean(x * x, axis=0)
    x_v = x_m2 - x_m1 * x_m1
    return np.sqrt(x_v)


def calc_reward(obs, act):
    env.reset()
    env.env.set_state(obs[0, :9], obs[0, 9:])
    reward_sum = 0
    for i in range(act.shape[0] - 1):
        _1, rd, _2, _3 = env.step(act[i, :])
        reward_sum += rd
    return reward_sum


def get_true_obs(obs, act):
    env.reset()
    env.env.set_state(obs[0, :9], obs[0, 9:])
    true_obs = np.zeros_like(obs)
    for i in range(act.shape[0]):
        nobs = np.concatenate([env.env.model.data.qpos,
                               env.env.model.data.qvel]).reshape(-1)
        true_obs[i, :] = nobs
        env.step(act[i, :])
    return true_obs


if __name__  == '__main__':
    import gym
    from dataset import Demonstrations

    env_name = "Walker2d-v1"
    env = gym.make(env_name)
    demos = Demonstrations(1, 34, 23, 1000000007)
    demos.load('data/' + env_name, 100)
    demos.set(50)
    obss, acts = demos.next_demo()

    std_act = get_std(acts).reshape((1, -1))
    std_obs = get_std(obss).reshape((1, -1))

    noised_obss = np.random.uniform(-1, 1, obss.shape) * std_obs * 1e-3
    noised_acts = np.random.uniform(-1, 1, acts.shape) * std_act * 1e-3

    noised_acts += acts
    noised_obss += obss

    noised_obss_t = get_true_obs(noised_obss, noised_acts)

    print('Original reward = %.5f\n' % calc_reward(obss, acts))
    print('Noised reward = %.5f\n' % calc_reward(noised_obss, noised_acts))
    solved_obs, solved_acts = \
        greedy_search_refactor(env, obss, noised_acts)

    print(np.sum((noised_obss_t - obss) * (noised_obss_t - obss)))
    print(np.sum((solved_obs - obss) * (solved_obs - obss)))
    print(np.sum((solved_obs - obss) * (solved_obs - obss), axis=1))