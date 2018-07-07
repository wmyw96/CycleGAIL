from SpinDrive3D import SpinDrive3D
from utils import greedy_reject_sampling

# hyperparameters
radius = 5
alpha = 5
valid_range = 0.2
stepsize = 0.05 * radius


def generate_one_trajectory(radius, alpha, valid_range, stepsize):
    # set up the environment
    env = SpinDrive3D(radius, alpha, valid_range)
    obs = env.reset()

    acts = []
    obss = []
    for i in range(500):
        act = greedy_reject_sampling(obs, alpha, radius, valid_range, stepsize)
        obss.append(obs)
        acts.append(act)
        obs, _, __, ___ = env.step(act)

    env.render()
    traj = {'obs': obss, 'act': acts}
    return traj

generate_one_trajectory(radius, alpha, valid_range, stepsize)
