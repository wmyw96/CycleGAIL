import argparse

import numpy as np
import tensorflow as tf

from CycleGAIL import CycleGAIL
from dataset import Demonstrations
from envs.SpinDrive3D import SpinDrive3D
from utils import greedy_reject_sampling, \
    spindrive3d_generate_random_trajectories, show_trajectory

###############################################################################
#                                   LEGACY                                    #
###############################################################################


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

    traj = {'obs': obss, 'act': acts}
    return env, traj


def generate_demos(name, nitems, radius, alpha, valid_range, stepsize):
    demos = Demonstrations(1, 34, 23, 1000000007)
    for i in range(nitems):
        env, traj = generate_one_trajectory(radius, alpha,
                                            valid_range, stepsize)
        state = np.array(traj['obs'])
        action = np.array(traj['act'])
        demos.add_demo(state, action)
        print(name + ": trajectory %d finished!" % i)
        # show_trajectory(env, state, action)
    demos.save(name)


def generate_expert_behaviors(nitems):
    print('Start generating demostrations:')
    radius_1 = 5
    alpha_1 = 1
    valid_range_1 = 0.2
    stepsize_1 = 0.05 * radius_1
    print('Expert Demostration #1 (%d, %.2f, %.2f, %.2f)' %
          (nitems, radius_1, alpha_1, valid_range_1))
    generate_demos('SpinDrive3D_%d_5_1_0.2' % nitems,
                   nitems, radius_1, alpha_1, valid_range_1, stepsize_1)

    radius_2 = 5
    alpha_2 = 5
    valid_range_2 = 0.2
    stepsize_2 = 0.05 * radius_2
    print('Expert Demostration #2 (%d, %.2f, %.2f, %.2f)' %
          (nitems, radius_2, alpha_2, valid_range_2))
    generate_demos('SpinDrive3D_%d_5_5_0.2' % nitems,
                   nitems, radius_2, alpha_2, valid_range_2, stepsize_2)

###############################################################################
#                            CURRENT VERSION                                  #
###############################################################################


def generate_auxiliary_samples(id, nitems, radius, alpha, dt, rg):
    print('Expert demonstration #%d (%d, %.2f, %.2f, %.2f %.2f)' %
          (id, nitems, radius, alpha, dt, rg))
    demos = Demonstrations(1, 34, 23, 1000000007)
    name = 'data/SpinDrive3D_%d_%.2f_%.2f_%.2f_%.2f' % \
           (nitems, radius, alpha, dt, rg)
    for i in range(nitems):
        state, action = \
            spindrive3d_generate_random_trajectories(300, radius,
                                                     alpha, dt, rg)
        demos.add_demo(state, action)
        print(name + ": trajectory %d finished!" % i)
    demos.save(name)


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--dhid', dest='dhid', type=int, default=30,
                    help='hidden size of the discriminator D')
parser.add_argument('--ghid', dest='ghid', type=int, default=10,
                    help='hidden size of the generator G')
parser.add_argument('--fhid', dest='fhid', type=int, default=10,
                    help='hidden size of the generator F')
parser.add_argument('--dlayer', dest='dlayer', type=int, default=3,
                    help='hidden size of the discriminator D')
parser.add_argument('--glayer', dest='glayer', type=int, default=2,
                    help='hidden size of the generator G')
parser.add_argument('--flayer', dest='flayer', type=int, default=2,
                    help='hidden size of the generator F')
parser.add_argument('--lga', dest='lambda_g_a', type=float, default=20.,
                    help='lambda value G(a)')
parser.add_argument('--lgb', dest='lambda_g_b', type=float, default=20.,
                    help='lambda value G(b)')
parser.add_argument('--lfa', dest='lambda_f_a', type=float, default=20.,
                    help='lambda value F(a)')
parser.add_argument('--lfb', dest='lambda_f_b', type=float, default=20.,
                    help='lambda value F(b)')

"""Arguments related to run mode"""
parser.add_argument('--mode', dest='mode', default='train',
                    help='gen, train, test')

"""Arguments related to training"""
parser.add_argument('--loss_metric', dest='loss_metric', default='L1',
                    help='L1, or L2')
parser.add_argument('--lr', dest='lr', type=float, default=0.005,
                    help='initial learning rate for adam') #0.0002
parser.add_argument('--epoch', dest='epoch', type=int, default=500,
                    help='# of epoch')
parser.add_argument('--round', dest='round', type=int, default=20,
                    help='# trajectories in a epoch')
parser.add_argument('--clip', dest='clip', type=float, default=0.2,
                    help='clip value')
parser.add_argument('--ntraj', dest='ntraj', type=int, default=20,
                    help='# of trajctories in the data set')

args = parser.parse_args()

if args.mode == 'gen':
    generate_auxiliary_samples(1, args.ntraj, 5.0, 1.0, 0.05, 0.05)
    generate_auxiliary_samples(1, args.ntraj, 5.0, 2.0, 0.05, 0.05)


if args.mode == 'train':
    demos_a = Demonstrations(1, 34, 23, 1000000007)
    demos_b = Demonstrations(1, 23, 34, 1000000009)
    print('Training Init')
    print('Load data : Expert #1 Demonstrations')
    demos_a.load('data/SpinDrive3D_%d_5.00_1.00_0.05_0.05' % args.ntraj,
                 args.ntraj)
    print('Load data : Expert #2 Demonstrations')
    demos_b.load('data/SpinDrive3D_%d_5.00_2.00_0.05_0.05' % args.ntraj,
                 args.ntraj)
    print('Load data finished !')
    enva = SpinDrive3D(5, 1, 0.2)
    envb = SpinDrive3D(5, 2, 0.2)

    #sa, aa = demos_a.next_demo()
    #show_trajectory(enva, sa, aa)
    #sb, ab = demos_b.next_demo()
    #show_trajectory(envb, sb, ab)

    def init_align(x):
        return x
    act_space_a = [3]
    act_space_b = [3]
    obs_space_a = [3]
    obs_space_b = [3]
    archi_d_a = archi_d_b = [args.dhid] * args.dlayer
    archi_g_ab = archi_g_ba = [args.ghid] * args.glayer
    archi_f_ab = archi_f_ba = [args.fhid] * args.flayer

    with tf.Session() as sess:
        model = CycleGAIL(sess, args.clip, 5, enva, envb, init_align,
                          act_space_a, act_space_b, obs_space_a, obs_space_b,
                          archi_d_a, archi_d_b, archi_g_ab, archi_g_ba,
                          archi_f_ab, archi_f_ba,
                          lambda_g_a=args.lambda_g_a,
                          lambda_g_b=args.lambda_g_b,
                          lambda_f_a=args.lambda_f_a,
                          lambda_f_b=args.lambda_f_b,
                          loss_metric=args.loss_metric,
                          checkpoint_dir=None)
        print('Training Process:')
        model.train(args, demos_a, demos_b, 4)
