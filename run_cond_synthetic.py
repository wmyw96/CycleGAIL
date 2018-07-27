import argparse

import numpy as np
import tensorflow as tf

from dataset import Demonstrations
from envs.SpinDrive3D import SpinDrive3D
from CondCycleGAIL import CondCycleGAIL
from utils import greedy_reject_sampling, \
    spindrive3d_generate_random_trajectories, show_trajectory


###############################################################################
#                            CURRENT VERSION                                  #
###############################################################################


def generate_auxiliary_samples(id, nitems, length, radius, alpha, dt, rg):
    print('Expert demonstration #%d (%d, %d, %.2f, %.2f, %.2f %.2f)' %
          (id, nitems, length, radius, alpha, dt, rg))
    demos = Demonstrations(1, 34, 23, 1000000007)
    name = 'data/SpinDrive3D_%d_%d_%.2f_%.2f_%.2f_%.2f' % \
           (nitems, length, radius, alpha, dt, rg)
    for i in range(nitems):
        state, action = \
            spindrive3d_generate_random_trajectories(length, radius,
                                                     alpha, dt, rg)
        demos.add_demo(state, action)
        print(name + ": trajectory %d finished!" % i)
    demos.save(name)


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--nhid', dest='nhid', type=int, default=10,
                    help='hidden size of the nn D,F,G')
parser.add_argument('--lg', dest='lambda_g', type=float, default=20.,
                    help='lambda value G')
parser.add_argument('--lf', dest='lambda_f', type=float, default=20.,
                    help='lambda value F')
parser.add_argument('--name', dest='name', type=str,
                    default='CycleGAIL', help='model name')
parser.add_argument('--loss', dest='loss', type=str, default='wgan',
                    help='wgan/wgan-gp/wgan-sn')

"""Arguments related to run mode"""
parser.add_argument('--mode', dest='mode', default='train',
                    help='gen, train, test')

"""Arguments related to training"""
parser.add_argument('--loss_metric', dest='loss_metric', default='L1',
                    help='L1, or L2')
parser.add_argument('--lr', dest='lr', type=float, default=0.00005,
                    help='initial learning rate for adam') #0.0002
parser.add_argument('--epoch', dest='epoch', type=int, default=500,
                    help='# of epoch')
parser.add_argument('--clip', dest='clip', type=float, default=0.01,
                    help='clip value')
parser.add_argument('--n_c', dest='n_c', type=int, default=5,
                    help='n_critic')
parser.add_argument('--orac', dest='orac', type=bool, default=True,
                    help='whether to use oracle loss')

"""Dataset setting"""
parser.add_argument('--ntraj', dest='ntraj', type=int, default=20,
                    help='# of trajctories in the data set')
parser.add_argument('--nd1', dest='nd1', type=int, default=10,
                    help='# of expert 1 trajectories for training')
parser.add_argument('--nd2', dest='nd2', type=int, default=10,
                    help='# of expert 2 trajectoreis for training')
parser.add_argument('--len', dest='len', type=int, default=300,
                    help='horizon of the trajectory (fixed)')

np.random.seed(1234)
tf.set_random_seed(1234)

args = parser.parse_args()

if args.mode == 'gen':
    generate_auxiliary_samples(1, args.ntraj, args.len, 5.0, 1.0, 0.05, 0.05)
    generate_auxiliary_samples(1, args.ntraj, args.len, 5.0, 2.0, 0.05, 0.05)


if args.mode == 'train':
    demos_a = Demonstrations(1, 34, 23, 1000000007)
    demos_b = Demonstrations(1, 23, 34, 1000000009)
    print('Training Init')
    print('Load data : Expert #1 Demonstrations')
    demos_a.load('data/SpinDrive3D_%d_%d_5.00_1.00_0.05_0.05' %
                 (args.ntraj, args.len), args.ntraj)
    print('Load data : Expert #2 Demonstrations')
    demos_b.load('data/SpinDrive3D_%d_%d_5.00_2.00_0.05_0.05' %
                 (args.ntraj, args.len), args.ntraj)
    demos_a.set(args.nd1)
    demos_b.set(args.nd2)
    print('Load data finished !')
    enva = SpinDrive3D(5, 1, 0.2)
    envb = SpinDrive3D(5, 2, 0.2)
    #sa, aa = demos_a.next_demo()
    #show_trajectory(enva, sa, aa)
    #sb, ab = demos_b.next_demo()
    #show_trajectory(envb, sb, ab)

    with tf.Session() as sess:
        model = CondCycleGAIL(args.name, sess, args.clip, enva, envb,
                              3, 3, 3, 3, args.nhid,
                              lambda_g=args.lambda_g,
                              lambda_f=args.lambda_f,
                              use_orac_loss=args.orac,
                              loss_metric=args.loss_metric,
                              checkpoint_dir=None,
                              loss=args.loss)
        print('Training Process:')
        model.train(args, demos_a, demos_b)

'''
Recommended command:
python run_synthetic.py --name logs/csyn-wgan --lr 0.00005 --clip 0.01 --nhid 128 --loss_metric L2 --epoch 100000
python run_synthetic.py --name logs/csyn-wgan-gp-128 --lr 0.0001 --clip 0.01 --nhid 128 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 0.0 --lg 1.0 --lf 1.0 --loss wgan-gp
python run_cond_synthetic.py --name logs/csyn-wgan-gp-30 --lr 0.0001 --clip 0.01 --nhid 10 --loss_metric L2 --epoch 100000 --ntraj 100 --lg 1.0 --lf 1.0 --loss wgan-gp --nd1 25 --nd2 25
python run_synthetic.py --name logs/csyn-wgan-gp-30 --lr 0.0001 --clip 0.01 --nhid 10 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 0.0 --lg 1.0 --lf 1.0 --loss wgan-gp

'''