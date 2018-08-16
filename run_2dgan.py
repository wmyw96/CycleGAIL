import argparse

import numpy as np
import tensorflow as tf

from dataset import Demonstrations, NonLinearTransform, IdentityTransform
from envs.SpinDrive3D import SpinDrive3D
from CycleGAN import CycleGAN
from utils import greedy_reject_sampling, \
    spindrive3d_generate_random_trajectories, show_trajectory


###############################################################################
#                            CURRENT VERSION                                  #
###############################################################################


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--nhid', dest='nhid', type=int, default=10,
                    help='hidden size of the nn D,F,G')
parser.add_argument('--lg', dest='lambda_g', type=float, default=20.,
                    help='lambda value G')
parser.add_argument('--lf', dest='lambda_f', type=float, default=20.,
                    help='lambda value F')
parser.add_argument('--gamma', dest='gamma', type=float, default=20.,
                    help='gamma value')
parser.add_argument('--name', dest='name', type=str,
                    default='CycleGAIL', help='model name')
parser.add_argument('--loss', dest='loss', type=str, default='wgan',
                    help='wgan/wgan-gp/wgan-sn')

"""Arguments related to training"""
parser.add_argument('--loss_metric', dest='loss_metric', default='L1',
                    help='L1, or L2')
parser.add_argument('--lr', dest='lr', type=float, default=0.00005,
                    help='initial learning rate for adam')  # 0.0002
parser.add_argument('--epoch', dest='epoch', type=int, default=500,
                    help='# of epoch')
parser.add_argument('--clip', dest='clip', type=float, default=0.01,
                    help='clip value')
parser.add_argument('--n_c', dest='n_c', type=int, default=5,
                    help='n_critic')
parser.add_argument('--markov', dest='markov', type=int, default=0,
                    help='markov concat steps')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=300,
                    help='batch size')
parser.add_argument('--log_interval', dest='log_interval', type=int,
                    default=100,
                    help='length of log interval')

np.random.seed(1234)
tf.set_random_seed(1234)

args = parser.parse_args()

if True:
    def generate_2d():
        length = 100
        state = np.zeros((length, 2))
        init_pos = np.random.uniform(-0.5, 0.5)
        init_atk = np.random.uniform(-0.2, 0.2)

        style = np.random.uniform(-0.2, 0.2)
        style2 = np.random.uniform(-0.2, 0.2)
        for j in range(length):
            t = j * 0.05
            state[j, 0] = init_pos + np.sin(t + style) + \
                          np.random.uniform(-0.02, 0.02)
            state[j, 1] = (init_atk + t * np.cos(t + style2) +
                           np.random.uniform(-0.02, 0.02) - 2.5) / 2
        return state

    ident_map = IdentityTransform()
    enva, envb = None, None
    # sa, aa = demos_a.next_demo()
    # show_trajectory(enva, sa, aa)
    # sb, ab = demos_b.next_demo()
    # show_trajectory(envb, sb, ab)

    # with tf.Session() as sess:
    if True:
        model = CycleGAN(args.name, args, args.clip, enva, envb,
                         2, 2, args.nhid, 128,
                         lambda_g=args.lambda_g,
                         gamma=args.gamma,
                         metric=args.loss_metric,
                         checkpoint_dir=None,
                         loss=args.loss,
                         vis_mode='1d',
                         concat_steps=args.markov)
        print('Training Process:')
        model.train(args, generate_2d, generate_2d,
                    ita2b=ident_map)

'''
Recommended command:
python run_synthetic.py --name logs/csyn-wgan --lr 0.00005 --clip 0.01 --nhid 128 --loss_metric L2 --epoch 100000
python run_synthetic.py --name logs/csyn-wgan-gp-128 --lr 0.0001 --clip 0.01 --nhid 128 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 1.0 --lg 1.0 --lf 1.0 --loss wgan-gp
python run_synthetic.py --name logs/csyn-wgan-gp-30 --lr 0.0001 --clip 0.01 --nhid 10 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 1.0 --lg 1.0 --lf 1.0 --loss wgan-gp --nd1 50 --nd2 50
python run_synthetic.py --name logs/csyn-wgan-gp-30 --lr 0.0001 --clip 0.01 --nhid 10 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 0.0 --lg 1.0 --lf 1.0 --loss wgan-gp

python run_1d.py --name logs/1d --lr 0.0001 --nhid 30 --loss_metric L2 --epoch 100000 --gam 1.0 --lg 1.0 --lf 1.0 --loss wgan-gp

'''
