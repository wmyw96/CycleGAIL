import argparse

import numpy as np
import tensorflow as tf
import gym

from dataset import Demonstrations
from dataset import IdentityTransform, LinearTransform, \
    NonLinearTransform, ConcatTransform
from CycleJointCAIL import CycleGAIL

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--nhidf', dest='nhidf', type=int, default=10,
                    help='hidden size of the nn F')
parser.add_argument('--nhidg', dest='nhidg', type=int, default=10,
                    help='hidden size of the nn G')
parser.add_argument('--nhidd', dest='nhidd', type=int, default=10,
                    help='hidden size of the nn D')
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
parser.add_argument('--enva', dest='enva', type=str,
                    default='HalfCheetah-v1', help='environment A name')
parser.add_argument('--envb', dest='envb', type=str,
                    default='HalfCheetah-v1', help='environment B name')
parser.add_argument("--ckdir", dest='ckdir', type=str,
                    default='aaa', help='checkpoint direction')
parser.add_argument('--markov', dest='markov', type=int, default=0,
                    help='markov concat steps')

"""Arguments related to run mode"""
parser.add_argument('--mode', dest='mode', default='train',
                    help='gen, train, test')
parser.add_argument('--exp', dest='exp', default='identity',
                    help='experiment type: identity, linear, nonlinear, real')

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
parser.add_argument('--log_interval', dest='log_interval', type=int, default=1,
                    help='length of log interval')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--seq_len', dest='seq_len', type=int, default=100,
                    help='seq len')

"""Dataset setting"""
parser.add_argument('--ntraj', dest='ntraj', type=int, default=20,
                    help='# of trajctories in the data set')
parser.add_argument('--nd1', dest='nd1', type=int, default=10,
                    help='# of expert 1 trajectories for training')
parser.add_argument('--nd2', dest='nd2', type=int, default=10,
                    help='# of expert 2 trajectoreis for training')
parser.add_argument('--len', dest='len', type=int, default=300,
                    help='horizon of the trajectory (fixed)')
parser.add_argument('--obsdim', dest='obsdim', type=int, default=17,
                    help='observation space dim (for linear & nonlinear trans)')
parser.add_argument('--actdim', dest='actdim', type=int, default=6,
                    help='action space dim (for linear & nonlinear trans)')
"""Demo setting"""
parser.add_argument('--expert_a', dest='expert_a', type=str, default=None,
                    help='policy net file of expert a')
parser.add_argument('--expert_b', dest='expert_b', type=str, default=None,
                    help='policy net file of expert a')


np.random.seed(1234)
tf.set_random_seed(1234)

args = parser.parse_args()

trans_obs = None
trans_act = None

expert_dira = 'data/'
expert_dirb = 'data/'

if args.exp == 'identity':
    trans_act = IdentityTransform()
    trans_obs = IdentityTransform()
    expert_dirb += '2-'
if args.exp == 'linear':
    trans_act = LinearTransform(args.actdim)
    trans_obs = LinearTransform(args.obsdim)
if args.exp == 'nonlinear':
    trans_act = NonLinearTransform(args.actdim, 2)
    trans_obs = NonLinearTransform(args.obsdim, 2)
if args.exp == 'real':
    expert_dira += 'T_'
    expert_dirb += 'T_'

ideal_trans = ConcatTransform(trans_obs, trans_act, args.obsdim)

demos_a = Demonstrations(1, 34, 23, 1000000007)
demos_b = Demonstrations(1, 23, 34, 1000000009, trans_obs, trans_act)
print('Init')
print('Load data : Expert #1 Demonstrations')
demos_a.load(expert_dira + args.enva, args.ntraj)
print('Load data : Expert #2 Demonstrations')
demos_b.load(expert_dirb + args.envb, args.ntraj)
demos_a.set(args.nd1)
demos_b.set(args.nd2)
demos_a.set_bz(args.batch_size, args.seq_len)
demos_b.set_bz(args.batch_size, args.seq_len)
try:
    enva = gym.make(args.enva)
    envb = gym.make(args.envb)
except:
    print('Unable to load environment!')
    enva = envb = None
print('Load data finished !')

model = CycleGAIL(args.name, args, args.clip, enva, envb,
                  demos_a.obs_dim, demos_a.act_dim,
                  demos_b.obs_dim, demos_b.act_dim,
                  args.nhidg, args.nhidd,
                  lambda_g=args.lambda_g,
                  gamma=args.gamma,
                  metric=args.loss_metric,
                  checkpoint_dir=None,
                  loss=args.loss,
                  vis_mode='mujoco',
                  concat_steps=args.markov)
print('Training Process:')
if args.mode == 'train':
    model.train(args, demos_a, demos_b, args.ckdir, ideal_trans)
else:
    raise NotImplementedError

'''
'''