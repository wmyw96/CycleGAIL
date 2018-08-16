import argparse

import numpy as np
import tensorflow as tf

from dataset import Demonstrations, NonLinearTransform, IdentityTransform
from envs.SpinDrive3D import SpinDrive3D
from CycleGAIL import CycleGAIL
from utils import greedy_reject_sampling, \
    spindrive3d_generate_random_trajectories, show_trajectory


###############################################################################
#                            CURRENT VERSION                                  #
###############################################################################


def generate_state_action_pair(len):
    state = np.zeros((len, 1))
    action = np.zeros((len, 1))

    init_pos = np.random.uniform(-0.5, 0.5)
    init_atk = np.random.uniform(-0.2, 0.2)

    style = np.random.uniform(-0.2, 0.2)
    style2 = np.random.uniform(-0.2, 0.2)
    for j in range(len):
        t = j * 0.05
        state[j, :] = init_pos + np.sin(t + style) + \
                      np.random.uniform(-0.02, 0.02)
        action[j, :] = init_atk + t + \
                       np.random.uniform(-0.02, 0.02)
    return state, action
#* np.cos(t + style2) + \

def generate_trajs(ntrajs, len, demos_a, demos_b, obs_trans, act_trans):
    for i in range(ntrajs):
        state, action = generate_state_action_pair(len)
        demos_a.add_demo(state, action)
        state2, action2 = generate_state_action_pair(len)
        state2 = obs_trans.run(state2)
        action2 = act_trans.run(action2)
        demos_b.add_demo(state2, action2)
    return demos_a, demos_b


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

"""Arguments related to run mode"""
parser.add_argument('--mode', dest='mode', default='train',
                    help='gen, train, test')

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

if args.mode == 'train':
    demos_a = Demonstrations(1, 34, 23, 1000000007)
    demos_b = Demonstrations(1, 23, 34, 1000000009)

    obs_trans = IdentityTransform() #NonLinearTransform(1, 10)
    act_trans = IdentityTransform() #NonLinearTransform(1, 10)
    generate_trajs(1000, 100, demos_a, demos_b, obs_trans, act_trans)

    demos_a.normalize()
    demos_b.normalize()
    demos_a.set(900)
    demos_b.set(900)

    demos_a.set_bz(100)
    demos_b.set_bz(100)

    enva, envb = None, None
    # sa, aa = demos_a.next_demo()
    # show_trajectory(enva, sa, aa)
    # sb, ab = demos_b.next_demo()
    # show_trajectory(envb, sb, ab)

    # with tf.Session() as sess:
    if True:
        model = CycleGAIL(args.name, args, args.clip, enva, envb,
                          1, 1, 1, 1, args.nhid, args.nhid, 128,
                          demos_a.obs_scalar, demos_b.obs_scalar,
                          demos_a.act_scalar, demos_b.act_scalar,
                          lambda_g=args.lambda_g,
                          lambda_f=args.lambda_f,
                          gamma=args.gamma,
                          use_orac_loss=False,
                          metric=args.loss_metric,
                          checkpoint_dir=None,
                          loss=args.loss,
                          vis_mode='1d',
                          concat_steps=args.markov)
        print('Training Process:')
        model.train(args, demos_a, demos_b,
                    ita2b_obs=obs_trans, ita2b_act=act_trans)

'''
Recommended command:
python run_synthetic.py --name logs/csyn-wgan --lr 0.00005 --clip 0.01 --nhid 128 --loss_metric L2 --epoch 100000
python run_synthetic.py --name logs/csyn-wgan-gp-128 --lr 0.0001 --clip 0.01 --nhid 128 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 1.0 --lg 1.0 --lf 1.0 --loss wgan-gp
python run_synthetic.py --name logs/csyn-wgan-gp-30 --lr 0.0001 --clip 0.01 --nhid 10 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 1.0 --lg 1.0 --lf 1.0 --loss wgan-gp --nd1 50 --nd2 50
python run_synthetic.py --name logs/csyn-wgan-gp-30 --lr 0.0001 --clip 0.01 --nhid 10 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 0.0 --lg 1.0 --lf 1.0 --loss wgan-gp

python run_1d.py --name logs/1d --lr 0.0001 --nhid 30 --loss_metric L2 --epoch 100000 --gam 1.0 --lg 1.0 --lf 1.0 --loss wgan-gp

'''
