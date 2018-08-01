import argparse

import numpy as np
import tensorflow as tf
import gym

from dataset import Demonstrations
from CycleGAIL import CycleGAIL

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
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100,
                    help='batch size')

"""Dataset setting"""
parser.add_argument('--ntraj', dest='ntraj', type=int, default=20,
                    help='# of trajctories in the data set')
parser.add_argument('--nd1', dest='nd1', type=int, default=10,
                    help='# of expert 1 trajectories for training')
parser.add_argument('--nd2', dest='nd2', type=int, default=10,
                    help='# of expert 2 trajectoreis for training')
parser.add_argument('--len', dest='len', type=int, default=300,
                    help='horizon of the trajectory (fixed)')
"""Demo setting"""
parser.add_argument('--expert_a', dest='expert_a', type=str, default=None,
                    help='policy net file of expert a')

np.random.seed(1234)
tf.set_random_seed(1234)

args = parser.parse_args()

demos_a = Demonstrations(1, 34, 23, 1000000007)
demos_b = Demonstrations(1, 23, 34, 1000000009)
print('Init')
print('Load data : Expert #1 Demonstrations')
demos_a.load('data/' + args.enva, args.ntraj)
print('Load data : Expert #2 Demonstrations')
demos_b.load('data/' + args.envb, args.ntraj)
demos_a.set(args.nd1)
demos_b.set(args.nd2)
demos_a.set_bz(args.batch_size)
demos_b.set_bz(args.batch_size)
try:
    enva = gym.make(args.enva)
    envb = gym.make(args.envb)
except:
    print('Unable to load environment!')
    enva = envb = None
print('Load data finished !')

model = CycleGAIL(args.name, None, args.clip, enva, envb,
                      6, 6, 18, 18, args.nhidf, args.nhidg, args.nhidd,
                      demos_a.obs_scalar, demos_b.obs_scalar,
                      demos_a.act_scalar, demos_b.act_scalar,
                      lambda_g=args.lambda_g,
                      lambda_f=args.lambda_f,
                      gamma=args.gamma,
                      use_orac_loss=False,
                      metric=args.loss_metric,
                      checkpoint_dir=None,
                      loss=args.loss,
                      vis_mode='mujoco',
                      concat_steps=args.markov)
print('Training Process:')
if args.mode == 'train':
    model.train(args, demos_a, demos_b, False, args.ckdir)
else:
    model.link_demo(demos_a, demos_b)
    model.load(args.ckdir)
    #model.evaluation(demos_a, demos_b, args.ckdir)

    # test a->b trans

    from policy_net import MlpPolicy
    envb.reset()
    obs_b = np.concatenate([envb.env.model.data.qpos,
                            envb.env.model.data.qvel]).reshape(-1)
    done = False
    total_rd = 0.0
    while not done:
        obs_a, _ = model.run_trans('b2a', obs=obs_b)
        policy_obs_a = obs_a.reshape(-1)[1:]

        policy = MlpPolicy(args.expert_a)
        act_a = policy.run(policy_obs_a)

        _, act_b = model.run_trans('a2b', act=act_a)
        _1, rd, done, _ = envb.step(act_b)
        obs_b = np.concatenate([envb.env.model.data.qpos,
                                envb.env.model.data.qvel]).reshape(-1)
        envb.render()
        total_rd += rd
    print('Total reward = %d\n' % total_rd)



'''
Demo:


Recommended command:
python run_mujoco.py --name logs/halfcheetah-ident --lr 0.0001 --nhidd 256 --nhidf 54 --nhidg 18 --loss_metric L2 --epoch 100000 --ntraj 100 --lf 1.0 --lg 1.0 --loss wgan-gp --nd1 50 --nd2 50 --enva HalfCheetah-v1 --envb HalfCheetah-v1 --ckdir model --mode test
python run_mujoco.py --name logs/halfcheetah-ident --lr 0.0001 --nhidd 256 --nhidf 54 --nhidg 18 --loss_metric L2 --epoch 100000 --ntraj 100 --lf 1.0 --lg 1.0 --loss wgan-gp --nd1 90 --nd2 90 --enva HalfCheetah-v1 --envb HalfCheetah-v1 --ckdir model

CUDA_VISIBLE_DEVICES=6 python run_mujoco.py --name logs/halfcheetah-ident --lr 0.0001 --nhidd 256 --nhidf 54 --nhidg 18 --loss_metric L2 --epoch 200000 --ntraj 100 --lf 0 --lg 0 --loss wgan-gp --nd1 50 --nd2 50 --enva HalfCheetah-v1 --envb HalfCheetah-v1 --ckdir model --mode train
CUDA_VISIBLE_DEVICES=5 python run_mujoco.py --name logs/halfcheetah-ident --lr 0.0001 --nhidd 256 --nhidf 54 --nhidg 18 --loss_metric L2 --epoch 200000 --ntraj 100 --lf 1.0 --lg 1.0 --loss wgan-gp --nd1 50 --nd2 50 --enva HalfCheetah-v1 --envb HalfCheetah-v1 --ckdir model2 --mode train


'''
