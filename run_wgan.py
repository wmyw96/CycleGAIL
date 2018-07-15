from SNWGAN import WGAN
import argparse
import tensorflow as tf
import numpy as np

import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc
import random

'''
class WassersteinGAN(object):
    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
                .minimize(self.d_loss_reg, var_list=self.d_net.vars)
            self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
                .minimize(self.g_loss_reg, var_list=self.g_net.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=64, num_batches=1000000):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            if t % 500 == 0 or t < 25:
                 d_iters = 100

            for _ in range(0, d_iters):
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)
                self.sess.run(self.d_clip)
                self.sess.run(self.d_rmsprop, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim)
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.x: bx})

            if t % 100 == 0:
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz, self.x: bx}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss - g_loss, g_loss))

            if t % 100 == 0:
                bz = self.z_sampler(batch_size, self.z_dim)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)

                figure = plt.figure(figsize=(8, 8))
                ax1 = figure.add_subplot(221)
                ax1.scatter(data[:, 0], data[:, 1], color='r', s=0.5)
                n = data.shape[0]
                z = np.random.uniform(0, 1, [n, self.n_z])
                fake_data = self.sess.run(self.fake_a, feed_dict={self.z: z})
                ax1.scatter(fake_data[:, 0], fake_data[:, 1], color='b', s=0.5)
                plt.show()
'''


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--name', dest='name', type=str,
                    default='WGAN', help='model name')
parser.add_argument('--scale', dest='scale', type=float, default=0.1,
                    help='scale') #0.0002

"""Arguments related to training"""
parser.add_argument('--loss_metric', dest='loss_metric', default='L1',
                    help='L1, or L2')
parser.add_argument('--lr', dest='lr', type=float, default=0.002,
                    help='initial learning rate for adam') #0.0002
parser.add_argument('--epoch', dest='epoch', type=int, default=500,
                    help='# of epoch')
parser.add_argument('--round', dest='round', type=int, default=20,
                    help='# trajectories in a epoch')
parser.add_argument('--clip', dest='clip', type=float, default=0.01,
                    help='clip value')
parser.add_argument('--ntraj', dest='ntraj', type=int, default=20,
                    help='# of trajctories in the data set')
parser.add_argument('--n_c', dest='n_c', type=int, default=5,
                    help='n_critic')
parser.add_argument('--n', dest='n', type=int, default=100,
                    help='n')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--loss', dest='loss', type=str, default='wgan',
                    help='wgan / wgan-gp / wgan-sn')

args = parser.parse_args()


def generate_data(n):
    ncluster = 8
    sigma = 0.01
    points = [[-1, 0], [1, 0], [0, 1], [0, -1], [0.707, 0.707], [0.707, -0.707],
       [-0.707, 0.707], [-0.707, -0.707]]
    data = np.zeros((n, 2))
    for i in range(n):
        x, y = random.choice(points)
        data[i, 0] = np.random.normal(x, sigma, 1)
        data[i, 1] = np.random.normal(y, sigma, 1)
    return data


np.random.seed(1234)
tf.set_random_seed(1234)

# run

with tf.Session() as sess:
    model = WGAN(args.loss, sess, args.clip, 256, 2,
                 loss=args.loss, scale=args.scale)
    model.train(args, generate_data)

'''
Recommend commands:
python run_wgan.py --lr 0.00005 --n 10000 --ghid 128 --dhid 128 --loss wgan --clip 0.01 --epoch 100000
python run_synthetic.py --name logs/csyn-wgan-gp --lr 0.0001 --clip 0.01 --nhid 10 --loss_metric L2 --epoch 100000 --ntraj 100 --gam 1.0 --lg 1.0 --lf 1.0 --loss wgan-gp
'''