import os
import time
import tensorflow as tf
import numpy as np
from utils import *
from sn import spectral_normed_weight
import warnings
from mujoco_utils import *


def dense(inp, out_dim, activation, name, std, reuse):
    with tf.variable_scope(name, reuse=reuse):
        weight_shape = [int(inp.get_shape()[1]), out_dim]
        w = tf.get_variable('w', weight_shape)
        w_t = w + tf.random_normal(shape=weight_shape,
                                   mean=0.0, stddev=std, dtype=tf.float32)
        b = tf.get_variable('b', [out_dim],
                            initializer=tf.constant_initializer(0))
        b_t = b + tf.random_normal(shape=[out_dim],
                                   mean=0.0, stddev=std, dtype=tf.float32)
        if activation is None:
            return tf.nn.bias_add(tf.matmul(inp, w_t), b_t)
        else:
            return activation(tf.nn.bias_add(tf.matmul(inp, w_t), b_t))


class CycleGAN(object):
    def __init__(self, name, args, clip, env_a, env_b,
                 a_dim, b_dim, hidden_g, hidden_d,
                 lambda_g, gamma, metric='L1',
                 checkpoint_dir=None, loss='wgan',
                 vis_mode='synthetic', concat_steps=0):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        self.clip = clip
        self.env_a = env_a
        self.env_b = env_b
        self.vis_mode = vis_mode
        self.metric = metric
        self.checkpoint_dir = checkpoint_dir
        self.dir_name = name
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.use_spect = loss == 'wgan-sn'
        self.hidden_g = hidden_g
        self.hidden_d = hidden_d
        self.lambda_g = lambda_g
        self.loss = loss
        self.gamma = gamma
        self.concat_steps = concat_steps
        self.expert_a = None
        self.expert_b = None
        print('FFFFFFFFFFFFFFFFFF')
        print('======= Settings =======')
        print('-------- Models --------')
        print('GAN: %s\nclip: %.3f\nG lambda %.3f\n'
              'Loss metric: %s\nGamma: %.3f\nG Hidden size: '
              '%d\nD Hidden size: %d\nConcat steps %d'
              % (loss, clip, lambda_g, metric,
                 gamma, hidden_g, hidden_d, concat_steps))
        print('----- Environments -----')
        print('Domain A: %d\nDomain B: %d\n' % (a_dim, b_dim))

        print('CycleGAIL: Start building graph ...')
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.build_model(args)
        print('CycleGAIL: Build graph finished !')

    def markov(self, current):
        stacks = []
        for i in range(self.concat_steps):
            stacks.append(current[i: -self.concat_steps + i, :])
        stacks.append(current[self.concat_steps:, :])
        return tf.concat(stacks, axis=1)

    def graident_penalty(self, name, real, fake):
        alpha = tf.random_uniform([tf.shape(real)[0], 1], 0., 1.)
        hat = self.markov(alpha * real + ((1 - alpha) * fake))
        critic_hat_a = self.dis_net(name, hat)
        gradients = tf.gradients(critic_hat_a, [hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        return tf.reduce_mean((slopes - 1) ** 2)

    def build_model(self, args):
        self.real_a = tf.placeholder(tf.float32, [None, self.a_dim])
        self.real_b = tf.placeholder(tf.float32, [None, self.b_dim])
        self.std = tf.placeholder(tf.float32, shape=())
        self.ts = tf.placeholder(tf.float32, [None, 1])

        self.fake_a = \
            tf.concat([self.gen_net('g_a', self.real_b[:, :1], 1, self.std, False),
                       self.gen_net('g_a_2', self.real_b[:, 1:], self.a_dim - 1, self.std, False)], axis=1)
        self.fake_b = \
            tf.concat([self.gen_net('g_b', self.real_a[:, :1], 1, self.std, False),
                       self.gen_net('g_b_2', self.real_a[:, 1:], self.b_dim - 1, self.std, False)], axis=1)
        self.inv_a = \
            tf.concat([self.gen_net('g_a', self.fake_b[:, :1], 1, self.std),
                       self.gen_net('g_a_2', self.fake_b[:, 1:], self.a_dim - 1, self.std)], axis=1)
        self.inv_b = \
            tf.concat([self.gen_net('g_b', self.fake_a[:, :1], 1, self.std),
                       self.gen_net('g_b_2', self.fake_a[:, 1:], self.b_dim - 1, self.std)], axis=1)

        self.cycle_a = \
            cycle_loss(self.real_a, self.inv_a, self.metric)
        self.cycle_b = \
            cycle_loss(self.real_b, self.inv_b, self.metric)

        self.d_real_a = self.dis_net('d_a', self.markov(self.real_a), False)
        self.d_real_b = self.dis_net('d_b', self.markov(self.real_b), False)
        self.d_fake_a = self.dis_net('d_a', self.markov(self.fake_a))
        self.d_fake_b = self.dis_net('d_b', self.markov(self.fake_b))

        self.wdist_a = tf.reduce_mean(self.d_real_a - self.d_fake_a)
        self.wdist_b = tf.reduce_mean(self.d_real_b - self.d_fake_b)
        self.wdist = self.wdist_a + self.wdist_b
        self.loss_d = - self.wdist_a - self.wdist_b
        if self.loss == 'wgan-gp':
            self.gp = self.graident_penalty('d_a', self.real_a, self.fake_a)
            self.gp += self.graident_penalty('d_b', self.real_b, self.fake_b)
            self.loss_d += self.gp * self.gamma
        self.loss_gf_a = -tf.reduce_mean(self.d_fake_a)
        self.loss_gf_b = -tf.reduce_mean(self.d_fake_b)

        self.loss_g = self.loss_gf_a + self.loss_gf_b + \
            self.lambda_g * (self.cycle_a + self.cycle_b)

        self.loss_ident = \
            cycle_loss(self.fake_a, self.real_b, self.metric) + \
            cycle_loss(self.fake_b, self.real_a, self.metric)

        self.loss_best_g = \
            cycle_loss(self.real_a, self.real_b, self.metric)

        t_vars = tf.trainable_variables()
        self.params_g_a = [var for var in t_vars if 'g_a' in var.name]
        self.params_g_b = [var for var in t_vars if 'g_b' in var.name]
        self.params_d_a = [var for var in t_vars if 'd_a' in var.name]
        self.params_d_b = [var for var in t_vars if 'd_b' in var.name]
        self.params_d = self.params_d_a + self.params_d_b
        self.params_g = self.params_g_a + self.params_g_b
        self.saver = tf.train.Saver()

        if self.loss == 'wgan':
            # lr = 5e-5
            self.d_opt = \
                tf.train.RMSPropOptimizer(args.lr).\
                    minimize(self.loss_d, var_list=self.params_d)
            if len(self.params_g) > 0:
                self.g_opt = \
                    tf.train.RMSPropOptimizer(args.lr).\
                        minimize(self.loss_g, var_list=self.params_g)
            else:
                self.g_opt = tf.no_opt()
            self.clip_d = self.clip_trainable_params(self.params_d)
        else:
            self.d_opt = \
                tf.train.AdamOptimizer(args.lr, beta1=0.5, beta2=0.9).\
                    minimize(self.loss_d, var_list=self.params_d)
            if len(self.params_g) > 0:
                self.g_opt = \
                    tf.train.AdamOptimizer(args.lr, beta1=0.5, beta2=0.9).\
                        minimize(self.loss_g, var_list=self.params_g)
            else:
                self.g_opt = tf.no_opt()
        self.show_params('generator g', self.params_g)
        self.show_params('discriminator d', self.params_d)

        # clip=0.01
        self.init = tf.global_variables_initializer()

        tf.summary.scalar('d loss', self.loss_d)
        tf.summary.scalar('g loss', self.loss_g)
        tf.summary.scalar('weierstrass distance', self.wdist)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs/' + self.dir_name,
                                            self.sess.graph)

    def gen_net(self, prefix, inp, out_dim, std, reuse=True):
        pre_dim = int(inp.get_shape()[-1])
        # if prefix[0] == 'f':
        #     return inp
        # if prefix == 'f_a':
        #     return inp * np.array([[1, 1, 0.5]])
        # if prefix == 'f_b':
        #     return inp * np.array([[1, 1, 2.0]])

        hidden = self.hidden_g

        out = dense(inp, hidden, activation=tf.nn.relu,
                    name=prefix + '.1', std=std, reuse=reuse)
        out = dense(out, hidden, activation=tf.nn.relu,
                    name=prefix + '.2', std=std, reuse=reuse)
        #out = dense(out, hidden, activation=tf.nn.relu,
        #            name=prefix + '.3', std=std, reuse=reuse)
        out = dense(out, out_dim, activation=None,
                    name=prefix + '.out', std=std, reuse=reuse)
        return out

    def dis_net(self, prefix, inp, reuse=True):
        hidden = self.hidden_d

        out = tf.layers.dense(inp, hidden, activation=tf.nn.relu,
                              name=prefix + '.1', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                              name=prefix + '.2', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                              name=prefix + '.3', reuse=reuse)
        out = tf.layers.dense(out, 1, activation=None,
                              name=prefix + '.out', reuse=reuse)
        return out

    def clip_trainable_params(self, params):
        ops = []
        for p in params:
            ops.append(p.assign(tf.clip_by_value(p, -self.clip, self.clip)))
        return ops

    def show_params(self, name, params):
        print('Training Parameters for %s module' % name)
        for param in params:
            print(param.name, ': ', param.get_shape())

    # suppose have same horizon H
    def train(self, args, data_a, data_b, ck_dir=None, ita2b=None):
        # data: numpy, [N x n_x]
        self.sess.run(self.init)
        ls_ds, ls_gs, wds, ls_is, ls_bgs = [], [], [], [], []

        start_time = time.time()
        for epoch_idx in range(0, args.epoch):
            if epoch_idx % 500 == 0 or epoch_idx < 25:
                n_c = 100
            else:
                n_c = args.n_c

            # add summary
            for i in range(n_c):
                batch_a, batch_b = data_a(), data_b()
                ls_d, _ = self.sess.run([self.loss_d, self.d_opt],
                                        feed_dict={self.real_a: batch_a,
                                                   self.real_b: batch_b,
                                                   self.std: 0.0})
                if self.loss == 'wgan':
                    self.sess.run(self.clip_d)
                ls_ds.append(ls_d)

            batch_a, batch_b = data_a(), data_b()
            ls_g, ls_bg, ls_i, _, wd = \
                self.sess.run([self.loss_g, self.loss_best_g, self.loss_ident,
                              self.g_opt, self.wdist],
                              feed_dict={self.real_a: batch_a,
                                         self.real_b: batch_b,
                                         self.std: 0.0})
            ls_gs.append(ls_g)
            ls_bgs.append(ls_bg)
            ls_is.append(ls_i)
            wds.append(wd)

            if (epoch_idx + 1) % args.log_interval == 0:
                end_time = time.time()
                print('Epoch %d (%.3f s), loss D = %.6f, loss G = %.6f,'
                      'w_dist = %.9f, loss ident G = %.6f, '
                      'loss best G = %.6f' %
                      (epoch_idx, end_time - start_time, float(np.mean(ls_ds)),
                       float(np.mean(ls_gs)), float(np.mean(wds)), float(np.mean(ls_is)),
                       float(np.mean(ls_bgs))))
                ls_ds, ls_gs, wds, ls_is, ls_bgs = [], [], [], [], []

                data = util_grid(-2.0, 2.0, 200, -2.0, 2.0, 200)
                x = np.ogrid[-2.0:2.0:200j]
                y = np.ogrid[-2.0:2.0:200j]
                v = self.sess.run(self.d_real_b,
                                  feed_dict={self.real_b: data,
                                             self.std: 0.0})
                v = util_togrid(v, 200, 200)
                plt.figure(figsize=(8, 8))
                plt.contourf(x.reshape(-1), y.reshape(-1), v, 20)

                gta = data_a()
                gtb = ita2b.run(gta)
                tb = self.sess.run(self.fake_b, feed_dict={self.real_a: gta,
                                                             self.std: 0.0})
                plt.scatter(gta[:, 0], gta[:, 1], color='b', s=0.5)
                plt.scatter(tb[:, 0], tb[:, 1], color='y', s=0.5)
                plt.scatter(gtb[:, 0], gtb[:, 1], color='r', s=0.5)
                plt.savefig(self.dir_name + '/a2b%d.jpg' % epoch_idx)
                plt.close()

                start_time = time.time()
