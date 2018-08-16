import os
import time
import tensorflow as tf
import numpy as np
from utils import *
from sn import spectral_normed_weight
import warnings
from mujoco_utils import *
from layers import *


def dense(inp, out_dim, activation, name, std, reuse):
    with tf.variable_scope(name, reuse=reuse):
        weight_shape = [int(inp.get_shape()[1]), out_dim]
        w = tf.get_variable('w', weight_shape)
        w_t = w #+ tf.random_normal(shape=weight_shape,
                #                   mean=0.0, stddev=std, dtype=tf.float32)
        b = tf.get_variable('b', [out_dim],
                            initializer=tf.constant_initializer(0))
        b_t = b #+ tf.random_normal(shape=[out_dim],
                #                   mean=0.0, stddev=std, dtype=tf.float32)
        if activation is None:
            return tf.nn.bias_add(tf.matmul(inp, w_t), b_t)
        else:
            return activation(tf.nn.bias_add(tf.matmul(inp, w_t), b_t))


class CycleGAIL(object):
    def __init__(self, name, args, clip, env_a, env_b,
                 a_obs_dim, a_act_dim, b_obs_dim, b_act_dim,
                 hidden_g, hidden_d, lambda_g, gamma, metric='L1',
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
        self.a_obs_dim = a_obs_dim
        self.a_act_dim = a_act_dim
        self.b_obs_dim = b_obs_dim
        self.b_act_dim = b_act_dim
        self.a_dim = a_obs_dim + a_act_dim
        self.b_dim = b_obs_dim + b_act_dim
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
        print('Domain A: %d\nDomain B: %d\n' % (self.a_dim, self.b_dim))

        print('CycleGAIL: Start building graph ...')
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.build_model(args)
        print('CycleGAIL: Build graph finished !')

    def markov(self, current):
        return current

    def graident_penalty(self, name, real, fake):
        alpha = tf.random_uniform([tf.shape(real)[0], 1, 1], 0., 1.)
        hat = self.markov(alpha * real + ((1 - alpha) * fake))
        critic_hat_a = self.dis_net(name, hat)
        gradients = tf.gradients(critic_hat_a, [hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1, 2]))
        return tf.reduce_mean((slopes - 1) ** 2)

    def build_model(self, args):
        self.real_a = tf.placeholder(tf.float32, [None, None, self.a_dim])
        self.real_b = tf.placeholder(tf.float32, [None, None, self.b_dim])
        self.std = tf.placeholder(tf.float32, shape=())
        self.ts = tf.placeholder(tf.float32, [None, None, 1])

        self.fake_a = \
            self.gen_net('g_a', self.real_b, self.a_dim, self.std, False)
        self.fake_b = \
            self.gen_net('g_b', self.real_a, self.b_dim, self.std, False)
        self.inv_a = self.gen_net('g_a', self.fake_b, self.a_dim, self.std)
        self.inv_b = self.gen_net('g_b', self.fake_a, self.b_dim, self.std)

        self.cycle_a = \
            cycle_loss(self.real_a, self.inv_a, self.metric)
        self.cycle_b = \
            cycle_loss(self.real_b, self.inv_b, self.metric)

        self.real_at = tf.concat([self.ts, self.real_a], 2)
        self.real_bt = tf.concat([self.ts, self.real_b], 2)
        self.fake_at = tf.concat([self.ts, self.fake_a], 2)
        self.fake_bt = tf.concat([self.ts, self.fake_b], 2)
        self.d_real_a = self.dis_net('d_a', self.markov(self.real_at), False)
        self.d_real_b = self.dis_net('d_b', self.markov(self.real_bt), False)
        self.d_fake_a = self.dis_net('d_a', self.markov(self.fake_at))
        self.d_fake_b = self.dis_net('d_b', self.markov(self.fake_bt))

        self.wdist_a = tf.reduce_mean(self.d_real_a - self.d_fake_a)
        self.wdist_b = tf.reduce_mean(self.d_real_b - self.d_fake_b)
        self.wdist = self.wdist_a + self.wdist_b
        self.loss_d = - self.wdist_a - self.wdist_b
        if self.loss == 'wgan-gp':
            self.gp = self.graident_penalty('d_a', self.real_at, self.fake_at)
            self.gp += self.graident_penalty('d_b', self.real_bt, self.fake_bt)
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
        hidden = self.hidden_g

        out = tf.layers.dense(inp, hidden, activation=tf.nn.tanh,
                              name=prefix + '.1', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.tanh,
                              name=prefix + '.2', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.tanh,
                              name=prefix + '.3', reuse=reuse)
        out = tf.layers.dense(out, out_dim, activation=None,
                              name=prefix + '.out', reuse=reuse)
        return out

    def sequence_generator(self, x, prefix, out_dim, reuse):
        with tf.variable_scope(prefix, reuse=reuse):
            pad_input = tf.pad(
                x, [[0, 0],
                    [32 // 2, 32 // 2],
                    [0, 0]], "CONSTANT")
            c = general_conv1d(
                pad_input,
                num_filters=32,
                filter_size=1,
                stride=1,
                stddev=0.02,
                name="c1")
            c = general_conv1d(
                c,
                num_filters=32 * 2,
                filter_size=1,
                stride=1,
                stddev=0.02,
                padding="SAME",
                name="c2")
            l = general_conv1d(
                c,
                num_filters=32 * 4,
                filter_size=1,
                stride=1,
                stddev=0.02,
                padding="SAME",
                name="c3")

            for i in range(5):
                l = build_resnet_block(
                    general_conv1d,
                    l,
                    dim=32 * 4,
                    filter_size=1,
                    pad=True,
                    name="r%d" % i)

            c = general_conv1d(
                l,
                num_filters=out_dim,
                filter_size=1,
                stride=1,
                stddev=0.02,
                padding="SAME",
                name="out")

            return c

    def dis_net(self, prefix, inp, reuse=True):
        with tf.variable_scope(prefix, reuse=reuse):
            # whether to use dropout ?
            x = build_n_layer_conv_stack(general_conv1d, inp, 15, 32, n=5,
                                         do_norm="layer")
            return x

    def clip_trainable_params(self, params):
        ops = []
        for p in params:
            ops.append(p.assign(tf.clip_by_value(p, -self.clip, self.clip)))
        return ops

    def show_params(self, name, params):
        print('Training Parameters for %s module' % name)
        for param in params:
            print(param.name, ': ', param.get_shape())

    def get_demo(self, expert_a, expert_b, istrain=True):
        if istrain:
            obs_a, act_a, ts_a = expert_a.next_batch()
            obs_b, act_b, ts_b = expert_b.next_batch()
            a_concat = np.concatenate([obs_a, act_a], 2)
            b_concat = np.concatenate([obs_b, act_b], 2)
        else:
            obs_a, act_a, ts_a = expert_a.next_demo(train=False)
            obs_b, act_b, ts_b = expert_b.next_demo(train=False)
            a_concat = np.concatenate([obs_a, act_a], 1)
            b_concat = np.concatenate([obs_b, act_b], 1)
        return a_concat, b_concat, ts_a

    # suppose have same horizon H
    def train(self, args, expert_a, expert_b, ck_dir=None, ita2b=None):
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
                batch_a, batch_b, batch_t = self.get_demo(expert_a, expert_b)
                ls_d, _ = self.sess.run([self.loss_d, self.d_opt],
                                        feed_dict={self.real_a: batch_a,
                                                   self.real_b: batch_b,
                                                   self.std: 0.0,
                                                   self.ts: batch_t})
                if self.loss == 'wgan':
                    self.sess.run(self.clip_d)
                ls_ds.append(ls_d)

            batch_a, batch_b, batch_t = self.get_demo(expert_a, expert_b)
            ls_g, ls_bg, ls_i, _, wd = \
                self.sess.run([self.loss_g, self.loss_best_g, self.loss_ident,
                              self.g_opt, self.wdist],
                              feed_dict={self.real_a: batch_a,
                                         self.real_b: batch_b,
                                         self.std: 0.0,
                                         self.ts: batch_t})
            ls_gs.append(ls_g)
            ls_bgs.append(ls_bg)
            ls_is.append(ls_i)
            wds.append(wd)

            if (epoch_idx + 1) % args.log_interval == 0:
                self.visual_eval(expert_a, expert_b,
                                 (epoch_idx + 1) // args.log_interval)
                # calculate ideal mapping loss
                tj_a, tj_b, _ = \
                    self.get_demo(expert_a, expert_b, istrain=False)
                transed_b = \
                    self.sess.run(self.fake_b,
                                  feed_dict={self.real_a: np.expand_dims(tj_a, 0),
                                             self.std: 0.0})
                transed_b = np.squeeze(transed_b)
                ideal_b = self.calc_ideal_b(tj_a, ita2b, expert_a, expert_b)

                end_time = time.time()
                print('Epoch %d (%.3f s), loss D = %.6f, loss G = %.6f,'
                      'w_dist = %.9f, loss ident G = %.6f, '
                      'loss best G = %.6f, EM = %.6f' %
                      (epoch_idx, end_time - start_time, float(np.mean(ls_ds)),
                       float(np.mean(ls_gs)), float(np.mean(wds)),
                       float(np.mean(ls_is)), float(np.mean(ls_bgs)),
                       float(np.mean(np.square(transed_b - ideal_b)))))
                ls_ds, ls_gs, wds, ls_is, ls_bgs = [], [], [], [], []
                start_time = time.time()

    def calc_ideal_b(self, traj, trans, domaina, domainb):
        obs_a, act_a = self.unzip(traj, domaina.obs_dim)
        obs_a = domaina.obs_r(obs_a)
        act_a = domaina.act_r(act_a)
        obs_b, act_b = self.unzip(trans.run(np.concatenate([obs_a, act_a], 1)),
                                  domainb.obs_dim)
        obs_b = domainb.obs_n(obs_b)
        act_b = domainb.act_n(act_b)
        return np.concatenate([obs_b, act_b], 1)

    def unzip(self, obs_act_pair, obs_dim):
        return obs_act_pair[:, :obs_dim], obs_act_pair[:, obs_dim:]

    def visual_eval(self, expert_a, expert_b, id):

        if self.vis_mode == 'synthetic':
            tj_a, tj_b, _ = self.get_demo(expert_a, expert_b, istrain=False)
            transed_b = \
                self.sess.run(self.fake_b,
                              feed_dict={self.real_a: tj_a, self.std: 0.0})
            obs_a, act_a = self.unzip(tj_a, self.a_obs_dim)
            gt_obs_b, gt_act_b = self.unzip(tj_b, self.b_obs_dim)
            obs_b, act_b = self.unzip(transed_b, self.b_obs_dim)
            distribution_diff(obs_a, act_a, gt_obs_b, gt_act_b, obs_b, act_b,
                              self.dir_name + '/img/' + str(id) + 'a2b_D.jpg')

            obs_b = expert_b.obs_r(obs_b)
            act_b = expert_b.act_r(act_b)
            show_trajectory(self.env_b, obs_b, act_b, obs_a[0, :],
                            self.dir_name + '/img/' + str(id) + 'a2b_T.jpg')
