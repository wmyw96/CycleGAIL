import os
import time
import tensorflow as tf
import numpy as np
from utils import *
from sn import spectral_normed_weight
import warnings
from mujoco_utils import *


class CycleGAIL(object):
    def __init__(self, name, args, clip, env_a, env_b,
                 a_act_dim, b_act_dim, a_obs_dim, b_obs_dim,
                 hidden_f, hidden_g, hidden_d,
                 w_obs_a, w_obs_b, w_act_a, w_act_b,
                 lambda_g, lambda_f, gamma, use_orac_loss, metric='L1',
                 checkpoint_dir=None, spect=True, loss='wgan',
                 vis_mode='synthetic', concat_steps=0, align=None):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        self.clip = clip
        self.env_a = env_a
        self.env_b = env_b
        self.vis_mode = vis_mode
        self.metric = metric
        self.checkpoint_dir = checkpoint_dir
        self.dir_name = name
        self.a_act_dim = a_act_dim
        self.b_act_dim = b_act_dim
        self.a_obs_dim = a_obs_dim
        self.b_obs_dim = b_obs_dim
        self.use_spect = loss == 'wgan-sn'
        self.hidden_f = hidden_f
        self.hidden_g = hidden_g
        self.hidden_d = hidden_d
        self.lambda_f = lambda_f
        self.lambda_g = lambda_g
        self.use_orac_loss = use_orac_loss
        self.loss = loss
        self.gamma = gamma
        self.concat_steps = concat_steps
        self.expert_a = None
        self.expert_b = None
        self.align = align
        print('FFFFFFFFFFFFFFFFFF')
        print('======= Settings =======')
        print('-------- Models --------')
        print('GAN: %s\nclip: %.3f\nG lambda %.3f\nF lambda %.3f\n'
              'Loss metric: %s\nUse oloss: %s\nGamma: %.3f\nF Hidden size: '
              '%d\nG Hidden size: %d\nD Hidden size: %d\nConcat steps %d'
              % (loss, clip, lambda_g, lambda_f, metric,
                 str(use_orac_loss), gamma, hidden_f, hidden_g, hidden_d,
                 concat_steps))
        print('----- Environments -----')
        print('Domain A Obs: %d\nDomain A Act: %d\nDomain B Obs: %d\n'
              'Domain B Act: %d\n'
              % (a_obs_dim, a_act_dim, b_obs_dim, b_act_dim))

        print('CycleGAIL: Start building graph ...')
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.real_act_a = tf.placeholder(tf.float32, [None, self.a_act_dim])
            self.real_act_b = tf.placeholder(tf.float32, [None, self.b_act_dim])
            self.real_obs_a = tf.placeholder(tf.float32, [None, self.a_obs_dim])
            self.real_obs_b = tf.placeholder(tf.float32, [None, self.b_obs_dim])
            self.orac_obs_a = tf.placeholder(tf.float32, [None, self.a_obs_dim])
            self.orac_obs_b = tf.placeholder(tf.float32, [None, self.b_obs_dim])
            self.real_a = tf.concat([self.real_obs_a, self.real_act_a], 1)
            self.real_b = tf.concat([self.real_obs_b, self.real_act_b], 1)
            self.build_model(w_obs_a, w_obs_b, w_act_a, w_act_b, args)
            self.build_dynamic_envs(args)
            self.build_train_settings(args)
            self.init = tf.global_variables_initializer()
        print('CycleGAIL: Build graph finished !')

    def markov(self, current, steps=None):
        stacks = []
        if steps is None:
            steps = self.concat_steps
        for i in range(steps):
            stacks.append(current[i: -steps + i, :])
        stacks.append(current[steps:, :])
        return tf.concat(stacks, axis=1)

    def graident_penalty(self, name, real, fake):
        alpha = tf.random_uniform([tf.shape(real)[0], 1], 0., 1.)
        hat = self.markov(alpha * real + ((1 - alpha) * fake))
        critic_hat_a = self.dis_net(name, hat)
        gradients = tf.gradients(critic_hat_a, [hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        return tf.reduce_mean((slopes - 1) ** 2)

    def build_model(self, w_obs_a, w_obs_b, w_act_a, w_act_b, args):
        self.ts = tf.placeholder(tf.float32, [None, 1])

        self.fake_act_a = \
            self.gen_net('g_a', self.real_act_b, self.a_act_dim, False)
        self.fake_act_b = \
            self.gen_net('g_b', self.real_act_a, self.b_act_dim, False)
        self.inv_act_a = self.gen_net('g_a', self.fake_act_b, self.a_act_dim)
        self.inv_act_b = self.gen_net('g_b', self.fake_act_a, self.b_act_dim)

        self.fake_obs_a = \
            self.gen_net('f_a', self.real_obs_b, self.a_obs_dim, False)
        self.fake_obs_b = \
            self.gen_net('f_b', self.real_obs_a, self.b_obs_dim, False)
        self.inv_obs_a = self.gen_net('f_a', self.fake_obs_b, self.a_obs_dim)
        self.inv_obs_b = self.gen_net('f_b', self.fake_obs_a, self.b_obs_dim)

        self.cycle_act_a = \
            cycle_loss(self.real_act_a, self.inv_act_a, self.metric)
        self.cycle_act_b = \
            cycle_loss(self.real_act_b, self.inv_act_b, self.metric)
        self.cycle_obs_a = \
            cycle_loss(self.real_obs_a, self.inv_obs_a, self.metric)
        self.cycle_obs_b = \
            cycle_loss(self.real_obs_b, self.inv_obs_b, self.metric)

        self.fake_a = tf.concat([self.fake_obs_a, self.fake_act_a], 1)
        self.fake_b = tf.concat([self.fake_obs_b, self.fake_act_b], 1)
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
            self.lambda_g * (self.cycle_act_a + self.cycle_act_b)
        self.loss_f = self.loss_gf_a + self.loss_gf_b + \
            self.lambda_f * (self.cycle_obs_a + self.cycle_obs_b)

        self.loss_gf = self.loss_gf_a + self.loss_gf_b + \
            self.lambda_g * (self.cycle_act_a + self.cycle_act_b) + \
            self.lambda_f * (self.cycle_obs_a + self.cycle_obs_b)

        self.loss_ident_f_a = \
            cycle_loss(self.fake_obs_b, self.real_obs_a, self.metric, w_obs_b)

        self.loss_ident_f = \
            cycle_loss(self.fake_obs_a, self.real_obs_b, self.metric, w_obs_a) + \
            cycle_loss(self.fake_obs_b, self.real_obs_a, self.metric, w_obs_b)
        self.loss_ident_g = \
            cycle_loss(self.fake_act_a, self.real_act_b, self.metric, w_act_a) + \
            cycle_loss(self.fake_act_b, self.real_act_a, self.metric, w_act_b)
        self.loss_best_g = \
            cycle_loss(self.real_act_a, self.real_act_b, self.metric, w_act_a)
        self.loss_best_f = \
            cycle_loss(self.real_obs_a, self.real_obs_b, self.metric, w_obs_a)

        self.loss_align = None
        if self.align is not None:
            obs_a, obs_b = self.align
            self.trans_b = self.gen_net('f_b', obs_a, self.b_obs_dim)
            self.trans_a = self.gen_net('f_a', obs_b, self.a_obs_dim)
            self.loss_align = \
                cycle_loss(self.trans_b, obs_b, self.metric) + \
                cycle_loss(self.trans_a, obs_a, self.metric)
            #self.loss_gf += self.loss_align
            #self.loss_f += self.loss_align

    def build_train_settings(self, args):
        t_vars = tf.trainable_variables()
        self.params_g_a = [var for var in t_vars if 'g_a' in var.name]
        self.params_g_b = [var for var in t_vars if 'g_b' in var.name]
        self.params_f_a = [var for var in t_vars if 'f_a' in var.name]
        self.params_f_b = [var for var in t_vars if 'f_b' in var.name]
        self.params_d_a = [var for var in t_vars if 'd_a' in var.name]
        self.params_d_b = [var for var in t_vars if 'd_b' in var.name]
        self.params_d = self.params_d_a + self.params_d_b
        self.params_g = self.params_g_a + self.params_g_b
        self.params_f = self.params_f_a + self.params_f_b
        self.params_gf = self.params_f + self.params_g
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
            if len(self.params_f) > 0:
                self.f_opt = \
                    tf.train.RMSPropOptimizer(args.lr).\
                        minimize(self.loss_f, var_list=self.params_f)
            else:
                self.f_opt = tf.no_op()
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
            if len(self.params_f) > 0:
                self.f_opt = \
                    tf.train.AdamOptimizer(args.lr, beta1=0.5, beta2=0.9).\
                        minimize(self.loss_f, var_list=self.params_f)
            else:
                self.f_opt = tf.no_op()
            if len(self.params_gf) > 0:
                self.gf_opt = \
                    tf.train.AdamOptimizer(args.lr, beta1=0.5, beta2=0.9).\
                    minimize(self.loss_gf, var_list=self.params_gf)
            else:
                self.gf_opt = tf.no_op()
        self.show_params('generator g', self.params_g)
        self.show_params('generator f', self.params_f)
        self.show_params('discriminator d', self.params_d)

        # clip=0.01

        tf.summary.scalar('d loss', self.loss_d)
        tf.summary.scalar('g loss', self.loss_g)
        tf.summary.scalar('f loss', self.loss_f)
        tf.summary.scalar('weierstrass distance', self.wdist)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs/' + self.dir_name,
                                            self.sess.graph)

    def build_dynamic_envs(self, args):
        self.next_a = self.env_net('e_a', self.real_a, self.a_obs_dim, False)
        self.next_b = self.env_net('e_b', self.real_b, self.b_obs_dim, False)
        self.true_next_a = tf.placeholder(tf.float32, [None, self.a_obs_dim])
        self.true_next_b = tf.placeholder(tf.float32, [None, self.b_obs_dim])
        self.loss_e = \
            tf.reduce_mean(tf.square(self.next_a - self.true_next_a)) + \
            tf.reduce_mean(tf.square(self.next_b - self.true_next_b))

        self.next_fa = self.env_net('e_a', self.fake_a, self.a_obs_dim)
        self.next_fb = self.env_net('e_b', self.fake_b, self.a_obs_dim)
        self.loss_gf_o = \
            tf.reduce_mean(tf.square(self.next_fa[:-1, :] -
                                     self.fake_obs_a[1:, :])) + \
            tf.reduce_mean(tf.square(self.next_fb[:-1, :] -
                                     self.fake_obs_b[1:, :]))
        t_vars = tf.trainable_variables()
        self.params_e_a = [var for var in t_vars if 'e_a' in var.name]
        self.params_e_b = [var for var in t_vars if 'e_b' in var.name]
        self.params_e = self.params_e_a + self.params_e_b
        self.e_opt = \
            tf.train.AdamOptimizer(args.lr). \
                minimize(self.loss_e, var_list=self.params_e)
        self.loss_f += self.loss_gf_o
        self.loss_g += self.loss_gf_o

    def env_net(self, prefix, inp, out_dim, reuse=True):
        hidden = 512

        out = tf.layers.dense(inp, hidden, activation=tf.nn.relu,
                              name=prefix + '.1', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                              name=prefix + '.2', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                              name=prefix + '.3', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                              name=prefix + '.4', reuse=reuse)
        out = tf.layers.dense(out, out_dim, activation=None,
                              name=prefix + '.out', reuse=reuse)
        return out

    def gen_net(self, prefix, inp, out_dim, reuse=True):
        # if prefix[0] == 'f':
        #     return inp
        # if prefix == 'f_a':
        #     return inp * np.array([[1, 1, 0.5]])
        # if prefix == 'f_b':
        #     return inp * np.array([[1, 1, 2.0]])
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)
        if prefix[0] == 'f':
            hidden = self.hidden_f
        else:
            hidden = self.hidden_g

        out = tf.layers.dense(inp, hidden, activation=tf.nn.tanh,
                              name=prefix + '.1', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.tanh,
                              name=prefix + '.2', reuse=reuse)
        #out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
        #                      name=prefix + '.3', reuse=reuse)
        out = tf.layers.dense(out, out_dim, activation=None,
                              name=prefix + '.out', reuse=reuse)
        return out

    def dis_net(self, prefix, inp, reuse=True):
        hidden = self.hidden_d

        out = tf.layers.dense(inp, hidden, activation=tf.nn.relu,
                              name=prefix + '.1', reuse=reuse)
        out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                              name=prefix + '.2', reuse=reuse)
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

    def get_demo(self, expert_a, expert_b, is_train=True):
        if is_train:
            obs_a, act_a, _ = expert_a.next_batch()
            obs_b, act_b, _ = expert_b.next_batch()
        else:
            obs_a, act_a, _ = expert_a.next_demo(False)
            obs_b, act_b, _ = expert_b.next_demo(False)
        demos = {self.real_obs_a: obs_a,
                 self.real_act_a: act_a,
                 self.real_obs_b: obs_b,
                 self.real_act_b: act_b,
                 self.ts: _}
        return demos

    def train_e_net(self, nepoches, expert_a, expert_b, during_train=False):
        les = []
        if during_train:
            mod_number = 2
        else:
            mod_number = 10
        npass = 0
        cur_time = -time.time()
        for epoch_idx in range(nepoches):
            demos = self.get_demo(expert_a, expert_b)
            if epoch_idx % mod_number:
                feed_d = {self.real_act_a: demos[self.real_act_a][:-1, :],
                          self.real_obs_a: demos[self.real_obs_a][:-1, :],
                          self.real_act_b: demos[self.real_act_b][:-1, :],
                          self.real_obs_b: demos[self.real_obs_b][:-1, :],
                          self.true_next_a: demos[self.real_obs_a][1:, :],
                          self.true_next_b: demos[self.real_obs_b][1:, :]}
            else:
                next_obs_a = \
                    self.env_a.predict(expert_a.obs_r(demos[self.real_obs_a][:-1, :]),
                                       expert_a.act_r(demos[self.real_act_a][:-1, :]))
                next_obs_b = \
                    self.env_b.predict(expert_b.obs_r(demos[self.real_obs_b][:-1, :]),
                                       expert_b.act_r(demos[self.real_act_b][:-1, :]))
                if next_obs_a is None or next_obs_b is None:
                    npass += 1
                    continue
                feed_d = {
                    self.real_act_a: demos[self.real_act_a][:-1, :],
                    self.real_obs_a: demos[self.real_obs_a][:-1, :],
                    self.real_act_b: demos[self.real_act_b][:-1, :],
                    self.real_obs_b: demos[self.real_obs_b][:-1, :],
                    self.true_next_a: expert_a.obs_n(next_obs_a),
                    self.true_next_b: expert_b.obs_n(next_obs_b)
                }
            le, _ = \
                self.sess.run([self.loss_e, self.e_opt], feed_dict=feed_d)
            les.append(le)
            if (epoch_idx + 1) % 100 == 0:
                demos = self.get_demo(expert_a, expert_b, is_train=False)
                feed_d = {
                    self.real_act_a: demos[self.real_act_a][:-1, :],
                    self.real_obs_a: demos[self.real_obs_a][:-1, :],
                    self.real_act_b: demos[self.real_act_b][:-1, :],
                    self.real_obs_b: demos[self.real_obs_b][:-1, :],
                    self.true_next_a: demos[self.real_obs_a][1:, :],
                    self.true_next_b: demos[self.real_obs_b][1:, :]
                }
                ls_test = self.sess.run(self.loss_e, feed_dict=feed_d)

                next_obs_a = \
                    self.env_a.predict(
                        expert_a.obs_r(demos[self.real_obs_a][:-1, :]),
                        expert_a.act_r(demos[self.real_act_a][:-1, :]))
                next_obs_b = \
                    self.env_b.predict(
                        expert_b.obs_r(demos[self.real_obs_b][:-1, :]),
                        expert_b.act_r(demos[self.real_act_b][:-1, :]))
                if next_obs_a is None or next_obs_b is None:
                    print('error!')
                    continue
                feed_d2 = {
                    self.real_act_a: demos[self.real_act_a][:-1, :],
                    self.real_obs_a: demos[self.real_obs_a][:-1, :],
                    self.real_act_b: demos[self.real_act_b][:-1, :],
                    self.real_obs_b: demos[self.real_obs_b][:-1, :],
                    self.true_next_a: expert_a.obs_n(next_obs_a),
                    self.true_next_b: expert_b.obs_n(next_obs_b)
                }
                ls_test2 = self.sess.run(self.loss_e, feed_dict=feed_d2)
                cur_time += time.time()
                print('[E net] Epoch %d (%.2f s), Passes = %d, '
                      'Train Loss E = %.6f, Test Loss E = %.6f, %.6f' %
                      (epoch_idx, cur_time, npass, float(np.mean(les)),
                       float(ls_test), float(ls_test2)))
                les = []
                cur_time = -time.time()
                npass = 0

    # suppose have same horizon H
    def train(self, args, expert_a, expert_b, eval_on=True,
              ck_dir=None, ita2b_obs=None, ita2b_act=None):
        # data: numpy, [N x n_x]
        print(self.loss)
        self.sess.run(self.init)

        self.train_e_net(10000, expert_a, expert_b)

        ls_ds, ls_gs, ls_fs, wds, ls_ifs, ls_igs, ls_bfs, ls_bgs, ls_gfos = \
            [], [], [], [], [], [], [], [], []

        t_obs_b_p, t_act_b_p, t_obs_a_p, t_act_a_p = None, None, None, None
        start_time = time.time()
        for epoch_idx in range(0, args.epoch):
            if epoch_idx % 500 == 0 or epoch_idx < 25:
                n_c = 100
            else:
                n_c = args.n_c

            # add summary
            demos = self.get_demo(expert_a, expert_b)
            summary = self.sess.run(self.merged, demos)
            self.writer.add_summary(summary, epoch_idx)

            if epoch_idx % 100 == 0:
                self.train_e_net(100, expert_a, expert_b, during_train=True)
            for i in range(n_c):
                demos = self.get_demo(expert_a, expert_b)
                ls_d, _ = self.sess.run([self.loss_d, self.d_opt],
                                        feed_dict=demos)
                if self.loss == 'wgan':
                    self.sess.run(self.clip_d)
                ls_ds.append(ls_d)

            '''demos = self.get_demo(expert_a, expert_b)
            ls_g, ls_f, ls_if, ls_ig, ls_bf, ls_bg, _, wd = \
                self.sess.run([self.loss_g, self.loss_f, self.loss_ident_f,
                               self.loss_ident_g, self.loss_best_f,
                               self.loss_best_g,
                               self.gf_opt, self.wdist],
                              feed_dict=demos)
            ls_gs.append(ls_g)
            ls_fs.append(ls_f)
            wds.append(wd)
            ls_ifs.append(ls_if)
            ls_igs.append(ls_ig)
            ls_bfs.append(ls_bf)
            ls_bgs.append(ls_bg)'''
            demos = self.get_demo(expert_a, expert_b)
            ls_g, _, wd = self.sess.run([self.loss_g, self.g_opt, self.wdist],
                                        feed_dict=demos)
            ls_gs.append(ls_g)
            wds.append(wd)
            demos = self.get_demo(expert_a, expert_b)
            ls_f, _ = self.sess.run([self.loss_f, self.f_opt],
                                    feed_dict=demos)
            ls_fs.append(ls_f)

            ls_if, ls_ig, ls_bf, ls_bg, ls_gfo = \
                self.sess.run([self.loss_ident_f,
                               self.loss_ident_g, self.loss_best_f,
                               self.loss_best_g, self.loss_gf_0],
                              feed_dict=demos)
            ls_ifs.append(ls_if)
            ls_igs.append(ls_ig)
            ls_bfs.append(ls_bf)
            ls_bgs.append(ls_bg)

            if (epoch_idx + 1) % args.log_interval == 0:
                end_time = time.time()
                if eval_on:
                    self.visual_evaluation(expert_a, expert_b,
                                           (epoch_idx + 1)//args.log_interval)
                if ck_dir is not None:
                    self.store(ck_dir, epoch_idx + 1)
                print('Epoch %d (%.3f s), loss D = %.6f, loss G = %.6f,'
                      'loss F = %6f, w_dist = %.9f, loss ident G = %.6f, '
                      'loss ident F = %.6f, loss best G = %.6f, '
                      'loss best F = %.6f, loss GF oracle = %.6f' %
                      (epoch_idx, end_time - start_time, float(np.mean(ls_ds)),
                       float(np.mean(ls_gs)), float(np.mean(ls_fs)),
                       float(np.mean(wds)), float(np.mean(ls_igs)),
                       float(np.mean(ls_ifs)), float(np.mean(ls_bgs)),
                       float(np.mean(ls_bfs)), float(np.mean(ls_gfo))))
                ls_ds, ls_gs, ls_fs, wds, ls_ifs, ls_igs, ls_bfs, \
                    ls_bgs, ls_gfo = [], [], [], [], [], [], [], [], []

                # comparation
                demos = self.get_demo(expert_a, expert_b, is_train=False)
                t_obs_a, t_act_a, t_obs_b, t_act_b = \
                    self.sess.run([self.fake_obs_a, self.fake_act_a,
                                   self.fake_obs_b, self.fake_act_b],
                                  feed_dict=demos)
                if t_obs_a_p is not None:
                    print('Rd[GF(0), GF(-100)] = %.6f, %.6f, %.6f, %.6f' %
                          (float(np.mean(np.square(t_obs_a - t_obs_a_p))),
                           float(np.mean(np.square(t_obs_b - t_obs_b_p))),
                           float(np.mean(np.square(t_act_a - t_act_a_p))),
                           float(np.mean(np.square(t_act_b - t_act_b_p)))))
                t_obs_a_p, t_obs_b_p, t_act_a_p, t_act_b_p = \
                    t_obs_a, t_obs_b, t_act_a, t_act_b

                # ideal mapping
                if (ita2b_act is not None) and (ita2b_obs is not None):
                    t_obs_b, t_act_b, fa = \
                        self.sess.run([self.fake_obs_b, self.fake_act_b,
                                       self.loss_ident_f_a],
                                      feed_dict=demos)
                    obs_a = demos[self.real_obs_a]
                    act_a = demos[self.real_act_a]
                    g_obs_b = ita2b_obs.run(expert_a.obs_r(obs_a))
                    g_act_b = ita2b_act.run(expert_a.act_r(act_a))
                    g_obs_b = expert_b.obs_n(g_obs_b)
                    g_act_b = expert_b.act_n(g_act_b)
                    error_obs = np.mean(np.square(t_obs_b - g_obs_b))
                    error_act = np.mean(np.square(t_act_b - g_act_b))
                    print('MSE error = %.6f (obs) %.6f, %.6f (act)' %
                          (float(error_obs), float(fa), float(error_act)))
                    if self.loss_align is not None:
                        print('AlignError = %.6f' %
                              float(self.sess.run(self.loss_align)))

                start_time = time.time()

    def visual_evaluation(self, expert_a, expert_b, id):
        # a2b
        demos = self.get_demo(expert_a, expert_b, is_train=False)
        obs_b, act_b = self.sess.run([self.fake_obs_b, self.fake_act_b],
                                     feed_dict=demos)
        if self.vis_mode == 'synthetic':
            distribution_diff(demos[self.real_obs_a], demos[self.real_act_b],
                              demos[self.real_obs_b], demos[self.real_act_b],
                              obs_b, act_b,
                              self.dir_name + '/img/' + str(id) + 'a2b_D.jpg')
            obs_b = expert_b.obs_r(obs_b)
            act_b = expert_b.act_r(act_b)
            show_trajectory(self.env_b, obs_b, act_b,
                            demos[self.real_obs_a][0, :],
                            self.dir_name + '/img/' + str(id) + 'a2b_T.jpg')
        else:
            prefix = self.dir_name + '/t' + str(id)
            if os.path.isdir(prefix):
                pass
            else:
                os.mkdir(prefix)
            tobs_b = save_trajectory_images(self.env_b,
                                            expert_b.obs_r(obs_b),
                                            expert_b.act_r(act_b), prefix)
            horizon = obs_b.shape[0]

            with open(prefix + 'gtatraj_obs.txt', 'w') as f:
                for i in range(horizon):
                    for j in range(demos[self.real_obs_a].shape[1]):
                        f.write('%.5f ' % demos[self.real_obs_a][i, j])
                    f.write('\n')
            with open(prefix + 'gtatraj_act.txt', 'w') as f:
                for i in range(horizon):
                    for j in range(demos[self.real_act_a].shape[1]):
                        f.write('%.5f ' % demos[self.real_act_a][i, j])
                    f.write('\n')
            with open(prefix + 'a2btraj_obs.txt', 'w') as f:
                for i in range(horizon):
                    for j in range(obs_b.shape[1]):
                        f.write('%.5f ' % obs_b[i, j])
                    f.write('\n')
            with open(prefix + 'a2btraj_act.txt', 'w') as f:
                for i in range(horizon):
                    for j in range(act_b.shape[1]):
                        f.write('%.5f ' % act_b[i, j])
                    f.write('\n')

            horizon = obs_b.shape[0]
            err_act = np.zeros(horizon)
            err_obs = np.zeros(horizon)
            err_obs_t = np.zeros(horizon)
            for i in range(horizon):
                err_act[i] = \
                    np.sum(np.abs(act_b[i, :] - demos[self.real_act_a][i, :]))
                err_obs[i] = \
                    np.sum(np.abs(obs_b[i, :] - demos[self.real_obs_a][i, :]))
                err_obs_t[i] = \
                    np.sum(np.abs(tobs_b[i, :] - obs_b[i, :]))
            for i in range(horizon):
                print('%.5f %.5f %.5f' %
                      (err_act[i], err_obs[i], err_obs_t[i]))
            print('%.5f %.5f %.5f' % (float(np.mean(err_act)),
                                      float(np.mean(err_obs)),
                                      float(np.mean(err_obs_t))))
            save_video(prefix + '/real', obs_b.shape[0])
            save_video(prefix + '/imag', obs_b.shape[0])
            #distribution_pdiff(demos[self.real_obs_a], demos[self.real_act_a],
            #                   demos[self.real_obs_b], demos[self.real_act_b],
            #                   obs_b, act_b, prefix + '/dist')

    def evaluation(self, expert_a, expert_b, checkpoint_dir):
        if self.load(checkpoint_dir):
            demos = self.get_demo(expert_a, expert_b, is_train=False)
            horizon = demos[self.real_obs_a].shape[0]

            if False:
                path_gta = self.dir_name + '/ground_truth_a'
                generate_dir(path_gta)
                save_trajectory_images(self.env_a,
                                       expert_a.obs_r(demos[self.real_obs_a]),
                                       expert_a.act_r(demos[self.real_act_a]),
                                       path_gta)
                save_video(self.dir_name + '/ground_truth_a/real', horizon)
                save_video(self.dir_name + '/ground_truth_a/imag', horizon)

                path_gtb = self.dir_name + '/ground_truth_b'
                generate_dir(path_gtb)
                save_trajectory_images(self.env_b,
                                       expert_b.obs_r(demos[self.real_obs_b]),
                                       expert_b.act_r(demos[self.real_act_b]),
                                       path_gtb)
                save_video(self.dir_name + '/ground_truth_b/real', horizon)
                save_video(self.dir_name + '/ground_truth_b/imag', horizon)

            self.visual_evaluation(expert_a, expert_b, 111)
            wds = []
            for i in range(50):
                demos = self.get_demo(expert_a, expert_b, is_train=False)
                wds.append(self.sess.run(self.wdist, feed_dict=demos))
            print('Test w_dist = %.5f\n' % float(np.mean(wds)))
        else:
            raise ValueError("Cannot load models")

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,
                               os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def store(self, checkpoint_dir, step):
        model_name = "CycleGAIL"
        model_dir = self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def link_demo(self, expert_a, expert_b):
        self.expert_a = expert_a
        self.expert_b = expert_b

    def run_trans(self, direction, obs=None, act=None, need_normalize=True):
        obs_t = None
        act_t = None
        if self.expert_a is None or self.expert_b is None:
            raise ValueError('Please link expert demonstrations first '
                             'using \'link_demo()\' method')
        if direction == 'a2b':
            if obs is not None:
                obs = np.array(obs).reshape(1, -1)
                if need_normalize:
                    obs = self.expert_a.obs_n(obs)
                obs_t = self.sess.run(self.fake_obs_b,
                                      feed_dict={self.real_obs_a: obs})
                if need_normalize:
                    obs_t = self.expert_b.obs_r(obs_t)
            if act is not None:
                act = np.array(act).reshape(1, -1)
                if need_normalize:
                    act = self.expert_a.act_n(act)
                act_t = self.sess.run(self.fake_act_b,
                                      feed_dict={self.real_act_a: act})
                if need_normalize:
                    act_t = self.expert_b.act_r(act_t)
        else:
            if obs is not None:
                obs = np.array(obs).reshape(1, -1)
                if need_normalize:
                    obs = self.expert_b.obs_n(obs)
                obs_t = self.sess.run(self.fake_obs_a,
                                      feed_dict={self.real_obs_b: obs})
                if need_normalize:
                    obs_t = self.expert_a.obs_r(obs_t)
            if act is not None:
                act = np.array(act).reshape(1, -1)
                if need_normalize:
                    act = self.expert_b.act_n(act)
                act_t = self.sess.run(self.fake_act_a,
                                      feed_dict={self.real_act_b: act})
                if need_normalize:
                    act_t = self.expert_a.act_r(act_t)
        return obs_t, act_t