import os
import time
import tensorflow as tf
import numpy as np
from utils import *
from sn import spectral_normed_weight
import warnings


import tflib as lib
import tflib.ops.linear


def relu_layer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output


class CondCycleGAIL(object):
    def __init__(self, name, sess, clip, env_a, env_b,
                 a_act_dim, b_act_dim, a_obs_dim, b_obs_dim, hidden, lambda_g,
                 lambda_f, use_orac_loss, loss_metric='L1',
                 checkpoint_dir=None, spect=True, loss='wgan'):
        self.sess = sess
        self.clip = clip
        self.env_a = env_a
        self.env_b = env_b

        self.loss_metric = loss_metric
        self.checkpoint_dir = checkpoint_dir
        self.dir_name = name
        self.a_act_dim = a_act_dim
        self.b_act_dim = b_act_dim
        self.a_obs_dim = a_obs_dim
        self.b_obs_dim = b_obs_dim
        self.use_spect = loss == 'wgan-sn'
        self.hidden = hidden
        self.lambda_f = lambda_f
        self.lambda_g = lambda_g
        self.use_orac_loss = use_orac_loss
        self.loss = loss
        print('======= Settings =======')
        print('-------- Models --------')
        print('GAN: %s\nclip: %.3f\nG lambda %.3f\nF lambda %.3f\n'
              'Loss metric: %s\nUse oloss: %s\nHidden size: %d\n'
              % (loss, clip, lambda_g, lambda_f, loss_metric,
                 str(use_orac_loss), hidden))
        print('----- Environments -----')
        print('Domain A Obs: %d\nDomain A Act: %d\nDomain B Obs: %d\n'
              'Domain B Act: %d\n'
              % (a_obs_dim, a_act_dim, b_obs_dim, b_act_dim))

        print('CycleGAIL: Start building graph ...')
        self.build_model()
        print('CycleGAIL: Build graph finished !')

    def graident_penalty(self, name, real, fake):
        alpha = tf.random_uniform([tf.shape(real)[0], 1], 0., 1.)
        hat = alpha * real + ((1 - alpha) * fake)
        critic_hat_a = self.dis_net(name, hat)
        gradients = tf.gradients(critic_hat_a, [hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        return tf.reduce_mean((slopes - 1) ** 2)

    def build_model(self):
        self.real_act_a = tf.placeholder(tf.float32, [None, self.a_act_dim])
        self.real_act_b = tf.placeholder(tf.float32, [None, self.b_act_dim])
        self.real_obs_a = tf.placeholder(tf.float32, [None, self.a_obs_dim])
        self.real_obs_b = tf.placeholder(tf.float32, [None, self.b_obs_dim])

        self.fake_obs_a = self.gen_net('f_a', self.real_obs_b, self.a_obs_dim)
        self.fake_obs_b = self.gen_net('f_b', self.real_obs_a, self.b_obs_dim)
        self.inv_obs_a = self.gen_net('f_a', self.fake_obs_b, self.a_obs_dim)
        self.inv_obs_b = self.gen_net('f_b', self.fake_obs_a, self.b_obs_dim)

        fake_act_a_inp = tf.concat([self.fake_obs_a, self.real_act_b], axis=1)
        fake_act_b_inp = tf.concat([self.fake_obs_b, self.real_act_a], axis=1)
        self.fake_act_a = self.gen_net('g_a', fake_act_a_inp, self.a_act_dim)
        self.fake_act_b = self.gen_net('g_b', fake_act_b_inp, self.b_act_dim)
        inv_act_a_inp = tf.concat([self.inv_obs_a, self.fake_act_b], axis=1)
        inv_act_b_inp = tf.concat([self.inv_obs_b, self.fake_act_a], axis=1)
        self.inv_act_a = self.gen_net('g_a', inv_act_a_inp, self.a_act_dim)
        self.inv_act_b = self.gen_net('g_b', inv_act_b_inp, self.b_act_dim)

        self.cycle_act_a = \
            cycle_loss(self.real_act_a, self.inv_act_a, self.loss_metric)
        self.cycle_act_b = \
            cycle_loss(self.real_act_b, self.inv_act_b, self.loss_metric)
        self.cycle_obs_a = \
            cycle_loss(self.real_obs_a, self.inv_obs_a, self.loss_metric)
        self.cycle_obs_b = \
            cycle_loss(self.real_obs_b, self.inv_obs_b, self.loss_metric)

        self.real_a = tf.concat([self.real_obs_a, self.real_act_a], 1)
        self.fake_a = tf.concat([self.fake_obs_a, self.fake_act_a], 1)
        self.real_b = tf.concat([self.real_obs_b, self.real_act_b], 1)
        self.fake_b = tf.concat([self.fake_obs_b, self.fake_act_b], 1)
        self.d_real_a = self.dis_net('d_a', self.real_a)
        self.d_real_b = self.dis_net('d_b', self.real_b)
        self.d_fake_a = self.dis_net('d_a', self.fake_a)
        self.d_fake_b = self.dis_net('d_b', self.fake_b)

        self.wdist_a = tf.reduce_mean(self.d_real_a - self.d_fake_a)
        self.wdist_b = tf.reduce_mean(self.d_real_b - self.d_fake_b)
        self.wdist = self.wdist_a + self.wdist_b
        self.loss_d = - self.wdist_a - self.wdist_b
        if self.loss == 'wgan-gp':
            self.gp = self.graident_penalty('d_a', self.real_a, self.fake_a)
            self.gp += self.graident_penalty('d_b', self.real_b, self.fake_b)
            self.loss_d += self.gp
        self.loss_gf_a = -tf.reduce_mean(self.d_fake_a)
        self.loss_gf_b = -tf.reduce_mean(self.d_fake_b)

        self.loss_g = self.loss_gf_a + self.loss_gf_b + \
            self.lambda_g * (self.cycle_act_a + self.cycle_act_b)
        self.loss_f = self.loss_gf_a + self.loss_gf_b + \
            self.lambda_f * (self.cycle_obs_a + self.cycle_obs_b)

        self.params_g_a = lib.params_with_name('g_a')
        self.params_g_b = lib.params_with_name('g_b')
        self.params_f_a = lib.params_with_name('f_a')
        self.params_f_b = lib.params_with_name('f_b')
        self.params_d_a = lib.params_with_name('d_a')
        self.params_d_b = lib.params_with_name('d_b')
        self.params_d = self.params_d_a + self.params_d_b
        self.params_g = self.params_g_a + self.params_g_b
        self.params_f = self.params_f_a + self.params_f_b

    def gen_net(self, prefix, inp, out_dim):
        pre_dim = int(inp.get_shape()[-1])
        out = relu_layer(prefix + '.1', pre_dim, self.hidden, inp)
        out = relu_layer(prefix + '.2', self.hidden, self.hidden, out)
        out = relu_layer(prefix + '.3', self.hidden, self.hidden, out)
        out = lib.ops.linear.Linear(prefix + '.4', self.hidden, out_dim, out)
        return out

    def dis_net(self, prefix, inp):
        pre_dim = int(inp.get_shape()[-1])
        out = relu_layer(prefix + '.1', pre_dim, 128, inp)
        out = relu_layer(prefix + '.2', 128, 128, out)
        out = relu_layer(prefix + '.3', 128, 128, out)
        out = lib.ops.linear.Linear(prefix + '.4', 128, 1, out)
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
        obs_a, act_a = expert_a.next_demo(is_train)
        obs_b, act_b = expert_b.next_demo(is_train)
        demos = {self.real_obs_a: obs_a,
                 self.real_act_a: act_a,
                 self.real_obs_b: obs_b,
                 self.real_act_b: act_b}
        return demos

    def sample(self, env, obs, cacts, obs_ph, act_ph, gen_obs, gen_act):
        img_obs = self.sess.run(gen_obs, feed_dict={obs_ph: obs})
        obs_dim = gen_obs.get_shape()[-1]
        act_dim = gen_act.get_shape()[-1]
        horizon = img_obs.shape[0]
        acts = np.zeros((horizon, act_dim))
        run_obs = np.zeros((horizon, obs_dim))
        current_obs = env.reset(img_obs[0, :])
        for i in range(horizon):
            run_obs[i, :] = current_obs
            fd = {obs_ph: np.reshape(current_obs, (1, obs_dim)),
                  act_ph: np.reshape(cacts[i, :], (1, -1))}
            acts[i, :] = np.reshape(self.sess.run(gen_act, feed_dict=fd), (act_dim,))
            current_obs, _, __, ___ = env.step(acts[i, :])
        return img_obs, run_obs, acts

    def run_trajectory(self, demos, dir='a2b'):
        if dir == 'a2b':
            return self.sample(self.env_b, demos[self.real_obs_a], 
                demos[self.real_act_a], self.real_obs_a, self.real_act_a, 
                self.fake_obs_b, self.fake_act_b)
        else:
            return self.sample(self.env_a, demos[self.real_obs_b], 
                demos[self.real_act_b], self.real_obs_b, self.real_act_b, 
                self.fake_obs_a, self.fake_act_a)

    # suppose have same horizon H
    def train(self, args, expert_a, expert_b):
        # data: numpy, [N x n_x]

        print(self.loss)

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
        self.show_params('generator g', self.params_g)
        self.show_params('generator f', self.params_f)
        self.show_params('discriminator d', self.params_d)

        # clip=0.01
        tf.global_variables_initializer().run()

        tf.summary.scalar('d loss', self.loss_d)
        tf.summary.scalar('g loss', self.loss_g)
        tf.summary.scalar('f loss', self.loss_f)
        tf.summary.scalar('weierstrass distance', self.wdist)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs/' + self.dir_name,
                                            self.sess.graph)
        ls_ds = []
        ls_gs = []
        ls_fs = []
        wds = []
        start_time = time.time()
        self.visual_evaluation(expert_a, expert_b, 0)
        for epoch_idx in range(0, args.epoch):
            if epoch_idx % 500 == 0 or epoch_idx < 25:
                n_c = 100
            else:
                n_c = args.n_c

            # add summary
            demos = self.get_demo(expert_a, expert_b)
            summary = self.sess.run(merged, demos)
            self.writer.add_summary(summary, epoch_idx)

            for i in range(n_c):
                demos = self.get_demo(expert_a, expert_b)
                ls_d, _ = self.sess.run([self.loss_d, self.d_opt],
                                        feed_dict=demos)
                if self.loss == 'wgan':
                    self.sess.run(self.clip_d)
                ls_ds.append(ls_d)

            demos = self.get_demo(expert_a, expert_b)
            ls_g, _, wd = self.sess.run([self.loss_g, self.g_opt, self.wdist],
                                        feed_dict=demos)
            ls_gs.append(ls_g)
            wds.append(wd)
            demos = self.get_demo(expert_a, expert_b)
            ls_f, _ = self.sess.run([self.loss_f, self.f_opt],
                                    feed_dict=demos)
            ls_fs.append(ls_f)

            if (epoch_idx + 1) % 100 == 0:
                end_time = time.time()
                self.visual_evaluation(expert_a, expert_b,
                                       (epoch_idx + 1) // 100)
                print('Epoch %d (%.3f s), loss D = %.6f, loss G = %.6f,'
                      'loss F = %6f, w_dist = %.9f' %
                      (epoch_idx, end_time - start_time, float(np.mean(ls_ds)),
                       float(np.mean(ls_gs)), float(np.mean(ls_fs)),
                       float(np.mean(wds))))
                ls_ds = []
                ls_gs = []
                ls_fs = []
                wds = []
                start_time = time.time()

    def visual_evaluation(self, expert_a, expert_b, id):
        # a2b
        demos = self.get_demo(expert_a, expert_b, is_train=False)
        img_obs_b, run_obs_b, act_b = self.run_trajectory(demos)
        trajectory_diff(img_obs_b, run_obs_b,
                        self.dir_name + '/img/' + str(id) + 'a2b_T.jpg')
        distribution_diff(demos[self.real_obs_a], demos[self.real_act_b],
                          demos[self.real_obs_b], demos[self.real_act_b],
                          img_obs_b, act_b,
                          self.dir_name + '/img/' + str(id) + 'a2b_D.jpg')