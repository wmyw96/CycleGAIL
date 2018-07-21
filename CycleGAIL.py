import os
import time
import tensorflow as tf
import numpy as np
from utils import *
from sn import spectral_normed_weight
import warnings
from mujoco_utils import *


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


class CycleGAIL(object):
    def __init__(self, name, sess, clip, env_a, env_b,
                 a_act_dim, b_act_dim, a_obs_dim, b_obs_dim,
                 hidden_f, hidden_g, hidden_d,
                 lambda_g, lambda_f, gamma, use_orac_loss, loss_metric='L1',
                 checkpoint_dir=None, spect=True, loss='wgan',
                 vis_mode='synthetic'):
        self.sess = sess
        self.clip = clip
        self.env_a = env_a
        self.env_b = env_b
        self.vis_mode = vis_mode

        self.loss_metric = loss_metric
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
        print('======= Settings =======')
        print('-------- Models --------')
        print('GAN: %s\nclip: %.3f\nG lambda %.3f\nF lambda %.3f\n'
              'Loss metric: %s\nUse oloss: %s\nGamma: %.3f\nF Hidden size: '
              '%d\nG Hidden size: %d\nD Hidden size: %d'
              % (loss, clip, lambda_g, lambda_f, loss_metric,
                 str(use_orac_loss), gamma, hidden_f, hidden_g, hidden_d))
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

        self.fake_act_a = self.gen_net('g_a', self.real_act_b, self.a_act_dim)
        self.fake_act_b = self.gen_net('g_b', self.real_act_a, self.b_act_dim)
        self.inv_act_a = self.gen_net('g_a', self.fake_act_b, self.a_act_dim)
        self.inv_act_b = self.gen_net('g_b', self.fake_act_a, self.b_act_dim)

        self.fake_obs_a = self.gen_net('f_a', self.real_obs_b, self.a_obs_dim)
        self.fake_obs_b = self.gen_net('f_b', self.real_obs_a, self.b_obs_dim)
        self.inv_obs_a = self.gen_net('f_a', self.fake_obs_b, self.a_obs_dim)
        self.inv_obs_b = self.gen_net('f_b', self.fake_obs_a, self.b_obs_dim)

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
            self.loss_d += self.gp * 10
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
        self.saver = tf.train.Saver()

    def gen_net(self, prefix, inp, out_dim):
        pre_dim = int(inp.get_shape()[-1])
        #if prefix[0] == 'f':
        #    return inp
        # if prefix == 'f_a':
        #     return inp * np.array([[1, 1, 0.5]])
        # if prefix == 'f_b':
        #     return inp * np.array([[1, 1, 2.0]])
        if prefix[0] == 'f':
            hidden = self.hidden_f
        else:
            hidden = self.hidden_g
        out = relu_layer(prefix + '.1', pre_dim, hidden, inp)
        out = relu_layer(prefix + '.2', hidden, hidden, out)
        out = relu_layer(prefix + '.3', hidden, hidden, out)
        out = lib.ops.linear.Linear(prefix + '.4', hidden, out_dim, out)
        return out

    def dis_net(self, prefix, inp):
        pre_dim = int(inp.get_shape()[-1])
        out = relu_layer(prefix + '.1', pre_dim, self.hidden_d, inp)
        out = relu_layer(prefix + '.2', self.hidden_d, self.hidden_d, out)
        out = relu_layer(prefix + '.3', self.hidden_d, self.hidden_d, out)
        out = lib.ops.linear.Linear(prefix + '.4', self.hidden_d, 1, out)
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

    # suppose have same horizon H
    def train(self, args, expert_a, expert_b, eval_on=True,
              ck_dir=None):
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

        if eval_on:
            self.visual_evaluation(expert_a, expert_b, 0)

        start_time = time.time()
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
                if (epoch_idx + 1) % 100 == 0 and eval_on:
                    self.visual_evaluation(expert_a, expert_b,
                                           (epoch_idx + 1) // 100)
                if ck_dir is not None:
                    self.store(ck_dir, epoch_idx + 1)
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
        obs_b, act_b = self.sess.run([self.fake_obs_b, self.fake_act_b],
                                     feed_dict=demos)
        if self.vis_mode == 'synthetic':
            show_trajectory(self.env_b, obs_b, act_b,
                            demos[self.real_obs_a][0, :],
                            self.dir_name + '/img/' + str(id) + 'a2b_T.jpg')
            distribution_diff(demos[self.real_obs_a], demos[self.real_act_b],
                              demos[self.real_obs_b], demos[self.real_act_b],
                              obs_b, act_b,
                              self.dir_name + '/img/' + str(id) + 'a2b_D.jpg')
        else:
            prefix = self.dir_name + '/t' + str(id)
            if os.path.isdir(prefix):
                pass
            else:
                os.mkdir(prefix)
            save_trajectory_images(self.env_b, obs_b, act_b, prefix)
            save_video(prefix + '/real', obs_b.shape[0])
            save_video(prefix + '/imag', obs_b.shape[0])

    def evaluation(self, expert_a, expert_b, checkpoint_dir):
        if self.load(checkpoint_dir):
            demos = self.get_demo(expert_a, expert_b, is_train=False)
            horizon = demos[self.real_obs_a].shape[0]

            path_gta = self.dir_name + '/ground_truth_a'
            generate_dir(path_gta)
            save_trajectory_images(self.env_a, demos[self.real_obs_a],
                                   demos[self.real_act_a], path_gta)
            save_video(self.dir_name + '/ground_truth_a/real', horizon)

            path_gtb = self.dir_name + '/ground_truth_b'
            generate_dir(path_gtb)
            save_trajectory_images(self.env_a, demos[self.real_obs_a],
                                   demos[self.real_act_a], path_gtb)
            save_video(self.dir_name + '/ground_truth_b/real', horizon)
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
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
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
