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
                 w_obs_a, w_obs_b, w_act_a, w_act_b,
                 lambda_g, lambda_f, gamma, use_orac_loss, metric='L1',
                 checkpoint_dir=None, spect=True, loss='wgan',
                 vis_mode='synthetic', concat_steps=0):
        self.sess = sess
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
        self.build_model(w_obs_a, w_obs_b, w_act_a, w_act_b)
        print('CycleGAIL: Build graph finished !')

    def markov_concat(self, current):
        stacks = []
        for i in range(self.concat_steps):
            stacks.append(current[i: -self.concat_steps + i, :])
            #print(current[i: -self.concat_steps + i, :].get_shape())
        stacks.append(current[self.concat_steps:, :])
        #print(current[self.concat_steps:, :].get_shape())
        return tf.concat(stacks, axis=1)

    def gradient_penalty(self, name, real, fake):
        alpha = tf.random_uniform([tf.shape(real)[0], 1], 0., 1.)

        hat = alpha * real + ((1 - alpha) * fake)
        critic_hat_a = self.dis_net(name, self.markov_concat(hat))
        gradients = tf.gradients(critic_hat_a, [hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        return tf.reduce_mean((slopes - 1) ** 2)

    def build_model(self, w_obs_a, w_obs_b, w_act_a, w_act_b):
        self.real_act_a = tf.placeholder(tf.float32, [100, self.a_act_dim])
        self.real_act_b = tf.placeholder(tf.float32, [100, self.b_act_dim])
        self.real_obs_a = tf.placeholder(tf.float32, [100, self.a_obs_dim])
        self.real_obs_b = tf.placeholder(tf.float32, [100, self.b_obs_dim])
        self.orac_obs_a = tf.placeholder(tf.float32, [100, self.a_obs_dim])
        self.orac_obs_b = tf.placeholder(tf.float32, [100, self.b_obs_dim])
        self.ts = tf.placeholder(tf.float32, [100, 1])

        self.fake_act_a = self.gen_net('g_a', self.real_act_b, self.a_act_dim)
        self.fake_act_b = self.gen_net('g_b', self.real_act_a, self.b_act_dim)
        self.inv_act_a = self.gen_net('g_a', self.fake_act_b, self.a_act_dim)
        self.inv_act_b = self.gen_net('g_b', self.fake_act_a, self.b_act_dim)

        self.fake_obs_a = self.gen_net('f_a', self.real_obs_b, self.a_obs_dim)
        self.fake_obs_b = self.gen_net('f_b', self.real_obs_a, self.b_obs_dim)
        self.inv_obs_a = self.gen_net('f_a', self.fake_obs_b, self.a_obs_dim)
        self.inv_obs_b = self.gen_net('f_b', self.fake_obs_a, self.b_obs_dim)

        self.cycle_act_a = \
            cycle_loss(self.real_act_a, self.inv_act_a, self.metric, w_act_a)
        self.cycle_act_b = \
            cycle_loss(self.real_act_b, self.inv_act_b, self.metric, w_act_b)
        self.cycle_obs_a = \
            cycle_loss(self.real_obs_a, self.inv_obs_a, self.metric, w_obs_a)
        self.cycle_obs_b = \
            cycle_loss(self.real_obs_b, self.inv_obs_b, self.metric, w_obs_b)

        self.real_a = tf.concat([self.ts, self.real_obs_a, self.real_act_a], 1)
        self.fake_a = tf.concat([self.ts, self.fake_obs_a, self.fake_act_a], 1)
        self.real_b = tf.concat([self.ts, self.real_obs_b, self.real_act_b], 1)
        self.fake_b = tf.concat([self.ts, self.fake_obs_b, self.fake_act_b], 1)
        self.d_real_a = self.dis_net('d_a', self.markov_concat(self.real_a))
        self.d_real_b = self.dis_net('d_b', self.markov_concat(self.real_b))
        self.d_fake_a = self.dis_net('d_a', self.markov_concat(self.fake_a))
        self.d_fake_b = self.dis_net('d_b', self.markov_concat(self.fake_b))

        self.wdist_a = tf.reduce_mean(self.d_real_a - self.d_fake_a)
        self.wdist_b = tf.reduce_mean(self.d_real_b - self.d_fake_b)
        self.wdist = self.wdist_a + self.wdist_b
        self.loss_d = - self.wdist_a - self.wdist_b
        if self.loss == 'wgan-gp':
            self.gp = self.gradient_penalty('d_a', self.real_a, self.fake_a)
            self.gp += self.gradient_penalty('d_b', self.real_b, self.fake_b)
            self.loss_d += self.gp * 10
        self.loss_gf_a = -tf.reduce_mean(self.d_fake_a)
        self.loss_gf_b = -tf.reduce_mean(self.d_fake_b)

        self.loss_g = self.loss_gf_a + self.loss_gf_b + \
            self.lambda_g * (self.cycle_act_a + self.cycle_act_b)
        self.loss_f = self.loss_gf_a + self.loss_gf_b + \
            self.lambda_f * (self.cycle_obs_a + self.cycle_obs_b)
        self.loss_gf = self.loss_gf_a + self.loss_gf_b + \
            self.lambda_g * (self.cycle_act_a + self.cycle_act_b) + \
            self.lambda_f * (self.cycle_obs_a + self.cycle_obs_b)

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

        self.params_g_a = lib.params_with_name('g_a')
        self.params_g_b = lib.params_with_name('g_b')
        self.params_f_a = lib.params_with_name('f_a')
        self.params_f_b = lib.params_with_name('f_b')
        self.params_d_a = lib.params_with_name('d_a')
        self.params_d_b = lib.params_with_name('d_b')
        self.params_d = self.params_d_a + self.params_d_b
        self.params_g = self.params_g_a + self.params_g_b
        self.params_f = self.params_f_a + self.params_f_b
        self.params_gf = self.params_f + self.params_g
        self.saver = tf.train.Saver()

    def gen_net(self, prefix, inp, out_dim):
        pre_dim = int(inp.get_shape()[-1])
        # if prefix[0] == 'f':
        #     return inp
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
        if is_train:
            obs_a, act_a, ts = expert_a.next_batch()
            obs_b, act_b, ts = expert_b.next_batch()
        else:
            obs_a, act_a, ts = expert_a.next_demo(is_train)
            obs_b, act_b, ts = expert_b.next_demo(is_train)
        demos = {self.real_obs_a: obs_a,
                 self.real_act_a: act_a,
                 self.real_obs_b: obs_b,
                 self.real_act_b: act_b,
                 self.ts: ts}
        return demos

    def local_oracle(self, sobs, sact, tobs_gen, tact_gen, tenv, demos):
        horizon = sobs.shape[0]
        obs_dim = tobs_gen.get_shape()[1]
        obs_orac = np.zeros((horizon, obs_dim))
        tobs = self.sess.run(tobs_gen, feed_dict=demos)
        tact = self.sess.run(tact_gen, feed_dict=demos)
        tenv.reset()
        tenv.env.set_state(sobs[0, :9], sobs[0, 9:])
        reward_sum = 0.0
        for i in range(horizon - 1):
            # local oracle: unstable
            # tenv.reset()
            # tenv.env.set_state(tobs[i, :9], tobs[i, 9:])
            _1, rd, _2, _3 = tenv.step(tact[i])
            reward_sum += rd
            nxt_obs = \
                np.concatenate([tenv.env.model.data.qpos,
                                tenv.env.model.data.qvel]).reshape(-1)
            obs_orac[i + 1, :] = nxt_obs
        obs_orac[0, :] = sobs[0, :]
        return obs_orac, reward_sum

    def get_oracle(self, demos):
        act_a = demos[self.real_act_a]
        obs_a = demos[self.real_obs_a]
        act_b = demos[self.real_act_b]
        obs_b = demos[self.real_obs_b]
        demos[self.orac_obs_a], rd_a = \
            self.local_oracle(obs_b, act_b, self.fake_obs_a, self.fake_act_a,
                              self.env_a, demos)
        demos[self.orac_obs_b], rd_b = \
            self.local_oracle(obs_a, act_a, self.fake_obs_b, self.fake_act_b,
                              self.env_b, demos)
        return demos, rd_a, rd_b

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
            self.gf_opt = \
                tf.train.RMSPropOptimizer(args.lr).\
                    minimize(self.loss_gf, var_list=self.params_gf)
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
            self.gf_opt = \
                tf.train.AdamOptimizer(args.lr, beta1=0.5, beta2=0.9).\
                minimize(self.loss_gf, var_list=self.params_gf)
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
        ls_gfs = []
        ls_bgs = []
        ls_bfs = []
        wds = []
        ls_igs = []
        ls_ifs = []

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
            # demos, _, __ = self.get_oracle(demos)
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
            _, ls_g, ls_f, ls_gf, wd, ls_ig, ls_if, ls_bf, ls_bg = \
                self.sess.run([self.gf_opt, self.loss_g, self.loss_f,
                               self.loss_gf, self.wdist,
                               self.loss_ident_g, self.loss_ident_f,
                               self.loss_best_f, self.loss_best_g],
                              feed_dict=demos)
            ls_gs.append(ls_g)
            ls_fs.append(ls_f)
            ls_gfs.append(ls_gf)
            ls_igs.append(ls_ig)
            ls_ifs.append(ls_if)
            ls_bfs.append(ls_bf)
            ls_bgs.append(ls_bg)
            wds.append(wd)

            ls_fs.append(ls_f)

            if (epoch_idx + 1) % args.log_interval == 0:
                end_time = time.time()
                if eval_on:
                    self.visual_evaluation(expert_a, expert_b,
                                           (epoch_idx + 1) // 100)
                if ck_dir is not None:
                    self.store(ck_dir, epoch_idx + 1)
                print('Epoch %d (%.3f s), loss D = %.6f, loss G = %.6f,'
                      'loss F = %6f, w_dist = %.9f, loss G_F = %.6f, '
                      'loss ident G = %.6f, loss ident F = %.6f, '
                      'loss best G = %.6f, loss best F = %.6f' %
                      (epoch_idx, end_time - start_time, float(np.mean(ls_ds)),
                       float(np.mean(ls_gs)), float(np.mean(ls_fs)),
                       float(np.mean(wds)), float(np.mean(ls_gfs)),
                       float(np.mean(ls_igs)), float(np.mean(ls_ifs)),
                       float(np.mean(ls_bgs)), float(np.mean(ls_bfs))))
                ls_ds = []
                ls_gs = []
                ls_fs = []
                wds = []
                ls_gfs = []
                ls_ifs = []
                ls_igs = []
                ls_bgs, ls_bfs = [], []
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
