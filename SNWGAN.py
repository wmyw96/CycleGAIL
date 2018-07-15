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


class WGAN(object):
    def __init__(self, name, sess, clip, n_z, n_x, scale=10.0, loss_metric='L1',
                 checkpoint_dir=None, loss='gan'):

        self.sess = sess
        self.clip = clip
        self.n_z = n_z
        self.n_x = n_x
        self.scale = scale

        self.loss_metric = loss_metric
        self.checkpoint_dir = checkpoint_dir
        self.dir_name = name
        self.use_spect = loss == 'wgan-sn'
        self.loss = loss
        print('WGAN-SN: Start building graph ...')
        self.build_model(spect=self.use_spect)
        print('WGAN-SN: Build graph finished !')

    def build_model(self, spect):
        self.real_a = tf.placeholder(tf.float32, [None, self.n_x])
        self.z = tf.placeholder(tf.float32, [None, self.n_z])
        self.fake_a = self.gen_net('g_a', self.z, self.n_x)

        self.critic_real_a = self.dis_net('d_a', self.real_a)
        self.critic_fake_a = self.dis_net('d_a', self.fake_a)

        # wgan loss
        self.w_dist = tf.reduce_mean(self.critic_real_a - self.critic_fake_a)
        self.loss_d = -self.w_dist
        if self.loss == 'wgan-gp':
            alpha = tf.random_uniform([tf.shape(self.real_a)[0], 1], 0., 1.)
            hat_a = alpha * self.real_a + ((1 - alpha) * self.fake_a)
            critic_hat_a = self.dis_net('d_a', hat_a)
            gradients = tf.gradients(critic_hat_a, [hat_a])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                           reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
            self.loss_d += gradient_penalty * self.scale

        self.loss_g = -tf.reduce_mean(self.critic_fake_a)

        self.params_d = lib.params_with_name('d_a')
        self.params_g = lib.params_with_name('g_a')
        self.saver = tf.train.Saver()

    def gen_net(self, prefix, inp, output_dim, name=None):
        pre_dim = int(inp.get_shape()[-1])
        #out = tf.identity(inp)
        out = relu_layer(prefix + '.1', pre_dim, 128, inp)
        out = relu_layer(prefix + '.2', 128, 128, out)
        out = relu_layer(prefix + '.3', 128, 128, out)
        out = lib.ops.linear.Linear(prefix + '.4', 128, 2, out)
        return out

    def dis_net(self, prefix, inp_a, update_collection=None):
        # mlp net
        pre_dim = int(inp_a.get_shape()[-1])
        #out = tf.identity(inp_a)
        out = relu_layer(prefix + '.1', pre_dim, 128, inp_a)
        out = relu_layer(prefix + '.2', 128, 128, out)
        out = relu_layer(prefix + '.3', 128, 128, out)
        out = lib.ops.linear.Linear(prefix + '.4', 128, 1, out)
        return out

    def clip_trainable_params(self, params):
        ops = []
        for p in params:
            ops.append(
                p.assign(tf.clip_by_value(p, -self.clip, self.clip)))
        return ops

    def show_params(self, name, params):
        print('Training Parameters for %s module' % name)
        for param in params:
            print(param.name, ': ', param.get_shape())

    # suppose have same horizon H
    def train(self, args, data):
        # data: numpy, [N x n_x]

        print(self.loss)

        if self.loss == 'wgan':
            # lr = 5e-5
            self.d_opt = \
                tf.train.RMSPropOptimizer(args.lr).\
                    minimize(self.loss_d, var_list=self.params_d)
            self.g_opt = \
                tf.train.RMSPropOptimizer(args.lr).\
                    minimize(self.loss_g, var_list=self.params_g)
            self.clip_d = self.clip_trainable_params(self.params_d)
        else:
            self.d_opt = \
                tf.train.AdamOptimizer(args.lr, beta1=0.5, beta2=0.9).\
                    minimize(self.loss_d, var_list=self.params_d)
            self.g_opt = \
                tf.train.AdamOptimizer(args.lr, beta1=0.5, beta2=0.9).\
                    minimize(self.loss_g, var_list=self.params_g)
        self.show_params('generator g', self.params_g)
        self.show_params('discriminator d', self.params_d)
        # clip=0.01
        tf.global_variables_initializer().run()
        self.visual_decision(data)
        bsz = args.batch_size

        ls_ds = []
        ls_gs = []
        w_ds = []
        for epoch_idx in range(0, args.epoch):
            start_time = time.time()

            if epoch_idx % 500 == 0 or epoch_idx < 25:
                n_c = 100
            else:
                n_c = args.n_c + 1

            for i in range(n_c):
                x = data(args.batch_size)
                z = np.random.normal(0, 1, size=(bsz, self.n_z))
                ls_d, _, a = self.sess.run(
                     [self.loss_d, self.d_opt, self.fake_a],
                     feed_dict={self.real_a: x,
                                self.z: z})
                if self.loss == 'wgan':
                    self.sess.run(self.clip_d)
                ls_ds.append(ls_d)

            x = data(args.batch_size)
            z = np.random.normal(0, 1, size=(bsz, self.n_z))
            ls_g, _, w_d = \
                self.sess.run([self.loss_g, self.g_opt, self.w_dist],
                              feed_dict={
                                  self.z: z,
                                  self.real_a: x
                              })
            ls_gs.append(ls_g)
            w_ds.append(w_d)

            end_time = time.time()
            if (epoch_idx + 1) % 100 == 0:
                print('Round %d (%.3f s), loss D = %.6f, loss G = %.6f, '
                      'w_dist = %.6f' %
                      (epoch_idx, end_time - start_time, float(np.mean(ls_ds)),
                      float(np.mean(ls_gs)), float(np.mean(w_ds))))
                self.visual_decision(data, 'logs/' + self.dir_name + '_img/' + str(epoch_idx) + '.png')
                ls_ds = []
                ls_gs = []
                w_ds = []

    def grid(self, x_l, x_r, x_steps, y_l, y_r, y_steps):
        delta_x = (x_r - x_l) / (x_steps - 1)
        delta_y = (y_r - y_l) / (y_steps - 1)
        data = np.zeros((x_steps * y_steps, 2))
        n = 0
        for i in range(x_steps):
            for j in range(y_steps):
                data[n, :] = np.array([delta_x * i + x_l, delta_y * j + y_l])
                n += 1
        return data

    def togrid(self, x, nx, ny):
        data = np.zeros((nx, ny))
        n = 0
        for i in range(nx):
            for j in range(ny):
                data[j, i] = x[n]
                n += 1
        return data

    def visual_decision(self, real_data_t, path=None):
        real_data = real_data_t(2000)
        data = self.grid(-1.5, 1.5, 500, -1.5, 1.5, 500)
        x = np.ogrid[-1.5:1.5:500j]
        y = np.ogrid[-1.5:1.5:500j]
        v = self.sess.run(self.critic_real_a, feed_dict={self.real_a: data})
        v = self.togrid(v, 500, 500)
        plt.figure(figsize=(8, 8))
        plt.contourf(x.reshape(-1), y.reshape(-1), v, 20)
        plt.scatter(real_data[:, 0], real_data[:, 1], color='r', s=0.3)
        fake_data = np.random.normal(0, 1, size=(real_data.shape[0], self.n_z))
        fake_critic, real_critic = self.sess.run([self.critic_fake_a, self.critic_real_a],
                                                 feed_dict={self.z: fake_data, self.real_a: real_data})
        fake_data = self.sess.run(self.fake_a, feed_dict={self.z: fake_data})
        plt.scatter(fake_data[:, 0], fake_data[:, 1], color='b', s=0.3)
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()
