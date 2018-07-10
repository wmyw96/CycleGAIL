import os
import time
import tensorflow as tf
import numpy as np
from utils import show_trajectory


def cycle_loss(origin, reconstructed, metric):
    if metric == 'L1':
        return tf.reduce_mean(tf.abs(origin - reconstructed))
    if metric == 'L2':
        return tf.reduce_mean(tf.square(origin - reconstructed))
    raise NotImplementedError


def dense_layer(inp, size, name, weight_init, bias_init=0, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [inp.get_shape()[1], size],
                            initializer=weight_init)
        b = tf.get_variable('b', [size],
                            initializer=tf.constant_initializer(bias_init))
        return tf.nn.bias_add(tf.matmul(inp, w), b)


def get_flatten_dim(shape):
    dim = 1
    for sub_dim in shape:
        dim *= int(sub_dim)
    return dim


# copy from openai/baselines/common/tf_util
def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):
        # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


class CycleGAIL(object):
    def __init__(self, sess, clip, bound, env_a, env_b, init_align,
                 act_space_a, act_space_b, obs_space_a, obs_space_b,
                 archi_d_a, archi_d_b, archi_g_ab, archi_g_ba,
                 archi_f_ab, archi_f_ba,
                 lambda_g_a=20.0, lambda_g_b=20., lambda_f_a = 20.,
                 lambda_f_b=20.0, loss_metric='L1',
                 checkpoint_dir=None):
        self.lambda_g_a = lambda_g_a
        self.lambda_g_b = lambda_g_b
        self.lambda_f_a = lambda_f_a
        self.lambda_f_b = lambda_f_b

        self.sess = sess
        self.clip = clip
        self.bound = bound
        self.env_a = env_a
        self.env_b = env_b
        self.init_align = init_align

        self.loss_metric = loss_metric
        self.checkpoint_dir = checkpoint_dir
        self.dir_name = ''
        self.act_space_a_shape = act_space_a
        self.act_space_b_shape = act_space_b
        self.obs_space_a_shape = obs_space_a
        self.obs_space_b_shape = obs_space_b
        self.archi_d_a = archi_d_a
        self.archi_d_b = archi_d_b
        self.archi_g_ab = archi_g_ab
        self.archi_g_ba = archi_g_ba
        self.archi_f_ab = archi_f_ab
        self.archi_f_ba = archi_f_ba
        print('CycleGAIL: Start building graph ...')
        self.build_model()
        print('CycleGAIL: Build graph finished !')

    def build_model(self):
        self.real_act_a = tf.placeholder(tf.float32,
                                         [None] + self.act_space_a_shape,
                                         name='real_act_a')
        self.real_state_a = tf.placeholder(tf.float32,
                                           [None] + self.obs_space_a_shape,
                                           name='real_obs_a')
        self.orac_state_a = tf.placeholder(tf.float32,
                                           [None] + self.obs_space_a_shape,
                                           name='oracle_obs_a')
        self.real_act_b = tf.placeholder(tf.float32,
                                         [None] + self.act_space_b_shape,
                                         name='real_act_b')
        self.real_state_b = tf.placeholder(tf.float32,
                                           [None] + self.obs_space_a_shape,
                                           name='real_obs_b')
        self.orac_state_b = tf.placeholder(tf.float32,
                                           [None] + self.obs_space_a_shape,
                                           name='oracle_obs_b')

        # generative action
        self.a2b = self.gen_net('g_a2b', self.real_act_a, self.archi_g_ab,
                                self.act_space_b_shape, reuse=False)
        self.b2a = self.gen_net('g_b2a', self.real_act_b, self.archi_g_ba,
                                self.act_space_a_shape, reuse=False)
        self.a2b2a = self.gen_net('g_b2a', self.a2b, self.archi_g_ba,
                                  self.act_space_a_shape, reuse=True)
        self.b2a2b = self.gen_net('g_a2b', self.b2a, self.archi_g_ab,
                                  self.act_space_b_shape, reuse=True)
        # generative state
        self.fake_state_b = \
            self.gen_net('f_a2b', self.real_state_a,
                         self.archi_f_ab, self.obs_space_b_shape, reuse=False)
        self.fake_state_a = \
            self.gen_net('f_b2a', self.real_state_b,
                         self.archi_f_ba, self.obs_space_a_shape, reuse=False)
        self.ffake_state_a = \
            self.gen_net('f_b2a', self.fake_state_b,
                         self.archi_f_ba, self.obs_space_a_shape, reuse=True)
        self.ffake_state_b = \
            self.gen_net('f_a2b', self.fake_state_a,
                         self.archi_f_ab, self.obs_space_b_shape, reuse=True)

        # action consistency loss
        self.lca = cycle_loss(self.real_act_a, self.a2b2a, self.loss_metric)
        self.lcb = cycle_loss(self.real_act_b, self.b2a2b, self.loss_metric)
        self.loss_g_c = self.lca * self.lambda_g_a + self.lcb * self.lambda_g_b

        # state consistency loss
        self.cfa = \
            cycle_loss(self.real_state_a, self.ffake_state_a, self.loss_metric)
        self.cfb = \
            cycle_loss(self.real_state_b, self.ffake_state_b, self.loss_metric)
        self.loss_f_c = self.cfa * self.lambda_f_a + self.cfb * self.lambda_f_b

        # state oracle loss
        self.loss_f_orcale_a = \
            cycle_loss(self.fake_state_a, self.orac_state_a, self.loss_metric)
        self.loss_f_orcale_b = \
            cycle_loss(self.fake_state_b, self.orac_state_b, self.loss_metric)
        self.loss_f_o = self.loss_f_orcale_a * self.lambda_f_a + \
                        self.loss_f_orcale_b * self.lambda_f_b

        # discriminator
        self.d_real_out_a = self.dis_net('d_a', self.real_state_a,
                                         self.real_act_a, self.archi_d_a,
                                         reuse=False)
        self.d_real_out_b = self.dis_net('d_b', self.real_state_b,
                                         self.real_act_b, self.archi_d_b,
                                         reuse=False)
        self.d_fake_out_a = self.dis_net('d_a', self.fake_state_a,
                                         self.b2a, self.archi_d_a,
                                         reuse=True)
        self.d_fake_out_b = self.dis_net('d_b', self.fake_state_b,
                                         self.a2b, self.archi_d_b,
                                         reuse=True)

        # wgan loss
        self.loss_d_a = tf.reduce_mean(self.d_fake_out_a) - \
                        tf.reduce_mean(self.d_real_out_a)
        self.loss_d_b = tf.reduce_mean(self.d_fake_out_b) - \
                        tf.reduce_mean(self.d_real_out_b)
        self.loss_d = self.loss_d_a + self.loss_d_b
        self.loss_g = self.loss_g_c - tf.reduce_mean(self.d_fake_out_a) - \
            tf.reduce_mean(self.d_fake_out_b)
        self.loss_f_woo = self.loss_f_c - tf.reduce_mean(self.d_fake_out_a) - \
                      tf.reduce_mean(self.d_fake_out_b)
        self.loss_f = self.loss_f_o + self.loss_f_woo

        self.params = tf.trainable_variables()
        self.params_d_a = [p for p in self.params if 'd_a' in p.name]
        self.params_d_b = [p for p in self.params if 'd_b' in p.name]
        self.params_g_ab = [p for p in self.params if 'g_a2b' in p.name]
        self.params_g_ba = [p for p in self.params if 'g_b2a' in p.name]
        self.params_f_ab = [p for p in self.params if 'f_a2b' in p.name]
        self.params_f_ba = [p for p in self.params if 'f_b2a' in p.name]
        self.params_d = self.params_d_a + self.params_d_b
        self.params_g = self.params_g_ab + self.params_g_ba
        self.params_f = self.params_f_ab + self.params_f_ba
        self.saver = tf.train.Saver()

    def clip_trainable_params(self, params):
        ops = []
        for p in params:
            ops.append(p.assign(tf.clip_by_value(p, -self.clip, self.clip)))
        return ops

    def gen_net(self, prefix, inp, archi, output_shape, reuse=False):
        # mlp net
        if prefix == 'f_a2b': #or prefix == 'g_a2b':
            return inp * np.array([[1, 1, 2]])
        if prefix == 'f_b2a': #or prefix == 'g_b2a':
            return inp * np.array([[1, 1, 0.5]])
        input_dim = get_flatten_dim(inp.get_shape()[1:])
        output_dim = get_flatten_dim(output_shape)

        out = tf.reshape(inp, [-1, input_dim])
        with tf.variable_scope(prefix, reuse=reuse):
            for (i, layer_size) in enumerate(archi):
                out = \
                    tf.nn.relu(dense_layer(out, layer_size,
                                           'layer%d' % i,
                                           normc_initializer(1.0),
                                           reuse=reuse))
            out = dense_layer(out, output_dim,
                              prefix + '_output',
                              normc_initializer(1.0), reuse=reuse)
            out = tf.reshape(out, [-1] + output_shape)
            return out

    def dis_net(self, prefix, inp_a, inp_s, archi, reuse=False):
        # mlp net
        inp = tf.concat([inp_a, inp_s], 0)
        input_dim = get_flatten_dim(inp.get_shape()[1:])

        out = tf.reshape(inp, [-1, input_dim])
        with tf.variable_scope(prefix, reuse=reuse):
            for (i, layer_size) in enumerate(archi):
                out = \
                    tf.nn.relu(dense_layer(out, layer_size,
                                           'layer%d' % i,
                                           normc_initializer(1.0),
                                           reuse=reuse))
            out = dense_layer(out, 1, prefix + '_output',
                              normc_initializer(1.), reuse=reuse)
            return out

    def get_demo(self, expert_a, expert_b):
        state_a, action_a = expert_a.next_demo()
        state_b, action_b = expert_b.next_demo()
        demos = {self.real_state_a: state_a,
                 self.real_act_a: action_a,
                 self.real_state_b: state_b,
                 self.real_act_b: action_b}
        return demos

    def sample(self, init_state, action, env):
        state_shape = [action.shape[0]] + list(init_state.shape)
        state = np.zeros(state_shape)
        env.reset(init_state)
        for i in range(action.shape[0] - 1):
            state[i + 1, :], _, _, _ = env.step(action[i, :])
        return state

    def get_oracle(self, demos):
        act_a, act_b = self.sess.run([self.b2a, self.a2b], demos)
        state_a = self.sample(self.init_align(demos[self.real_state_a][0, :]),
                              act_a, self.env_a)
        state_b = self.sample(self.init_align(demos[self.real_state_b][0, :]),
                              act_b, self.env_b)
        demos[self.orac_state_a] = state_a
        demos[self.orac_state_b] = state_b
        return demos

    def show_params(self, name, params):
        print('Training Parameters for %s module' % name)
        for param in params:
            print(param.name, ': ', param.get_shape())

    # suppose have same horizon H
    def train(self, args, expert_a, expert_b, n_c):
        self.d_opt = tf.train.RMSPropOptimizer(args.lr).\
            minimize(self.loss_d, var_list=self.params_d)
        self.g_opt = tf.train.RMSPropOptimizer(args.lr).\
            minimize(self.loss_g, var_list=self.params_g)
        self.show_params('Generator', self.params_g)
        #self.f_opt = tf.train.RMSPropOptimizer(args.lr).\
        #    minimize(self.loss_f_woo, var_list=self.params_f)
        self.clip_d = self.clip_trainable_params(self.params_d)
        tf.global_variables_initializer().run()
        self.writer = tf.summary.FileWriter('./logs/' + self.dir_name,
                                            self.sess.graph)

        for epoch_idx in range(0, args.epoch):
            # state [H, D_S]
            # action [H, D_A]
            ls_ds = []
            ls_gs = []
            ls_fs = []
            ls_gcs = []
            ls_fcs = []
            ls_fos = []
            start_time = time.time()
            for round_idx in range(0, args.round):
                #print('round # %d' % round_idx)
                for i in range(n_c):
                    demos = self.get_demo(expert_a, expert_b)
                    ls_d, _ = self.sess.run([self.loss_d, self.d_opt],
                                            feed_dict=demos)
                    self.sess.run(self.clip_d)
                    ls_ds.append(ls_d)
                demos = self.get_demo(expert_a, expert_b)
                ls_g, ls_gc, _ = self.sess.run([self.loss_g, self.loss_g_c,
                                                self.g_opt],
                                               feed_dict=demos)
                ls_gs.append(ls_g)
                ls_gcs.append(ls_gc)
                for i in range(1):
                    demos = self.get_demo(expert_a, expert_b)
                    demos = self.get_oracle(demos)
                    ls_f, ls_f_c, ls_f_o, gen_state, gs2, cfa = \
                        self.sess.run([self.loss_f, self.loss_f_c,
                                       self.loss_f_o, self.fake_state_b,
                                       self.ffake_state_a, self.cfb],
                                      feed_dict=demos)
                    #print('====== Coupling ======')
                    #print(np.mean(np.abs(demos[self.real_state_a] - gs2)))
                    #print(cfa)
                    ls_fs.append(ls_f)
                    ls_fcs.append(ls_f_c)
                    ls_fos.append(ls_f_o)
            end_time = time.time()
            print('Epoch %d (%.3f s), loss D = %.3f, loss G = %.3f (o = %.3f),'
                  ' loss F = %.3f (o = %.3f, c = %.3f)' %
                  (epoch_idx, end_time - start_time, float(np.mean(ls_ds)),
                   float(np.mean(ls_gs)), float(np.mean(ls_gcs)),
                   float(np.mean(ls_fs)), float(np.mean(ls_fos)),
                   float(np.mean(ls_fcs))))
            if (epoch_idx + 1) % 50 == 0:
                self.trans_evaluation(1, expert_a, expert_b)

    def test(self):
        raise NotImplementedError

    def trans_visual(self, env, gen_state, gen_act, feed):
        state, act = self.sess.run([gen_state, gen_act], feed_dict=feed)
        show_trajectory(env, state, act)

    def trans_evaluation(self, round, expert_a, expert_b):
        for i in range(round):
            demos = self.get_demo(expert_a, expert_b)
            # a-> b
            show_trajectory(self.env_b, demos[self.real_state_a],
                            demos[self.real_act_a])
            self.trans_visual(self.env_b, self.fake_state_b, self.a2b, demos)

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError