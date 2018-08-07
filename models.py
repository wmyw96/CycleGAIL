import tensorflow as tf
import numpy as np


class AutoEncoder(object):
    def __init__(self, input_size, embed_size, hidden_size,
                 num_layers, lr):
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.build_graph(lr)

    def build_graph(self, lr):
        self.inp = tf.placeholder(tf.float32, [None, self.input_size])
        self.embed = self.mlp('embed_net', self.inp, self.num_layers,
                                self.hidden_size, self.embed_size, False)
        self.out = self.mlp('rinp_net', self.embed, self.num_layers,
                               self.hidden_size, self.input_size, False)
        self.embed_inp = tf.placeholder(tf.float32, [None, self.embed_size])
        self.embed_out = self.mlp('rinp_net', self.embed, self.num_layers,
                                  self.hidden_size, self.input_size)
        self.loss_r = tf.reduce_mean(tf.square(self.inp - self.out))

        t_vars = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9).\
            minimize(self.loss_r, var_list=t_vars)
        self.init = tf.global_variables_initializer()

    def mlp(self, prefix, inp, nlayers, hidden, out_dim, reuse=True):
        out = tf.layers.dense(inp, hidden, activation=tf.nn.relu,
                              name=prefix + '.1', reuse=reuse)
        for i in range(nlayers - 1):
            out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                                  name=prefix + '.2', reuse=reuse)
        out = tf.layers.dense(out, out_dim, activation=tf.nn.tanh,
                              name=prefix + '.out', reuse=reuse)
        return out

    def train(self, data, nepoches, batch_size, log_interval):
        N = data.shape[0]
        print('run epoch')
        index = np.array(range(N))
        self.sess.run(self.init)

        lss = []
        for epoch in range(nepoches):
            nbatch = N // batch_size
            np.random.shuffle(index)
            for t in range(nbatch):
                batch_data = data[index[t * batch_size,
                                        (t + 1) * batch_size], :]
                ls, _ = self.sess.run([self.loss_r, self.opt],
                                      feed_dict={self.inp: batch_data})
                lss.append(ls)
            if (epoch + 1) % log_interval == 0:
                print('Epoch %d: Reconstruction Loss = %.6f' %
                      (epoch, float(np.mean(lss))))
                lss = []

    def reconstruct(self, embed):
        return self.sess.run(self.embed_out, feed_dict={self.embed_inp: embed})

    def embed(self, inp):
        return self.sess.run(self.embed, feed_dict={self.inp: inp})
