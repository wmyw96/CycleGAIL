import tensorflow as tf


class MlpPolicy(object):
    def __init__(self, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        import load_policy
        with self.graph.as_default():
            self.policy_fn = load_policy.load_policy(loc)

    def run(self, obs):
        with self.sess:
            import tf_util
            tf_util.initialize()
            return self.policy_fn(obs[None, :])
