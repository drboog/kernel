from .model import MMD_GAN, tf


class GAN(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        config.dof_dim = 1
        super(GAN, self).__init__(sess, config, **kwargs)

    def set_loss(self, G, images):
        self.d_loss = tf.reduce_mean(tf.nn.softplus(-images) + tf.nn.softplus(G))
        self.g_loss = tf.reduce_mean(tf.nn.softplus(-G))
        self.optim_name = 'gan%d_loss' % int(self.config.gradient_penalty)

        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
