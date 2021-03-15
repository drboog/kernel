from .model import MMD_GAN, tf


class Sinkhorn(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        self.config = config
        super(Sinkhorn, self).__init__(sess, config, **kwargs)

    def cost_matrix(self, x, y):  # compute the cost matrix (L2 distances)
        x_expand = tf.expand_dims(x, axis=-2)
        y_expand = tf.expand_dims(y, axis=-3)
        c = tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1)  # sum over the dimensions
        return c

    def M(self, u, v, eps):
        return (-self.c + tf.expand_dims(u, -2) + tf.expand_dims(v, -1))/eps  # the return shape is (batch_size, batch_size)

    def compute_loss(self, x, y):  # X and Y are batch of samples/transferred samples here, X is the real, Y is the fake
        self.c = self.cost_matrix(x, y)
        mu = tf.ones(self.batch_size)/self.batch_size  # shape (batch_size)
        nu = tf.ones(self.batch_size)/self.batch_size  # shape (batch_size)
        threshold = 10**(-1)  # threshold to stop the iteration
        epsilon = self.config.eps
        u = mu*0.
        v = nu*0.
        err = 0.  # some initialization
        for i in range(20):
            u1 = u  # used to check the error later
            u += (tf.log(mu) - tf.log(tf.reduce_sum(tf.exp(self.M(u, v, epsilon)), axis=-1) + 1e-6))
            v += (tf.log(nu) - tf.log(tf.reduce_sum(tf.exp(tf.transpose(self.M(u, v, epsilon))), axis=-1) + 1e-6))
            #err = tf.reduce_sum(tf.abs(u - u1))
        pi = tf.exp(self.M(u, v, epsilon))  # pi is the transportation plan, i.e. the coupling
        cost = tf.reduce_sum(pi*self.c)
        return cost, tf.exp(-self.c/epsilon)

    def set_loss(self, G, images):
        D_G = G#self.discriminator(G, self.batch_size)
        D_images = images#self.discriminator(images, self.batch_size)
        sinkhorn_loss, kernel_matrix = self.compute_loss(D_G, D_images)
        sinkhorn_loss_1, kernel_matrix_1 = self.compute_loss(D_G, D_G)
        sinkhorn_loss_2, kernel_matrix_2 = self.compute_loss(D_images, D_images)
        with tf.variable_scope('loss'):
            self.g_loss = (2*sinkhorn_loss - sinkhorn_loss_1 - sinkhorn_loss_2)
            self.d_loss = -self.g_loss
        self.optim_name = 'sinkhorn_loss'
        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
        print('[*] Loss set')
