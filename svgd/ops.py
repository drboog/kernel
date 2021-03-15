from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
import numpy as np
import math
from tqdm import tqdm

def sqr_dist(x, y, e=1e-8):

    #assert len(list(x.get_shape())) == 2 and len(list(y.get_shape()))==2, 'illegal inputs'
    xx = tf.reduce_sum(tf.square(x) + 1e-10, axis=1)
    yy = tf.reduce_sum(tf.square(y) + 1e-10, axis=1)
    xy = tf.matmul(x, y, transpose_b=True)
    dist = tf.expand_dims(xx, 1) + tf.expand_dims(yy, 0) - 2.* xy
    return dist


def median_distance(H):
    V = tf.reshape(H, [-1])
    n = tf.size(V)
    top_k, _ = tf.nn.top_k(V, k= (n // 2) + 1)
    return tf.cond(
        tf.equal(n%2, 0),
        lambda: (top_k[-1] + top_k[-2]) / 2.0,
        lambda: top_k[-1]
    )
    #return tf.maximum(h, 1e-6)
    return h / tf.log(tf.cast(tf.shape(H)[0], tf.float32) + 1.)


def poly_kernel(x, subtract_mean=True, e=1e-8):
    if subtract_mean:
        x = x - tf.reduce_mean(x, axis=0)
    kxy = 1 + tf.matmul(x, x, transpose_b=True)
    kxkxy = x * x.get_shape()[0]
    return kxy, dxkxy


def rbf_kernel(x, h=-1, to3d=False):

    H = sqr_dist(x, x)
    if h == -1:
        h = tf.maximum(1e-6, median_distance(H))

    kxy = tf.exp(-H / h)
    dxkxy = -tf.matmul(kxy, x)
    sumkxy = tf.reduce_sum(kxy, axis=1, keep_dims=True)
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    if to3d: dxkxy = -(tf.expand_dims(x, 1) - tf.expand_dims(x, 0)) * tf.expand_dims(kxy, 2) * 2. / h
    return kxy, dxkxy


def imq_kernel(x, h=-1):
    H = sqr_dist(x, x)
    if h == -1:
        h = median_distance(H)

    kxy = 1. / tf.sqrt(1. + H / h) 

    dxkxy = .5 * kxy / (1. + H / h)
    dxkxy = -tf.matmul(dxkxy, x)
    sumkxy = tf.reduce_sum(kxy, axis=1, keep_dims=True)
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    return kxy, dxkxy


def kernelized_stein_discrepancy(X, score_q, kernel='rbf', h=-1, **model_params):
    n, dim = tf.cast(tf.shape(X)[0], tf.float32), tf.cast(tf.shape(X)[1], tf.float32)
    Sqx = score_q(X, **model_params)

    H = sqr_dist(X, X)
    if h == -1:
        h = median_distance(H) # 2sigma^2
    h = tf.sqrt(h/2.)
    # compute the rbf kernel
    Kxy = tf.exp(-H / h ** 2 / 2.)

    Sqxdy = -(tf.matmul(Sqx, X, transpose_b=True) - tf.reduce_sum(Sqx * X, axis=1, keep_dims=True)) / (h ** 2)

    dxSqy = tf.transpose(Sqxdy)
    dxdy = (-H / (h ** 4) + dim / (h ** 2))

    M = (tf.matmul(Sqx, Sqx, transpose_b=True) + Sqxdy + dxSqy + dxdy) * Kxy 
    #M2 = M - T.diag(T.diag(M)) 

    #ksd_u = tf.reduce_sum(M2) / (n * (n - 1)) 
    #ksd_v = tf.reduce_sum(M) / (n ** 2) 

    #return ksd_v
    return M


def svgd_gradient(x, grad, kernel='rbf', temperature=1., u_kernel=None, **kernel_params):
    assert x.get_shape()[1:] == grad.get_shape()[1:], 'illegal inputs and grads'
    p_shape = tf.shape(x)
    if tf.keras.backend.ndim(x) > 2:
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        grad = tf.reshape(grad, (tf.shape(grad)[0], -1))

    if u_kernel is not None:
        kxy, dxkxy = u_kernel['kxy'], u_kernel['dxkxy']
        dxkxy = tf.reshape(dxkxy, tf.shape(x))
    else:
        if kernel == 'rbf':
            kxy, dxkxy = rbf_kernel(x, **kernel_params)
        elif kernel == 'poly':
            kxy, dxkxy = poly_kernel(x)
        elif kernel == 'imq':
            kxy, dxkxy = imq_kernel(x)
        elif kernel == 'none':
            kxy = tf.eye(tf.shape(x)[0])
            dxkxy = tf.zeros_like(x)

        ###### code added 1/7/2020 #####
        elif kernel == 'hk':
            print(x, 'hk kernel input')
            kxy, dxkxy = hk_kernel(x, **kernel_params)
        elif kernel == 'hkdk':
            print(x, 'hk dk kernel input')
            kxy, dxkxy, _ = hkdk_kernel(x, **kernel_params)
        ###### code added 1/7/2020 #####

        else:
            raise NotImplementedError
    print(kxy, grad, dxkxy)
    svgd_grad = (tf.matmul(kxy, grad) + temperature * dxkxy) / tf.reduce_sum(kxy, axis=1, keep_dims=True)

    svgd_grad = tf.reshape(svgd_grad, p_shape)
    return svgd_grad


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def conv2d(inputs, num_outputs, activation_fn=tf.nn.relu,
           kernel_size=5, stride=2, padding='SAME', name="conv2d"):

    with tf.variable_scope(name):
        return tf.contrib.layers.conv2d( inputs, num_outputs, kernel_size, stride=stride, padding=padding, activation_fn=activation_fn)


def deconv2d(inputs, num_outputs, activation_fn=tf.nn.relu,
        kernel_size=5, stride=2, padding='SAME', name="deconv2d"):

    with tf.variable_scope(name):
        return tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, stride=stride, padding=padding, activation_fn=activation_fn)


def fc(input, output_shape, activation_fn=tf.nn.relu, init=None, name="fc"):
    if init is None: init = tf.glorot_uniform_initializer()
    output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn, weights_initializer=init)
    return output

# code below added on 1/7/2020
def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=2, update_collection=None, with_sigma=False, stop_grad=True):
    # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1
    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            u,
            tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
    )
    if stop_grad:
        u_final = tf.stop_gradient(u_final)
        v_final = tf.stop_gradient(v_final)

    if update_collection is None:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_final))
    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar

def linear(input_, output_size, name="Linear", stddev=0.01, scale=1.0, with_learnable_sn_scale=False, with_sn=False, bias_start=0.0, with_w=False, update_collection=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       tf.get_variable_scope().name)
        has_summary = any([('Matrix' in v.op.name) for v in scope_vars])
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))

        if with_sn:
            s = tf.get_variable('s', shape=[1], initializer=tf.constant_initializer(scale), trainable=with_learnable_sn_scale, dtype=tf.float32)

            matrix_bar, sigma = spectral_normed_weight(matrix, update_collection=update_collection, with_sigma=True)
            matrix_bar = s*matrix_bar
            mul = tf.matmul(input_, matrix_bar)

        else:
            mul = tf.matmul(input_, matrix)

        bias = tf.get_variable(
            "bias",
            [output_size],
            initializer=tf.constant_initializer(bias_start))

        if with_w:
            return mul + bias, matrix, bias
        else:
            return mul + bias


def any_function(name, z_, dim_f, layer_shape, sn, reuse_=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse_):
        output_ = z_
        outputs = []
        for i in range(len(layer_shape)):
            # output_ = slim.fully_connected(output_, layer_shape[i], activation_fn=tf.nn.relu, scope='omega_%i'%i)
            output_ = tf.nn.relu(linear(output_, layer_shape[i], name=name + 'linear_%i'%i, with_sn=sn))
            outputs.append(output_)
        # out = slim.fully_connected(output_, dim_f, activation_fn=None, scope='omega_last')
        out = linear(output_, dim_f, name=name + 'linear_last', with_sn=sn)
        outputs.append(out)
    return out, outputs

def any_function_no_sn(name, z_, dim_f, layer_shape, sn=False, reuse_=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse_):
        output_ = z_
        outputs = []
        for i in range(len(layer_shape)):
            output_ = slim.fully_connected(output_, layer_shape[i], activation_fn=tf.nn.relu, scope='omega_%i'%i)
            # output_ = tf.nn.relu(linear(output_, layer_shape[i], name=name + 'linear_%i'%i, with_sn=sn))
            outputs.append(output_)
        out = slim.fully_connected(output_, dim_f, activation_fn=None, scope='omega_last')
        # out = linear(output_, dim_f, name=name + 'linear_last', with_sn=sn)
        outputs.append(out)
    return out, outputs

def hk_kernel(x, dim_z=1, sn=False, h=1.):
    input_dim = x.get_shape().as_list()[-1]
    if input_dim <= 50:
        z, _ = any_function_no_sn('hk' + str(input_dim), x, dim_z, layer_shape=[input_dim//2, input_dim//4], sn=sn)
        H = sqr_dist(tf.stop_gradient(z), z)
        kxy = tf.exp(-H / h)
        d = kxy.get_shape().as_list()[1]
        gradients = tf.stack(
            [tf.gradients(kxy[i, :], x)[0] for i in range(d)])  #  this order is important !
        dxkxy = tf.reduce_sum(gradients, axis=0)

        sumkxy = tf.reduce_sum(kxy, axis=1, keep_dims=True)
        dxkxy = dxkxy + (x * sumkxy) * 2. / h
    else:
        H = sqr_dist(x, x)
        h = tf.maximum(1e-6, median_distance(H))
        kxy = tf.exp(-H / h)
        dxkxy = -tf.matmul(kxy, x)
        sumkxy = tf.reduce_sum(kxy, axis=1, keep_dims=True)
        dxkxy = (dxkxy + x * sumkxy) * 2. / h
    return kxy, dxkxy


def delete_diag(matrix):
    return matrix - tf.matrix_diag(tf.matrix_diag_part(matrix))  # return matrix, while k_ii is 0


def compute_sinkhorn_loss(data, kernel_matrix, batch_size, mu, nu):
    sinkhorn_loss, coupling_matrix = compute_loss(data, kernel_matrix, batch_size, mu, nu)
    return sinkhorn_loss, coupling_matrix


def compute_kde(kernel_matrix, batch_size):  # KDE based on kernel matrix
    kde = tf.reduce_sum(kernel_matrix, axis=-1)/batch_size
    return kde


def cost_matrix(x, y):  # compute the cost matrix (L2 distances)
    x_expand = tf.expand_dims(x, axis=-2)
    y_expand = tf.expand_dims(y, axis=-3)
    c = tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1)  # sum over the dimensions
    return c


def compute_loss(data, kernel_matrix, batch_size, mu, nu):
    def M(c, u, v, eps):
        return (c + tf.expand_dims(u, -2) + tf.expand_dims(v, -1))/eps  # the return shape is (batch_size, batch_size)

    epsilon = 10.
    u = mu * 0.
    v = nu * 0.
    c = cost_matrix(data, tf.stop_gradient(data))
    for i in range(10):
        u += (tf.log(mu + 1e-7) - tf.log(tf.reduce_sum(tf.exp(M(c, u, v, epsilon)), axis=-1) + 1e-7))
        v += (tf.log(nu + 1e-7) - tf.log(tf.reduce_sum(tf.exp(tf.transpose(M(c, u, v, epsilon))), axis=-1) + 1e-7))
    pi = tf.exp(M(c, u, v, epsilon))  # pi is the transportation plan, i.e. the coupling
    pi /= tf.stop_gradient(tf.reduce_sum(pi) + 1e-7)  # the sinkhorn algorithm produce a unique solution UP TO A CONSTANT, we make it a joint probability
    cost = tf.reduce_sum(pi * c)
    return cost, pi


class marginal(object):
    def __init__(self, kernel_matrix, batch_size):
        self.mu = compute_kde(kernel_matrix, batch_size)
        self.kernel_matrix = kernel_matrix
        self.nu = tf.ones(batch_size) / batch_size
        self.batch_size = batch_size

    def update_mu(self, new_nu):
        self.nu = tf.constant(new_nu)
        # self.mu = compute_kde(self.kernel_matrix, self.batch_size)#/tf.stop_gradient(compute_kde(kernel_matrix, batch_size)) * tf.stop_gradient(self.nu)
        # self.mu /= tf.stop_gradient(tf.reduce_sum(self.mu))
        # self.nu /= tf.stop_gradient(tf.reduce_sum(self.nu))

    def reset_mu(self):
        # self.mu = compute_kde(self.kernel_matrix, self.batch_size)
        self.nu = tf.ones(self.batch_size) / self.batch_size
