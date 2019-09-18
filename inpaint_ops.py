import tensorflow as tf
import cv2 as cv
import random
import os
import math
import numpy as np
from tensorflow.contrib.framework.python.ops import add_arg_scope

def random_interpolates(x, y, alpha=None):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def gradients_penalty(x, y, mask=None, norm=1.):
    """Improved Training of Wasserstein GANs
    - https://arxiv.org/abs/1704.00028
    """
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))


def standard_conv(x,mask,cnum,ksize=3,stride=1,rate=1,name='conv',padding='SAME'):
        '''
        define convolution for generator
        Args:
                x:iput image
                cnum: channel number
                ksize: kernal size
                stride: convolution stride
                rate : rate for dilated conv
                name: name of layers
        '''
        p       = int(rate*(ksize-1)/2)
        x       = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]],'REFLECT')
        padding = 'VALID'
        x       = tf.layers.conv2d(x,cnum,ksize,stride,dilation_rate=rate,activation=tf.nn.elu
                                  ,padding=padding,name=name+'_1')
        return x
def standard_dconv(x,mask,cnum,name='deconv',padding='VALID'):
        '''
        define upsample convolution for generator
        Args:
        x: input image
        mask: input mask
        name: name of layers
        '''
        rate   = 1
        ksize  = 3
        stride = 1
        shape  = x.get_shape().as_list()
        x      = tf.image.resize_nearest_neighbor(x,[shape[1]*2,shape[2]*2])
        p3     = int(1 * (3 - 1) / 2)
        x      = tf.pad(x, [[0,0], [p3,p3], [p3,p3], [0,0]], 'REFLECT')
        x      = tf.layers.conv2d(x,cnum,ksize,stride,dilation_rate=rate,activation=tf.nn.elu
                                          ,padding=padding,name=name+'_1')
        return x


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1,name='dasd'):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable(name+"u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = l2_norm(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = l2_norm(u_)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
   w_norm = w / sigma

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm


def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def dis_conv(x, cnum, ksize=5, stride=2, activation = 'leak_relu', name='conv'):
    """
    covolution for discriminator.
    Args:
        x: input image
        cnum: channel number.
        ksize: kernel size.
        Stride: convolution stride.
        name: name of layers.
    """

    x_shape = x.get_shape().as_list()
    w = tf.get_variable(name = name+'_w',shape = [ksize, ksize, x_shape[-1]] + [cnum])
    w = spectral_norm(w, name = name)
    x = tf.nn.conv2d(x, w, strides = [1, stride, stride, 1], padding = 'SAME')
    bias = tf.get_variable(name=name+'_bias',shape=[cnum])
    if activation != None:
        return  LeakyRelu(x + bias, name = name)
    else:
        return x + bias


