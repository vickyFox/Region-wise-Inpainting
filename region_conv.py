import tensorflow as tf
from inpaint_ops import *
import cv2 as cv
import numpy as np
from inpaint_ops import *
def region_deconv(x_complete, x_missing, mask,name = 'com_'):
    shape = x_complete.get_shape().as_list()
    mask = tf.image.resize_nearest_neighbor(mask, size = [shape[1], shape[2]], align_corners=True)
    mask = tf.reshape(mask, [shape[0], shape[1], shape[2], 1])
    x_complete = x_complete * mask
    shape = x_missing.get_shape().as_list()
    x_missing  = x_missing  * (1 - mask)
    x_fusion   = tf.concat([x_complete,x_missing], axis = -1)

    x_fusion   = standard_dconv(x_fusion,mask,shape[-1],name = name+"_fusion")
    return x_fusion

def region_conv(x_complete, x_missing, mask,name = 'com_'):
    shape = x_complete.get_shape().as_list()
    mask = tf.image.resize_nearest_neighbor(mask, size = [shape[1], shape[2]], align_corners=True)
    mask = tf.reshape(mask, [shape[0], shape[1], shape[2], 1])
    p3 = int(1 * (3 - 1) / 2)

    x_complete = x_complete * mask
    shape = x_missing.get_shape().as_list()
    x_missing  = x_missing  * (1 - mask)
    x_fusion   = tf.concat([x_complete,x_missing], axis = -1)
    x_fusion   = tf.pad(x_fusion, [[0,0], [p3,p3], [p3,p3], [0,0]], 'REFLECT')

    x_fusion   = tf.layers.conv2d(x_fusion, shape[-1], kernel_size = 3, strides = 1, padding = 'VALID', dilation_rate = 1, name = name + "_fusion")
    x_fusion   = tf.nn.elu(x_fusion)
    return x_fusion