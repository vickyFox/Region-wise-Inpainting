import tensorflow as tf
from inpaint_ops import *
import cv2 as cv
import numpy as np
from vgg.vgg16 import *
from region_conv import *
def RW_generator(x,mask,padding='SAME',name='inpaint_net',reuse=False):
    '''
    Region-wise generator
    Args:
            x: incomplete image
            mask: mask region {0,1}
    returns:
            predicted image
    '''
    x1 = SINet(x, mask, reuse=reuse)
    x_combine = x * mask + x1 * (1 - mask)
    x2 = GPNet(x_combine,mask, reuse=reuse)
    return x1, x2

def SINet(x,mask,padding='SAME',name='inpaint_net',reuse=False):
    '''
    Semantic inferring network
    Args:
            x: incomplete image
            mask: mask region {0,1}
    returns:
            image predicted by semantic inferring network
    '''
    xin=x;
    mask_in=mask
    cnum=32
    ones_x = tf.ones_like(x)[:, :, :, 0:1]  #一层
    x = tf.concat([x, mask], axis=3) #拼接
    with tf.variable_scope(name,reuse=reuse):
            x1=standard_conv(x,mask,cnum,5,1,name='conv1')
            x2=standard_conv(x1,mask,2*cnum,3,2,name='conv2_downsample')
            x3=standard_conv(x2,mask,2*cnum,3,1,name='conv3')
            x4=standard_conv(x3,mask,4*cnum,3,2,name='conv4_downsample')
            x5=standard_conv(x4,mask,4*cnum,3,1,name='conv5')
            x6=standard_conv(x5,mask,4*cnum,3,1,name='conv6')
           
            #dilated conv
            x7=standard_conv(x6,mask,4*cnum,3,rate=2,name='conv7_atrous')
            x8=standard_conv(x7,mask,4*cnum,3,rate=4,name='conv8_atrous')
            x9=standard_conv(x8,mask,4*cnum,3,rate=8,name='conv9_atrous')
            x10=standard_conv(x9,mask,4*cnum,3,rate=16,name='conv10_atrous')
       


            x11=standard_conv(tf.concat([x10,x6],axis=-1), mask,4*cnum,3,1,name='conv11')
            x12=standard_conv(tf.concat([x11,x5],axis=-1),mask,4*cnum,3,1,name='conv12')
            
            x_complete, x_missing = tf.concat([x12,x4],axis=-1),x12                      
            x13 = region_deconv(x_complete, x_missing, mask,name = 'com_13')

            x_complete, x_missing = tf.concat([x13,x3],axis=-1),x13 
            x14 = region_conv(x_complete,x_missing,mask,name='com_14')                     
            
            x_complete, x_missing = tf.concat([x14,x2],axis=-1),x14
            x15 = region_deconv(x_complete, x_missing, mask, name = 'com_15')

            x16 = standard_conv(x15,mask,cnum,3,1,name='conv16')

            x17=standard_conv(x16,mask,cnum//2,3,1,name='conv17')
            x18=standard_conv(x17,mask,3,3,1,name='conv18')
            x18=tf.clip_by_value(x18,-1.,1.)
            
            return x18
def GPNet(x,mask,padding='SAME',name='inpaint_net_1',reuse=False):
    '''
    Global perceiving network
    Args:
            x: incomplete image
            mask: mask region {0,1}
    returns:
            image predicted by global perceiving network
    '''
    xin=x;
    mask_in=mask
    cnum=32
    x = tf.concat([x, mask], axis=3) #concat
    with tf.variable_scope(name,reuse=reuse):
            x1=standard_conv(x,mask,cnum,5,1,name='conv1')
            x2=standard_conv(x1,mask,2*cnum,3,2,name='conv2_downsample')
            x3=standard_conv(x2,mask,2*cnum,3,1,name='conv3')
            x4=standard_conv(x3,mask,4*cnum,3,2,name='conv4_downsample')
            x5=standard_conv(x4,mask,4*cnum,3,1,name='conv5')
            x6=standard_conv(x5,mask,4*cnum,3,1,name='conv6')
           
            #dilated conv
            x7=standard_conv(x6,mask,4*cnum,3,rate=2,name='conv7_atrous')
            x8=standard_conv(x7,mask,4*cnum,3,rate=4,name='conv8_atrous')
            x9=standard_conv(x8,mask,4*cnum,3,rate=8,name='conv9_atrous')
            x10=standard_conv(x9,mask,4*cnum,3,rate=16,name='conv10_atrous')
       


            x11=standard_conv(tf.concat([x10,x6],axis=-1), mask,4*cnum,3,1,name='conv11')
            x12=standard_conv(tf.concat([x11,x5],axis=-1),mask,4*cnum,3,1,name='conv12')
            
            x13=standard_dconv(tf.concat([x12,x4],axis=-1),mask,2*cnum,name='conv13_upsample')
            x14=standard_conv(tf.concat([x13,x3],axis=-1),mask,2*cnum,3,1,name='conv14')
            x15=standard_dconv(tf.concat([x14,x2],axis=-1),mask,cnum,name='conv15_upsample')
            x16=standard_conv(tf.concat([x15,x1],axis=-1),mask,cnum//2,3,1,name='conv16')
            x17=standard_conv(x16,mask,3,3,1,name='conv17')
            x18=tf.clip_by_value(x17,-1.,1.)
            
            return x18

def RW_discriminator(x, mask, batch_size, activation = 'leaky_relu',reuse=False):
    '''
    Region-wise discriminator
    Args:
            x: input images
            mask: mask region {0,1}
    returns:
            matrix {real, fake}
    '''
    with tf.variable_scope('discriminator',reuse=reuse):
            cnum=64
            x=tf.concat([x,mask],axis=3)
            x=dis_conv(x,cnum,name='d_conv1')
            x=dis_conv(x,2*cnum,name='d_conv2')
            x=dis_conv(x,4*cnum,name='d_conv3')
            x=dis_conv(x,4*cnum,name='d_conv4')
            x=dis_conv(x,4*cnum,name='d_conv5')
            x=dis_conv(x,4*cnum,name='d_conv6', activation = activation)
            return x
def build_graph_with_loss(batch_data, batch_size, mask, vgg_path, adv_type, stage = 0, 
                         lambda_style = 0.001, lambda_cor = 0.00001, alpha = 0.01 ,lambda_adv = 1.0,
                         reuse=False, training = True):
 
    image_gt=tf.subtract(tf.divide(batch_data,127.5),1.)
    date_shape=batch_data.get_shape().as_list()   
    batch_incomplete=image_gt*mask 

    image_p1, image_p2 = RW_generator(batch_incomplete, mask)
    image_c1 = image_p1 * (1 - mask) + image_gt * mask
    image_c2 = image_p2 * (1 - mask) + image_gt * mask
    rec_loss = tf.reduce_sum(tf.abs(image_gt - image_p1)) + tf.reduce_sum(tf.abs(image_gt - image_p2))

    vgg = Vgg16(vgg_path)
    vgg.build(image_gt)
    vgg_pos = [vgg.pool1,vgg.pool2, vgg.pool3]
    vgg.build(image_c1)
    vgg_x1 =  [vgg.pool1,vgg.pool2, vgg.pool3]
    vgg.build(image_c2)
    vgg_x2 =  [vgg.pool1,vgg.pool2, vgg.pool3]

    cor_loss = loss_cor(vgg_x1, vgg_pos)
    style_loss= loss_style(vgg_x2, vgg_pos)
    cor_loss = cor_loss * lambda_cor
    style_loss = style_loss * lambda_style
    cor_style = cor_loss + style_loss


    if stage == 1:
        activation = None
        if adv_type == ' ':
            activation = 'leaky_relu'
        d_pred = RW_discriminator(image_c1 * (1 - mask), mask, batch_size, activation) 
        d_pred2 = RW_discriminator(image_c2 * (1 - mask), mask, batch_size, activation, reuse = True) 
        d_real = RW_discriminator(image_gt * (1 - mask), mask, batch_size, activation, reuse = True) 
        mask_label = 1 - mask
        shape = d_pred.get_shape().as_list()
        mask_label =  tf.image.resize_nearest_neighbor(mask_label, [shape[1],shape[2]])
        if adv_type == 'wgan_gp':
            penalty_img = random_interpolates(image_gt, image_c2)
            dout_penalty = RW_discriminator(penalty_img, mask, batch_size, activation,reuse = True)
            penalty_loss = gradients_penalty(penalty_img, dout_penalty, mask = mask)
            d_loss =  tf.reduce_mean(d_pred * mask_label) + tf.reduce_mean(d_pred2 * mask_label) - 0.01 * tf.reduce_mean(d_real * mask_label) +penalty_loss
            d_g_loss = -1 * tf.reduce_mean(d_pred * mask_label) - tf.reduce_mean(d_pred2 * mask_label)

        elif adv_type == 'gan':
            adv_d_loss = 0.01 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)) * mask_label) 
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_pred, labels=tf.zeros_like(d_pred)) * mask_label) 
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_pred2, labels=tf.zeros_like(d_pred2))* mask_label)  
            adv_g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=d_pred, labels=tf.ones_like(d_real)) * mask_label)  + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_pred2, labels=tf.ones_like(d_real)) * mask_label)

        elif adv_type == 'hinge':
            
            adv_d_loss =  0.01 * tf.reduce_mean(tf.nn.relu(1 - d_real) * mask_label) + tf.reduce_mean(tf.nn.relu(1 + d_pred) * mask_label) 
            + tf.reduce_mean(tf.nn.relu(1 + d_pred2) * mask_label) 
            adv_g_loss = -1 * tf.reduce_mean(tf.nn.relu(d_pred) * mask_label) - tf.reduce_mean(tf.nn.relu(d_pred2) * mask_label) 
        else:

            adv_d_loss =  alpha*tf.reduce_sum(tf.abs(mask_label - d_real)) + tf.reduce_sum(tf.abs(0 - d_pred)) + tf.reduce_sum(tf.abs(0 - d_pred2)) 
            adv_g_loss = tf.reduce_sum(tf.abs(mask_label - d_pred)) + tf.reduce_sum(tf.abs(mask_label - d_pred2)) 
    else:
        adv_d_loss = None
        adv_g_loss = None

    if stage == 1:
        g_loss= rec_loss+cor_style + lambda_adv * adv_g_loss
    else:
        g_loss = rec_loss + cor_style
    g_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'inpaint_net')
    d_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator') if stage == 1 else None

    tf.summary.image('incomplete',batch_incomplete, max_outputs=7)
    tf.summary.image('image_p1',image_p1, max_outputs=7)
    tf.summary.image('image_p2',image_p2, max_outputs=7)
    tf.summary.image('image_c2',image_c2, max_outputs=7)
    tf.summary.scalar('rec_loss',rec_loss)
    tf.summary.scalar('correlation loss', cor_loss)
    tf.summary.scalar('style loss', style_loss)
    if stage == 1:
        tf.summary.scalar('adv_g_loss', adv_g_loss)
        tf.summary.scalar('adv_d_loss', adv_d_loss)
    return g_vars,d_vars,g_loss,adv_d_loss,rec_loss,cor_loss,style_loss
