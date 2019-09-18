#from tensorflow.python import pywrap_tensorflow
import math
import random
import gc
import os
import cv2
import glob
import socket
import tensorflow as tf
from inpaint_model import *
from mask_online import *
import argparse

if __name__=='__main__':
        parser = argparse.ArgumentParser(description='training code')
        parser.add_argument('--train_data_path',type=str ,default="" ,help='training data path')
        parser.add_argument('--epoch',type=int ,default=20 ,help='training epoch')
        parser.add_argument('--batch_size',type=int ,default=8 ,help='batch_size')
        parser.add_argument('--width',type=int ,default=256 ,help='images width')
        parser.add_argument('--height',type=int ,default=256 ,help='images height')
        parser.add_argument('--mask_area',type=int ,default=60 ,help='mask_area')
        parser.add_argument('--learning_rate',type=float ,default=0.0001 ,help='training epoch')
        parser.add_argument('--beta1',type=float ,default=0.5 ,help='beta1')
        parser.add_argument('--beta2',type=float ,default=0.9 ,help='beta2')
        parser.add_argument('--lambda_style',type=float ,default=0.001 ,help='weight of style_loss')
        parser.add_argument('--lambda_cor',type=int ,default=0.00001 ,help='weight of correlation loss')
        parser.add_argument('--lambda_adv',type=int ,default=1.0 ,help='weight of adversial loss')
        parser.add_argument('--alpha',type=int ,default=0.001 ,help='weight of penalization of discriminator for ground-truth images')
        parser.add_argument('--stage',type=int ,default=0 ,help='training stage')
        parser.add_argument('--mask_type',type=int ,default=1 ,help='0: discontinuous mask 1: continuous mask')
        parser.add_argument('--adv_type',type=str ,default=' ' ,help='type of adversial loss:wgan, gan, hinge')
        parser.add_argument('--vgg_path',type=str ,default='./vgg/vgg16.npy' ,help='path of vgg16')
        parser.add_argument('--pretrained_model',type=str ,default='./pretrained_model/v19.ckpt' ,help='pretrained_model path')
        parser.add_argument('--output',type=str ,default='./output/' ,help='path to save the model and summary')
        args = parser.parse_args()


        batch_size = args.batch_size
        fnames     = glob.glob(args.train_data_path + '\\*.png')
        filename_queue = tf.train.string_input_producer(fnames, shuffle = True)
        reader = tf.WholeFileReader()
        _,img_bytes = reader.read(filename_queue)
        images = tf.image.decode_jpeg(img_bytes, channels = 3)
        images = tf.image.resize_images(images,[args.height, args.width])
        images = tf.train.batch([images],batch_size, dynamic_pad = True)
        mask   = tf.placeholder(tf.float32,[batch_size, args.height, args.width, 1], name = 'mask')

        sess = tf.Session()
        if args.stage == 0:
            g_vars, _, adv_g_loss, _,rec_loss, correlation_loss, style_loss = build_graph_with_loss(images, batch_size, mask, args.vgg_path, args.adv_type
        		                                                                                      , args.stage,args.lambda_style, args.lambda_cor
                                                                                                      , args.alpha, args.lambda_adv)
        else:
            g_vars, d_vars, adv_g_loss, adv_d_loss, rec_loss, correlation_loss, style_loss = build_graph_with_loss(images, batch_size, mask, args.vgg_path, args.adv_type
        		                                                                                      , args.stage,args.lambda_style, args.lambda_cor
                                                                                                      , args.alpha , args.lambda_adv)
            d_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1 = 0.5, beta2 = 0.9).minimize(adv_d_loss, var_list = d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1 = 0.5, beta2 = 0.9).minimize(adv_g_loss, var_list = g_vars)
        
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()) # tf.initialize_all_variables()
        sess.run(init_op)
        if args.stage == 1:
            saver_pre = tf.train.Saver(g_vars)
            saver_pre.restore(sess, args.pretrained_model)
        
        tf.train.start_queue_runners(sess)
        summarywriter = tf.summary.FileWriter(args.output + '/summary', tf.get_default_graph())
        merge = tf.summary.merge_all()
        Iters = int(len(fnames)/batch_size)
        saver = tf.train.Saver(max_to_keep=20)
        low   = 0
        it    = 0
        high=args.height
        num=args.mask_area

        for j in range(0,20):
                for i in range(Iters):
                    if args.mask_type == 0:
                        mask_ = np.stack([discontinuous_mask(args.height,args.width,num,low,high) for _ in range(batch_size)], axis=0)  
                    else:
                        mask_ = np.stack([continuous_mask(args.width,args.height,num,360,32,50) for _ in range(batch_size)], axis=0)
                        
                    if i%100==0:
                            summary=sess.run(merge,feed_dict={mask:mask_})
                            summarywriter.add_summary(summary,it)
                    if args.stage == 0:
                    	_,g,rec,closs,sloss=sess.run([g_optimizer,adv_g_loss,rec_loss,correlation_loss,style_loss],feed_dict={mask:mask_,})
                    else:
                        _,_,g,d,rec,closs,sloss=sess.run([g_optimizer,d_optimizer,adv_g_loss,adv_d_loss,rec_loss,correlation_loss,style_loss],feed_dict={mask:mask_,})
                    if i % 20==0:
                        if args.stage == 0:
                            print('[{}/{}]rec_loss: {} correlation_loss:{}  style_loss: {}'.format(i,Iters,rec,closs,sloss))
                        else:
                            print('[{}/{}]g_loss: {}  rec_loss: {} correlation_loss:{}  style_loss: {} d_loss: {}'.format(i,Iters,g,rec,closs,sloss,d))
                            
                    it += 1
                saver.save(sess, args.output + '/model/v{}.ckpt'.format(j), global_step=Iters)
