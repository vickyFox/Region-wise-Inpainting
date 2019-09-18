# import matplotlib.pyplot as plt
import random
from PIL import Image
import gc
import os
import cv2
import glob
import socket
import tensorflow as tf
from inpaint_model import *
from mask_online import *
import argparse
def infer(batch_data,mask,reuse=False):
        shape=batch_data.get_shape().as_list()
        batch_gt=batch_data/127.5-1.
        batch_incomplete=batch_gt*mask

        image_p1, image_p2=RW_generator(batch_incomplete,mask,reuse=reuse)

        image_c2=batch_incomplete*mask+ image_p2*(1.-mask)
        image_c2=(image_c2+1.)*127.5
        return image_c2

if __name__=='__main__':
        parser = argparse.ArgumentParser(description='training code')
        parser.add_argument('--test_data_path',type=str ,default="C:\\Users\\bsh\\Desktop\\github\\origin_img00027222.png" ,help='test_data_path')
        parser.add_argument('--mask_path',type=str ,default="C:\\Users\\bsh\\Desktop\\github\\img00027222_mask.png" ,help='mask_path')
        parser.add_argument('--model_path',type=str ,default="..\\celeba\\face\\v8.ckpt-3500" ,help='model_path')
        parser.add_argument('--file_out',type=str ,default="./result" ,help='result_path')
        parser.add_argument('--width',type=int ,default=256 ,help='images width')
        parser.add_argument('--height',type=int ,default=256 ,help='images height')
        args = parser.parse_args()
        file_test=args.test_data_path
        file_mask=args.mask_path
        
        images=tf.placeholder(tf.float32,[1,args.height,args.width,3],name = 'image')
        mask=tf.placeholder(tf.float32,[1,args.height,args.width,1],name='mask')
        sess = tf.Session()        
        inpainting_result=infer(images,mask)
        saver_pre=tf.train.Saver()
        init_op = tf.group(tf.initialize_all_variables(),tf.initialize_local_variables()) 
        sess.run(init_op)
        saver_pre.restore(sess,args.model_path)
        
        test_mask = cv2.resize(cv2.imread(file_mask),(args.height,args.width))
        test_mask = test_mask[:,:,0:1]
        test_mask = 0. + test_mask//255
        test_mask[test_mask >= 0.5] = 1
        test_mask[test_mask <  0.5] = 0
        test_mask  = 1 -test_mask
        test_image = cv2.imread(file_test)[...,::-1]
        test_image = cv2.resize(test_image, (args.height, args.width))
        test_mask = np.expand_dims(test_mask,0)
        test_image = np.expand_dims(test_image,0)
        img_out=sess.run(inpainting_result,feed_dict={mask:test_mask,images:test_image})

        cv2.imwrite(args.file_out+"/big.png", img_out[0][...,::-1])
        cv2.imwrite(args.file_out+"/big_00028739.png.png", test_image[0][...,::-1] * test_mask[0] + 255 * (1 - test_mask[0]))

