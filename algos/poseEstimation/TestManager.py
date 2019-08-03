# import argparse

import os
import requests

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import cv2
import numpy as np
from tensorflow.contrib import slim

import algos.poseEstimation.vgg as vgg
from algos.poseEstimation.cpm import PafNet
# import common
from algos.poseEstimation.tensblur.smoother import Smoother
from algos.poseEstimation.estimator import PoseEstimator, TfPoseEstimator


def init(InCheckPointPath='checkpoints/train/', vgg19_path='checkpoints/vgg/vgg_19.ckpt', use_bn=False):
    # tf.logging.set_verbosity(tf.logging.WARN)

    checkpoint_path = InCheckPointPath
    if os.path.isfile(checkpoint_path + 'model-59000.ckpt.data-00000-of-00001') is False:
        print('Downloading checkpoints .. ')
        download_file(
            'http://download2263.mediafire.com/7tq403a7wdng/fs9ag3b1bdihjtd/model-59000.ckpt.data-00000-of-00001',
            InCheckPointPath + 'model-59000.ckpt.data-00000-of-00001',
            FileSize=741)

    backbone_net_ckpt_path = vgg19_path
    if os.path.isfile(backbone_net_ckpt_path) is False:
        print('Downloading vgg weights .. ')
        download_file('http://download2266.mediafire.com/aqy5u9s0t71g/y93ud1n21401ed8/vgg_19.ckpt',
                      backbone_net_ckpt_path, FileSize=548)
    # logger.info('checkpoint_path: ' + checkpoint_path)

    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        img_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='original_image_size')

    img_normalized = raw_img / 255 - 0.5

    # define vgg19
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    # get net graph
    # logger.info('initializing model...')
    net = PafNet(inputs_x=vgg_outputs, use_bn=use_bn)
    hm_pre, cpm_pre, added_layers_out = net.gen_net()

    hm_up = tf.image.resize_area(hm_pre[5], img_size)
    cpm_up = tf.image.resize_area(cpm_pre[5], img_size)
    # hm_up = hm_pre[5]
    # cpm_up = cpm_pre[5]
    smoother = Smoother({'data': hm_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                            tf.zeros_like(gaussian_heatMat))

    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    trainable_var_list = trainable_var_list + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')

    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19'), name='vgg_restorer')
    saver = tf.train.Saver(trainable_var_list)
    # logger.info('initialize session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start tf session
    sess = tf.Session(config=config)
    sess.run(tf.group(tf.global_variables_initializer()))
    restorer.restore(sess, vgg19_path)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))

    return [sess, tensor_peaks, hm_up, cpm_up, raw_img, img_size]


def processFrame(InFrame, InTensorflowSession):
    predict = InTensorflowSession[0]
    tensor_peaks = InTensorflowSession[1]
    hm_up = InTensorflowSession[2]
    cpm_up = InTensorflowSession[3]
    raw_img = InTensorflowSession[4]
    img_size = InTensorflowSession[5]

    ori_w = InFrame.shape[1]
    ori_h = InFrame.shape[0]

    size = [int(654 * (ori_h / ori_w)), 654]
    h = int(654 * (ori_h / ori_w))

    img = np.array(cv2.resize(InFrame, (654, h)))

    # img_corner = np.array(cv2.resize(image, (360, int(360 * (ori_h / ori_w)))))
    img = img[np.newaxis, :]
    peaks, heatmap, vectormap = predict.run([tensor_peaks, hm_up, cpm_up],
                                            feed_dict={raw_img: img, img_size: size})
    bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
    InFrame = TfPoseEstimator.draw_humans(InFrame, bodys, imgcopy=False)

    # image = common.read_imgfile(InFrame)
    # image = InFrame
    # size = [image.shape[0], image.shape[1]]
    # if image is None:
    #     print('Image can not be read, path=%s' % InFrame)
    #     # logger.error('Image can not be read, path=%s' % InFrame)
    #     sys.exit(-1)
    # h = int(654 * (size[0] / size[1]))
    # img = np.array(cv2.resize(image, (654, h)))
    # # cv2.imwrite('/media/ramdisk/img.png', img)
    # # cv2.imshow('ini', img)
    # img = img[np.newaxis, :]
    # peaks, heatmap, vectormap = predict.run([tensor_peaks, hm_up, cpm_up],
    #                                                 feed_dict={raw_img: img, img_size: size})
    # cv2.imwrite('/media/ramdisk/vector.png', vectormap[0, :, :, 0])
    # # cv2.imshow('in', vectormap[0, :, :, 0])
    #
    # bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
    # image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
    # cv2.imshow(' ', image)
    cv2.imwrite('/media/ramdisk/output.png', InFrame)
    return InFrame


#


def download_file(url, fileName, FileSize):
    # local_filename =
    r = requests.get(url, stream=True)
    f = open(fileName, 'wb')
    i = 0
    for chunk in r.iter_content():
        if i % (1024 * 1024) == 0:
            print(f'Loading {(i // (1024 * 1024)) / 100} MB of {FileSize} MB')
        f.write(chunk)
        i += 100
    f.close()
    return
