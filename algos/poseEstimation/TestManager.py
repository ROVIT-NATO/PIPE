import warnings

warnings.filterwarnings('ignore')

import argparse
import tensorflow as tf
import sys
import time
import logging
import cv2
import numpy as np
from tensorflow.contrib import slim
import vgg
from cpm import PafNet
import common
from tensblur.smoother import Smoother
from estimator import PoseEstimator, TfPoseEstimator


# logger = logging.getLogger('run')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)


def init(InCheckPointPath='checkpoints/train/', vgg19_path='checkpoints/vgg/vgg_19.ckpt', use_bn=False):
    tf.logging.set_verbosity(tf.logging.WARN)

    checkpoint_path = InCheckPointPath
    backbone_net_ckpt_path = vgg19_path
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


def processFrame(InFrame, IntfSession):
    predict = IntfSession[0]
    tensor_peaks = IntfSession[1]
    hm_up = IntfSession[2]
    cpm_up = IntfSession[3]
    raw_img = IntfSession[4]
    img_size = IntfSession[5]

    image = common.read_imgfile(InFrame)
    size = [image.shape[0], image.shape[1]]
    if image is None:
        print('Image can not be read, path=%s' % InFrame)
        # logger.error('Image can not be read, path=%s' % InFrame)
        sys.exit(-1)
    h = int(654 * (size[0] / size[1]))
    img = np.array(cv2.resize(image, (654, h)))
    # cv2.imwrite('/media/ramdisk/img.png', img)
    # cv2.imshow('ini', img)
    img = img[np.newaxis, :]
    peaks, heatmap, vectormap = predict.run([tensor_peaks, hm_up, cpm_up],
                                                    feed_dict={raw_img: img, img_size: size})
    cv2.imwrite('/media/ramdisk/vector.png', vectormap[0, :, :, 0])
    # cv2.imshow('in', vectormap[0, :, :, 0])

    bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
    image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
    # cv2.imshow(' ', image)
    cv2.imwrite('/media/ramdisk/image.png', image)


if __name__ == '__main__':
    session = init(InCheckPointPath='checkpoints/train/',
                   vgg19_path='checkpoints/vgg/vgg_19.ckpt')
    image = '/ocean/anish/Developer/RnD/poseEstimation/deep-high-resolution-net.pytorch/data/mpii/images/000916555.jpg'
    processFrame(image, session)
    session[0].close()