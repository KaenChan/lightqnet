"""Main implementation class of PFE
"""
# MIT License
#
# Copyright (c) 2021 Kaen Chan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from abc import ABC

import numpy as np
import cv2
from time import time
import sys

import os
import tensorflow as tf
from numpy.linalg import norm

from common import load_model

TF_FEAT_MODEL_PATH = 'lightqnet-dm050.pb'
TF_FEAT_IMG_W = 96
TF_FEAT_IMG_H = 96


# @singleton
class TfFaceQaulityModel(ABC):
    def __init__(self, model_data_path=TF_FEAT_MODEL_PATH, image_w=TF_FEAT_IMG_W, image_h=TF_FEAT_IMG_H):
        self.image_w = image_w
        self.image_h = image_h
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        start_time = time()
        with self.graph.as_default():
            load_model(model_data_path)
            gd = self.sess.graph.as_graph_def()
            self.image_ph = tf.get_default_graph().get_tensor_by_name("input:0")
            self.score = tf.get_default_graph().get_tensor_by_name("confidence_st:0")
            self.model_data_path = model_data_path
        end_time = time()
        print('load model use {:.2f}ms'.format((end_time - start_time) * 1000))

    def inference(self, img, to_rgb=True, verbose=0):
        """extract the features from the given bgr_images using cnn."""
        grayscale = self.image_ph.get_shape()[3] == 1
        img = cv2.resize(img, (self.image_w, self.image_h))
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = (img - 128.) /128.
        images = np.array([img])
        start_time = time()
        feed_dict = {self.image_ph: images}
        qscore = self.sess.run(self.score, feed_dict=feed_dict)
        print('qscore', qscore)
        end_time = time()
        if verbose < 0:
            print('Inference {:.2f} ms'.format((end_time - start_time) * 1000))
        return qscore[0][0]


if __name__ == '__main__':
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        help='image path',
                        default='images/Aaron_Eckhart_0001.jpg')
    parser.add_argument('--batch_size',
                        type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--num_batches',
                        type=int, default=10,
                        help='Number of batches to run.')
    args = parser.parse_args()

    print('batch_size', str(args.batch_size))

    img = cv2.imread(args.image)
    face_qaulity_predictor = TfFaceQaulityModel()

    for _ in range(10):
        score = face_qaulity_predictor.inference(img)
    # print(feat2)
    # exit()

    t1 = time()
    iters = 10
    for i in range(iters):
        img2 = img + i*np.random.random(img.shape)*10
        img2 = img2.astype(np.uint8)
        score = face_qaulity_predictor.inference(img2)
        # cv2.imshow('img', img2)
        # cv2.waitKey(1)
        print(score)
    print('average %.2f ms' % ((time() - t1)*1000./iters))
    print('----------------')

    # print(feat2.shape)
    # print(','.join([str(l) for l in feat2.ravel()]))
