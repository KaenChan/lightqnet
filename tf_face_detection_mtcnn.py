#!/usr/bin/python

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time

import cv2
import numpy as np
import tensorflow as tf

from align import detect_face


# @singleton
from common import scan_image_tree


class TfFaceDetectorMtcnn(object):
    def __init__(self, joint_align=True):
        self.joint_align = joint_align

        self.minsize = 20
        self.threshold = [0.6, 0.9, 0.9]
        self.threshold = [0.6, 0.7, 0.9]
        self.threshold = [0.6, 0.7, 0.7]
        self.threshold = [0.6, 0.7, 0.8]
        self.factor = 0.6
        self.factor = 0.8
        self.factor = 0.709
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)

    def mtcnn_detect(self, bgr_img, detect_scale=1.0):
        self.minsize = 20 / detect_scale
        detect_scale = 1.
        im = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im_scale = cv2.resize(im, (int(im.shape[1] * detect_scale),
                                int(im.shape[0] * detect_scale)))
        bboxes, points = detect_face.detect_face(
            im_scale, self.minsize, self.pnet, self.rnet, self.onet,
            self.threshold, self.factor)

        if len(bboxes) == 0:
            bboxes = np.array([])
            points = np.array([])
        else:
            bboxes /= detect_scale
            w = bboxes[:,2] - bboxes[:,0]
            h = bboxes[:,3] - bboxes[:,1]
            bboxes[:,2] = w
            bboxes[:,3] = h
            points = np.array(points) / detect_scale

            sort_index = np.argsort(-bboxes[:, 2])
            bboxes = bboxes[sort_index]
            points = points[:, sort_index]

            points_out = []
            for point in points.T:
                points_out += [np.array([[point[i], point[i+5]]
                                         for i in range(5)])]
            points = np.array(points_out)
            # bboxes = bboxes.astype(int)

        return bboxes, points

    def get_face_bboxes(self, im, detect_scale=1., verbose=0):
        if type(im)==str:
            im = cv2.imread(im, cv2.IMREAD_COLOR)
        elif type(im)==np.ndarray:
            pass
        else:
            print(im)
            raise('error image input')

        t1 = time()
        bboxes, shapes = self.mtcnn_detect(im, detect_scale=detect_scale)
        if self.joint_align:
            if verbose>0:
                print('detection %.2f ms'%((time()-t1)*1000.))
            return np.array(bboxes), np.array(shapes)

        if verbose>0:
            print('detection %.2f ms'%((time()-t1)*1000.))

        return np.array(bboxes)


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        type=str,
                        help='image file, or the directory which contains images',
                        # default='./tmp/A.J._Cook'
                        default='images'
                        )
    parser.add_argument('-j', '--joint_align',
                        type=int,
                        help='[0|1] mtcnn joint detection and alignment',
                        default=0
                        )
    parser.add_argument('-s', '--detect_scale',
                        type=float,
                        default=0.3
                        )
    parser.add_argument('-v', '--verbose',
                        type=int,
                        help="0-2",
                        default=1
                        )
    args = parser.parse_args()
    file_input = args.input
    detect_scale = args.detect_scale
    joint_align = args.joint_align

    verbose = args.verbose

    print('input', file_input)
    print('joint_align', joint_align)
    print('detect_scale', detect_scale)
    print('verbose', verbose)

    face_detector = TfFaceDetectorMtcnn(joint_align=joint_align)

    if os.path.isdir(file_input):
        src_path = file_input
        if src_path[-1] not in ['/', '\\']:
            src_path += '/'
        img_list = scan_image_tree(src_path)
    else:
        if(file_input.endswith('.jpg') or file_input.endswith('.png') or \
                file_input.endswith('.bmp')):
            img_list = [file_input]
        else:
            img_list = []

    t1 = time()
    for img in img_list:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        print(img.shape)
        ret = face_detector.get_face_bboxes(img, detect_scale=detect_scale, verbose=verbose)


if __name__ == '__main__':
    main()

