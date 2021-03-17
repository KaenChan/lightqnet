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

import os
import cv2
import numpy as np
from time import time
import subprocess

import copy
import sys
from skimage import transform as trans


# @singleton
class FaceNormalizer(object):
    def __init__(self):
        pass

    def get_face_aligned(self, img, points=None, image_size=160):
        nrof_faces = len(points)
        # print(nrof_faces)
        if nrof_faces > 0 and len(points[0].shape) == 2:
            img_size = np.asarray(img.shape)[0:2]
            imgs_aligned = []
            for point in list(points):
                coord_point = np.array([[34.19, 46.16], [65.65, 45.98], [50.12, 82.40]], dtype=np.float32) # mxnet

                coord_point /= 100
                if point.shape[0] == 5:
                    mouth = point[3:, :].mean(axis=0)
                    cur_point = np.array([point[0], point[1], mouth],
                                         dtype=np.float32)
                elif point.shape[0] == 4:
                    cur_point = np.array([point[0], point[1], point[3]],
                                         dtype=np.float32)
                elif point.shape[0] == 68:
                    point = cvt_68pts_to_5pts(point)
                    mouth = point[3:, :].mean(axis=0)
                    cur_point = np.array([point[0], point[1], mouth],
                                         dtype=np.float32)

                # H = cv2.getAffineTransform(cur_point, image_size * coord_point)
                estimate_type = 'skimage'
                # estimate_type = 'cv2'
                if estimate_type == 'cv2':
                    H = cv2.getAffineTransform(cur_point, image_size * coord_point)
                elif estimate_type == 'skimage':
                    tform = trans.SimilarityTransform()
                    tform.estimate(cur_point, image_size * coord_point)
                    H = tform.params[0:2,:]
                else:
                    print('estimate_type error---->', estimate_type)
                    exit(0)

                imgs_aligned.append(cv2.warpAffine(img, H, (image_size, image_size)))
            return np.array(imgs_aligned)
        else:
            print('Unable to detect face')
            return None


def cvt_68pts_to_5pts(points68):
    points5 = np.array([
        np.mean(points68[36:42], axis=0),  # left eye
        np.mean(points68[42:48], axis=0),  # right eye
        np.mean(points68[30:35], axis=0),  # node
        points68[48],  # left mouth
        points68[54]]  # right mouth
    )
    return points5


if __name__ == '__main__':
    pass
