from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
from time import time, sleep
import numpy as np
from common import scan_image_tree
from face_normalizer import FaceNormalizer
from tf_face_detection_mtcnn import TfFaceDetectorMtcnn
from tf_face_quality_model import TfFaceQaulityModel


def read_image(path):
    image_origin = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # print(image_origin.shape[0])
    return image_origin


def main():
    image_dir = 'images'
    output_dir = 'images_out'

    face_detector = TfFaceDetectorMtcnn()
    face_normalizer = FaceNormalizer()
    face_qaulity_predictor = TfFaceQaulityModel()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_path_list = scan_image_tree(image_dir)
    for i in range(len(img_path_list)):
        print('---------- ' + str(i) + '/' + str(len(img_path_list)) + '----------')
        img_path = img_path_list[i]
        print(img_path)
        img = read_image(img_path)
        # frame1 = cv2.resize(frame1, (frame1.shape[1]*2, frame1.shape[0]*2))

        bboxes, points = face_detector.get_face_bboxes(img, detect_scale=0.3, verbose=0)
        if len(bboxes) == 0:
            continue
        img_align_list = face_normalizer.get_face_aligned(img, points)

        for img_align in img_align_list:
            qaulity_score = face_qaulity_predictor.inference(img_align)
            name = '{}/{:.4f}_{}'.format(output_dir, qaulity_score, os.path.split(img_path)[-1])
            cv2.imwrite(name, img_align)


if __name__ == '__main__':
    main()

