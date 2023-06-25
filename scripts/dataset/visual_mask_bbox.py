'''
Description: 
Author: wangdx
Date: 2021-11-13 09:38:49
LastEditTime: 2022-01-18 09:27:34
'''
"""
Visualize mask and bbox labels 
"""

import cv2
import os
import glob
import scipy.io as scio
import numpy as np
import math
import random


def bbox_mask(path):
    im_camera_mask = scio.loadmat(path + '/camera_mask.mat')['A'].astype(np.uint8)  # 相机mask
    im_camera_rgb = cv2.imread(path + '/camera_rgb.png')

    # 根据mask值设置彩色
    im_camera_visual = np.zeros((im_camera_mask.shape[0], im_camera_mask.shape[1], 3)).astype(np.uint8)
    mask_id = np.unique(im_camera_mask)
    for idx in mask_id[2:]:
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        im_camera_visual[im_camera_mask==idx] = color

        # 绘制bbox
        mask = np.zeros_like(im_camera_mask)
        mask[im_camera_mask==idx] = 255
        x, y, w, h = cv2.boundingRect(mask)
        cv2.rectangle(im_camera_rgb, (x,y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow('bbox', im_camera_rgb)
    cv2.imshow('im_camera_visual', im_camera_visual)
    cv2.waitKey()


if __name__ == "__main__":
    dataset_path = 'D:/research/grasp_detection/sim_grasp/openSource/dataset'
    for sample in os.listdir(dataset_path):
        print('sample =', sample)
        sample_path = os.path.join(dataset_path, sample)
        bbox_mask(sample_path)
