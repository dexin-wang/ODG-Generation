'''
Description: 
Author: wangdx
Date: 2021-11-18 15:49:00
LastEditTime: 2022-01-07 13:19:50
'''
import torch
import numpy as np
from skimage.filters import gaussian
import sys
from scripts.dataset.generate_graspFig import GRASP_MAX_W


def post_process_output(able_pred, angle_pred, width_pred):
    """
    :param able_pred:  (1, 1, h, w)           (as torch Tensors)
    :param angle_pred: (1, angle_k, h, w)     (as torch Tensors)
    :param width_pred: (1, angle_k, h, w)     (as torch Tensors)
    """

    # 抓取置信度
    able_pred = able_pred.squeeze().cpu().numpy()    # (h, w)
    able_pred = gaussian(able_pred, 1.0, preserve_range=True)

    # 抓取角
    angle_pred = np.argmax(angle_pred.cpu().numpy().squeeze(), 0)   # (h, w)    每个元素表示预测的抓取角类别

    # 根据抓取角类别获取抓取宽度
    size = angle_pred.shape[0]
    cols = np.arange(size)[np.newaxis, :]
    cols = cols.repeat(size, axis=0)
    rows = cols.T
    width_pred = width_pred.squeeze().cpu().numpy() * GRASP_MAX_W  # (angle_k, h, w)  实际长度
    width_pred = width_pred[angle_pred, rows, cols] # (h, w)

    return able_pred, angle_pred, width_pred


def post_process_output_1(able_pred, angle_pred, width_pred):
    """
    angle_pred的置信度乘以该位置处的able_pred
    """
    # 抓取角
    angle_pred = angle_pred.cpu().numpy().squeeze()   # (angle, h, w)    每个元素表示预测的抓取角类别
    # 抓取置信度
    able_pred = able_pred.cpu().numpy().squeeze()    # (h, w)
    able_pred = gaussian(able_pred, 1.0, preserve_range=True)
    able_pred = np.expand_dims(able_pred, 0).repeat(angle_pred.shape[0], 0) # (angle, h, w)
    angle_pred = angle_pred * able_pred
    # 抓取宽度
    width_pred = width_pred.cpu().numpy().squeeze() * GRASP_MAX_W  # (angle_k, h, w)  实际长度

    return angle_pred, width_pred



