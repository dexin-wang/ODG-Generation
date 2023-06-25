# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：test-sgdn.py
@ IDE ：PyCharm
@ Function : sgdn测试类
"""

import cv2
import os
from numpy.lib.function_base import append
import torch
import time
import math
from skimage.feature import peak_local_max
import numpy as np
from grasp_methods.sgdn.models.ggcnn.common import post_process_output, post_process_output_1
from grasp_methods.sgdn.models.loss import get_pred
from grasp_methods.sgdn.models import get_network
from .utils.img import Image, RGBImage, DepthImage



def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    """
    绘制grasp
    img:    rgb图像
    grasps: list()	元素是 [row, col, angle, width]
    mode:   line or region
    """
    assert mode in ['line', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp[:4]

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255

        if mode == 'line':
            width = width / 2

            angle2 = calcAngle2(angle)
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            col = int(col)
            row = int(row)

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)

        else:
            img[row, col] = [color_b, color_g, color_r]

    return img

def drawRect(img, rect):
    """
    绘制矩形
    rect: [x1, y1, x2, y2]
    """
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


def depth2Gray(im_depth):
    """
    将深度图转至8位灰度图
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)


def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


def input_depth(img, input_size):
    """
    对图像进行修补、裁剪，保留中间的图像块
    img: 深度图像, np.ndarray (h, w)
    :return: 直接输入网络的tensor, 裁剪区域的左上角坐标
    """
    assert img.shape[0] >= input_size and img.shape[1] >= input_size, '输入的深度图必须大于等于{}*{}'.format(input_size, input_size)

    # img = img / 1000.0

    # 修补
    img = inpaint(img)

    # 裁剪中间的图像块
    crop_x1 = int((img.shape[1] - input_size) / 2)
    crop_y1 = int((img.shape[0] - input_size) / 2)
    crop_x2 = crop_x1 + input_size
    crop_y2 = crop_y1 + input_size
    im_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # print(im_crop.min(), im_crop.max())
    # cv2.imwrite('/home/wangdx/research/grasp_correction/sgdn/img/output/im_crop.png', depth2Gray(im_crop))

    # 归一化
    im_crop = np.clip((im_crop - im_crop.mean()), -1, 1)

    # 调整顺序，和网络输入一致
    im_crop = im_crop[np.newaxis, np.newaxis, :, :]     # (1, 1, h, w)
    im_tensor = torch.from_numpy(im_crop.astype(np.float32))  # np转tensor

    return im_tensor, crop_x1, crop_y1


def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    array: 二维array
    thresh: float阈值
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs



def similarity(grasp1, grasp2):
    """
    计算两个抓取的相似性
    grasp1: [confidence, row, col, angle, width, depth]
    相似的条件：
        (1) 抓取点距离不大于5像素
        (2) 抓取角之差不大于30°
        (3) 抓取宽度之差不大于1cm
    """
    pt_thresh = 10
    angle_thresh = 30.0 / 180 * math.pi
    width_thresh = 0.01

    _, row1, col1, angle1, width1, __ = grasp1
    _, row2, col2, angle2, width2, __ = grasp2
    if (row1 - row2) ** 2 + (col1 - col2) ** 2 > pt_thresh ** 2:
        return False
    if abs(angle1 - angle2) > angle_thresh\
        and abs(angle1 - angle2) < (math.pi-angle_thresh):
        return False
    if abs(width1 - width2) > width_thresh:
        return False
    return True

def NMS(grasps, out=50):
    """
    grasps: np.array, (n, 5) [confidence, row, col, angle, width, depth]
    """
    assert grasps.shape[0] > 0 

    ret = [ [int(grasps[0][1]), int(grasps[0][2]), grasps[0][3], grasps[0][4], grasps[0][5]] ]

    i = 0   # 上一个记录的抓取
    for j in range(1, grasps.shape[0]):
        if not similarity(grasps[i], grasps[j]) or (grasps.shape[0] - j + len(ret) == out):   # 相似或剩下的刚好够out个
            ret.append([int(grasps[j][1]), int(grasps[j][2]), grasps[j][3], grasps[j][4], grasps[j][5]])
            i = j
        if len(ret) == 50:
            break

    assert len(ret) == 50
    return ret



class SGDN:
    def __init__(self, net:str, use_rgb, use_dep, model, device, angle_cls):
        """
        net: 网络架构 'ggcnn2', 'deeplabv3', 'grcnn', 'unet', 'segnet', 'stdc', 'danet'
        input_channels: 输入通道数 1/3/4
        model: 训练好的模型路径
        device: cpu/cuda:0
        """
        self.use_rgb = use_rgb
        self.use_dep = use_dep
        self.angle_cls = angle_cls

        input_channels = int(use_rgb)*3+int(use_dep)

        print('>> loading SGDN')
        sgdn = get_network(net)
        self.net = sgdn(input_channels=input_channels, angle_cls=self.angle_cls)
        self.device = torch.device(device)
        # 加载模型
        self.net.load_state_dict(torch.load(model, map_location=self.device), strict=True)
        self.net.to(self.device)
        print('>> load done')

    @staticmethod
    def numpy_to_torch(s):
        """
        numpy转tensor
        """
        if len(s.shape) == 2:
            return torch.from_numpy(s[np.newaxis, np.newaxis, :, :].astype(np.float32))
        elif len(s.shape) == 3:
            return torch.from_numpy(s[np.newaxis, :, :, :].astype(np.float32))
        else:
            raise np.AxisError

    def predict(self, img_rgb, img_dep, input_size, mode, thresh=0.7, peak_dist=1):
        """
        预测抓取模型
        img_rgb: rgb图像 np.array (h, w, 3)
        img_dep: 深度图 np.array (h, w)
        input_size: 图像送入神经网络时的尺寸，需要在原图上裁剪
        mode: max, peak, all, nms
        thresh: 置信度阈值
        peak_dist: 置信度筛选峰值
        :return:
            pred_grasps: list([row, col, angle, width])  width单位为米
        """
        # 获取输入tensor
        image = Image()
        if self.use_rgb:
            im_rgb = RGBImage(img_rgb)
            image.rgbimg = im_rgb
        if self.use_dep:
            im_dep = DepthImage(img_dep)
            image.depthimg = im_dep
        self.crop_x1, self.crop_y1, _, _ = image.crop(input_size)
        input = self.numpy_to_torch(image.nomalise())    # (n, h, w) / (h, w)

        # 预测
        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, self.width_out = get_pred(self.net, input.to(self.device))

            if mode == 'nms':
                angle_pred, width_pred = post_process_output_1(self.able_out, self.angle_out, self.width_out)
                # 获取所有大于thresh**2的抓取，再NMS
                while True:
                    res = np.where(angle_pred > thresh**2)
                    if res[0].shape[0] > 100:
                        break
                    else:
                        thresh -= 0.1
                grasps = None
                for i in range(res[0].shape[0]):
                    _bin, _row, _col = res[0][i], res[1][i], res[2][i]
                    _grasp = np.array([[angle_pred[_bin, _row, _col], _row+self.crop_y1, _col+self.crop_x1, _bin/self.angle_cls*np.pi, width_pred[_bin, _row, _col], -1]], dtype=np.float64)
                    if grasps is None:
                        grasps = _grasp
                    else:
                        grasps = np.concatenate((grasps, _grasp), axis=0)   # shape = (n, 5)

                # 按照置信度从大到小排序
                grasps = grasps[np.argsort(grasps[:,0])[::-1]]
                return NMS(grasps)
            
            able_pred, angle_pred, width_pred = post_process_output(self.able_out, self.angle_out, self.width_out)

            if mode == 'peak':
                # 置信度峰值 抓取点
                pred_pts = peak_local_max(able_pred, min_distance=peak_dist, threshold_abs=thresh)
            elif mode == 'all':
                # 超过阈值的所有抓取点
                pred_pts = arg_thresh(able_pred, thresh=thresh)
            elif mode == 'max':
                # 置信度最大的点
                loc = np.argmax(able_pred)
                row = loc // able_pred.shape[0]
                col = loc % able_pred.shape[0]
                pred_pts = np.array([[row, col]])
            else:
                raise ValueError

            pred_grasps = []
            for idx in range(pred_pts.shape[0]):
                row, col = pred_pts[idx]
                angle = angle_pred[row, col] / self.angle_cls * np.pi  # 预测的抓取角弧度
                width = width_pred[row, col]    # 实际长度 m
                row += self.crop_y1
                col += self.crop_x1

                pred_grasps.append([row, col, angle, width, None])  # 抓取深度设为None

        return pred_grasps

    
    def predict_in(self, img_rgb, img_dep, im_mask, input_size, mode, thresh=0.7, peak_dist=1):
        """
        预测抓取模型

        *** 只获取物体区域内的 ***

        img_rgb: rgb图像 np.array (h, w, 3)
        img_dep: 深度图 np.array (h, w)
        input_size: 图像送入神经网络时的尺寸，需要在原图上裁剪
        mode: max, peak, all, nms
        thresh: 置信度阈值
        peak_dist: 置信度筛选峰值
        :return:
            pred_grasps: list([row, col, angle, width])  width单位为米
        """
        # 获取输入tensor
        image = Image()
        if self.use_rgb:
            im_rgb = RGBImage(img_rgb)
            image.rgbimg = im_rgb
        if self.use_dep:
            im_dep = DepthImage(img_dep)
            image.depthimg = im_dep
        self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2 = image.crop(input_size)
        input = self.numpy_to_torch(image.nomalise())    # (n, h, w) / (h, w)

        # 裁剪 mask
        im_mask_crop = im_mask[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]

        # 预测
        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, self.width_out = get_pred(self.net, input.to(self.device))

            # 将物体mask以外的抓取置信度全部设为0
            im_mask_crop = torch.from_numpy(im_mask_crop)
            im_mask_crop = torch.unsqueeze(torch.unsqueeze(im_mask_crop, dim=0), dim=0)
            im_mask_crop = im_mask_crop.to(self.device)
            # print(im_mask_crop.min(), im_mask_crop.max())
            assert  im_mask_crop.shape == self.able_out.shape

            self.able_out = self.able_out * im_mask_crop

            if mode == 'nms':
                angle_pred, width_pred = post_process_output_1(self.able_out, self.angle_out, self.width_out)
                # 获取所有大于thresh**2的抓取，再NMS
                while True:
                    res = np.where(angle_pred > thresh**2)
                    if res[0].shape[0] > 100:
                        break
                    else:
                        thresh -= 0.1
                grasps = None
                for i in range(res[0].shape[0]):
                    _bin, _row, _col = res[0][i], res[1][i], res[2][i]
                    _grasp = np.array([[angle_pred[_bin, _row, _col], _row+self.crop_y1, _col+self.crop_x1, _bin/self.angle_cls*np.pi, width_pred[_bin, _row, _col], -1]], dtype=np.float64)
                    if grasps is None:
                        grasps = _grasp
                    else:
                        grasps = np.concatenate((grasps, _grasp), axis=0)   # shape = (n, 5)

                # 按照置信度从大到小排序
                grasps = grasps[np.argsort(grasps[:,0])[::-1]]
                return NMS(grasps)
            
            able_pred, angle_pred, width_pred = post_process_output(self.able_out, self.angle_out, self.width_out)

            if mode == 'peak':
                # 置信度峰值 抓取点
                pred_pts = peak_local_max(able_pred, min_distance=peak_dist, threshold_abs=thresh)
            elif mode == 'all':
                # 超过阈值的所有抓取点
                pred_pts = arg_thresh(able_pred, thresh=thresh)
            elif mode == 'max':
                # 置信度最大的点
                loc = np.argmax(able_pred)
                row = loc // able_pred.shape[0]
                col = loc % able_pred.shape[0]
                pred_pts = np.array([[row, col]])
            else:
                raise ValueError

            pred_grasps = []
            for idx in range(pred_pts.shape[0]):
                row, col = pred_pts[idx]
                angle = angle_pred[row, col] / self.angle_cls * np.pi  # 预测的抓取角弧度
                width = width_pred[row, col]    # 实际长度 m
                row += self.crop_y1
                col += self.crop_x1

                pred_grasps.append([row, col, angle, width, None])  # 抓取深度设为None

        return pred_grasps


    def confidentMap(self, img_rgb, img_dep, input_size, align=True):
        """
        返回抓取置信度

        img_rgb: rgb图像 np.array (h, w, 3)
        img_dep: 深度图 np.array (h, w)
        input_size: 图像送入神经网络时的尺寸，需要在原图上裁剪
        align: 是否将输出的预测图与输入图像尺寸对齐

        :return:
            confident_map: 抓取置信度图  抓取点置信度 * 抓取角置信度

            如果 align 为 True: confident_map的尺寸与输入图像一致  (angle_bins, h, w)
            否则：尺寸为 (angle_bins, input_size, input_size)
        """
        # 获取输入tensor
        image = Image()
        if self.use_rgb:
            im_rgb = RGBImage(img_rgb)
            image.rgbimg = im_rgb
        if self.use_dep:
            im_dep = DepthImage(img_dep)
            image.depthimg = im_dep
        self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2 = image.crop(input_size)
        input = self.numpy_to_torch(image.nomalise())    # (n, h, w) / (h, w)

        # 预测
        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, _ = get_pred(self.net, input.to(self.device))

        # 抓取角
        angle_pred = self.angle_out.cpu().numpy().squeeze()   # (angle, h, w)    每个元素表示预测的抓取角类别
        # 抓取置信度
        able_pred = self.able_out.cpu().numpy().squeeze()    # (h, w)
        able_pred = np.expand_dims(able_pred, 0).repeat(angle_pred.shape[0], 0) # (angle, h, w)
        confident_map = angle_pred * able_pred

        if align:
            h, w = img_dep.shape
            confident_map_align = np.zeros((confident_map.shape[0], h, w))
            confident_map_align[:, self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2] = confident_map
            confident_map = confident_map_align

        return confident_map



    def able_map(self, img_rgb, img_dep, input_size):
        """
        预测抓取模型

        *** 注意：输出的map尺寸为input_size ***

        img_rgb: rgb图像 np.array (h, w, 3)
        img_dep: 深度图 np.array (h, w)
        input_size: 图像送入神经网络时的尺寸，需要在原图上裁剪
        :return:
            pred_grasps: list([row, col, angle, width])  width单位为米
        """
        # 获取输入tensor
        image = Image()
        if self.use_rgb:
            im_rgb = RGBImage(img_rgb)
            image.rgbimg = im_rgb
        if self.use_dep:
            im_dep = DepthImage(img_dep)
            image.depthimg = im_dep
        self.crop_x1, self.crop_y1, _, _ = image.crop(input_size)
        input = self.numpy_to_torch(image.nomalise())    # (n, h, w) / (h, w)

        # 预测
        self.net.eval()
        with torch.no_grad():
            self.able_out, self.angle_out, self.width_out = get_pred(self.net, input.to(self.device))

            able_map = self.able_out.detach().numpy().squeeze() # 0-1
        
        return able_map
