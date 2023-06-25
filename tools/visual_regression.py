"""
可视化抓取回归数据集的标签
读取深度图和txt格式的标签
"""

import cv2
import sys
import os
import glob
import scipy.io as scio
import numpy as np
import math
sys.path.append(os.curdir)
import utils.tool as tool


GRASP_ANGLE_BINS = 18


def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    """
    绘制grasp
    file: img路径
    grasps: list()	元素是 [row, col, angle, width]
    angle: 弧度
    width: 单位 像素
    mode: 显示模式 'line' or 'region'
    """

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

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
            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

        # color_b = 255 / num * i
        # color_r = 0
        # color_g = -255 / num * i + 255

        color_b = 0
        color_r = 0
        color_g = 255

        if mode == 'line':
            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)
        else:
            img[row, col] = [color_b, color_g, color_r]
        

    return img

def load_grasptxt(file, HEIGHT=480, WIDTH=640, angle_k=18):
    """
    file: txt格式的抓取回归标签文件
    file内容分为三个部分：抓取点置信度为1且抓取角置信度为1的抓取，抓取点置信度为1且抓取角置信度为0.5的抓取， 抓取点置信度为0.5且抓取角置信度为0.5的抓取
    每部分用一个空行隔开
    """
    grasp_point_map = np.zeros((HEIGHT, WIDTH), np.float64)
    grasp_angle_map = np.zeros((angle_k, HEIGHT, WIDTH), np.float64)
    grasp_width_map = np.zeros((angle_k, HEIGHT, WIDTH), np.float64)

    # 如果txt文件的最后一行没有字符，则f.readline()会返回None
    mode = 0
    with open(file) as f:
        while 1:
            line = f.readline()
            if not line:
                break
            strs = line.split(' ')
            if len(strs) != 5:
                mode += 1
                continue

            row, col, angle_bin, width, depth = int(strs[0]), int(strs[1]), int(strs[2]), float(strs[3]), float(strs[4])
            grasp_point_map[row, col] = 1.0 if mode in [0, 1] else 0.5
            grasp_angle_map[angle_bin, row, col] = 1.0 if mode == 0 else 0.5
            grasp_width_map[angle_bin, row, col] = width
    
    return grasp_point_map, grasp_angle_map, grasp_width_map


def visual_regression(dataset_path, start_id=0):
    """
    dataset_path: 数据集路径
    start_id: 从start_id个样本开始可视化
    """
    grasp_label_files = glob.glob(os.path.join(dataset_path, '*grasp.txt'))

    for grasp_label_file in grasp_label_files[start_id:]:
        camera_depth_file = grasp_label_file.replace('grasp.txt', 'depth.mat')
        print(camera_depth_file)

        # 读取depth
        camera_depth = scio.loadmat(camera_depth_file)['A']
        camera_depth = tool.depth2Gray3(camera_depth)
        cv2.imshow('camera_depth', camera_depth)

        # 读取txt格式的抓取
        grasp_point_map, grasp_angle_map, grasp_width_map = load_grasptxt(grasp_label_file)
        # 绘制抓取
        grasps = []
        grasp_pts = np.where(grasp_point_map > 0)
        for i in range(grasp_pts[0].shape[0]):
            
            # if i % 2 != 0:
            #     continue

            row, col = grasp_pts[0][i], grasp_pts[1][i]
            angle_bins = np.where(grasp_angle_map[:, row, col] > 0)[0]
            for angle_bin in angle_bins:
                angle = (angle_bin / GRASP_ANGLE_BINS) * math.pi
                width = grasp_width_map[angle_bin, row, col] / 0.0016   # 这里直接
                grasps.append([row, col, angle, width])
        
        camera_depth_region = camera_depth.copy()
        camera_depth_line = camera_depth.copy()
        im_grasp_region = drawGrasps(camera_depth_region, grasps, mode='region')
        im_grasp_line = drawGrasps(camera_depth_line, grasps, mode='line')
        cv2.imshow('im_grasp_region', im_grasp_region)
        cv2.imshow('im_grasp_line', im_grasp_line)

        if cv2.waitKey() == 27:
            break
    


if __name__ == "__main__":
    visual_regression('E:/research/dataset/grasp/sim_grasp/imgs/dataset/train', start_id=0)

    