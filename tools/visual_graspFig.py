"""
1 可视化原始的像素级抓取标签
2 获取所有标签的数量和平均每张图像的标签数
"""

import cv2
import os
import glob
import scipy.io as scio
import numpy as np
import math

HEIGHT = 480
WIDTH = 640
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

def show_grasp(path):
    camera_rgb = scio.loadmat(path + '/camera_rgb.mat')['A']
    table_depth = cv2.imread(path + '/table_depth.png')
    grasp_point_map = np.zeros((HEIGHT, WIDTH), np.float64)
    grasp_angle_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.float64)
    grasp_width_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.float64)

    # 如果txt文件的最后一行没有字符，则f.readline()会返回None
    with open(path+'/graspLabel.txt') as f:
        while 1:
            line = f.readline()
            if not line:
                break
            strs = line.split(' ')

            row, col, angle_bin, width, depth = int(strs[0]), int(strs[1]), int(strs[2]), float(strs[3]), float(strs[4])
            grasp_point_map[row, col] = 1.0
            grasp_angle_map[row, col, angle_bin] = 1.0
            grasp_width_map[row, col, angle_bin] = width

    grasps = []
    grasp_pts = np.where(grasp_point_map == 1.0)
    for i in range(grasp_pts[0].shape[0]):
        
        if i % 2 != 0:
            continue

        row, col = grasp_pts[0][i], grasp_pts[1][i]
        angle_bins = np.where(grasp_angle_map[row, col] == 1.0)[0]
        for angle_bin in angle_bins:
            angle = (angle_bin / GRASP_ANGLE_BINS) * math.pi
            width = grasp_width_map[row, col, angle_bin] / 0.0016
            grasps.append([row, col, angle, width])
    
    im_rgb_region = camera_rgb.copy()
    im_rgb_line = camera_rgb.copy()
    im_grasp_region = drawGrasps(im_rgb_region, grasps, mode='region')
    im_grasp_line = drawGrasps(im_rgb_line, grasps, mode='line')

    print('grasps = ', len(grasps))

    cv2.imshow('im_grasp_region', im_grasp_region)
    cv2.imshow('im_grasp_line', im_grasp_line)
    cv2.imshow('table_depth', cv2.resize(table_depth, (1000, 1000)))

def max_depth(path):
    camera_dep_rev = scio.loadmat(path + '/camera_depth_rev.mat')['A']
    _max = np.max(camera_dep_rev)
    if _max > 0.5:
        print(path, _max)
        print('\n')

def grasps_count(path):
    grasp_point_map = scio.loadmat(path + '/grasp_18/grasp_point_map.mat')['A']
    grasp_angle_map = scio.loadmat(path + '/grasp_18/grasp_angle_map.mat')['A']

    count = 0
    grasp_pts = np.where(grasp_point_map > 0)
    for i in range(grasp_pts[0].shape[0]):
        row, col = grasp_pts[0][i], grasp_pts[1][i]
        angle_bins = np.where(grasp_angle_map[row, col] > 0)[0]
        count += len(angle_bins)
    
    return count


if __name__ == "__main__":
    img_path = 'E:/research/grasp_detection/sim_grasp/docs/test'

    file_names = os.listdir(img_path)
    start_i = 0
    i = start_i

    grasps_counts = 0   # 记录抓取标签的数量
    for file_name in file_names[start_i:]:
        print('filename: ', file_name, ', i = ', i)
        i += 1
        path = os.path.join(img_path, file_name)
        
        show_grasp(path)    # 可视化像素级抓取标签
        # max_depth(path)   # 打印最大深度
        # grasps_counts += grasps_count(path) # 获取每张图像的抓取标签数量

        key = cv2.waitKeyEx()
        if key == 49:   # 1
            break
    
    print(grasps_counts*1.0 / len(file_names[start_i:]))

    # 可视化某个图片的标签
    # path = 'E:/research/dataset/grasp/my_dense/imgs/imgs_dex_1/err/01_000069_2'
    # show_grasp(path)
    # cv2.waitKeyEx()