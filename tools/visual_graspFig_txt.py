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
from shutil import copytree
import sys
sys.path.append(os.curdir)
from utils.tool import depth2Gray3
from scripts.dataset.generate_graspFig import GRASP_ANGLE_BINS
from utils.camera import Camera



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


def save_graspLabel_img(camera_rgb, camera_dep, camera:Camera, path):
    """
    绘制PLGP抓取标签

    path: 抓取数据的保存路径
    camera_rgb: np.uint8
    camera_dep: np.float
    camera: Camera 实例
    """
    # camera_rgb = cv2.imread(path + '/camera_rgb.png')
    # camera_dep = scio.loadmat(path + '/camera_depth.mat')['A']

    # 如果txt文件的最后一行没有字符，则f.readline()会返回None
    grasps = []
    with open(path+'/graspLabel.txt') as f:
        while 1:
            line = f.readline()
            if not line:
                break
            strs = line.split(' ')
            row, col, angle_bin, width, depth = int(strs[0]), int(strs[1]), int(strs[2]), float(strs[3]), float(strs[4])
            angle = (angle_bin / GRASP_ANGLE_BINS) * math.pi
            width = camera.length_TO_pixels(width, camera_dep[row, col])
            grasps.append([row, col, angle, width])
    camera_rgb = np.ascontiguousarray(camera_rgb)
    camera_rgb = drawGrasps(camera_rgb, grasps, mode='line')
    cv2.imwrite(path+'/graspLabel.png', camera_rgb)

def show_grasp(path):
    camera_rgb = scio.loadmat(path + '/camera_rgb.mat')['A']
    camera_dep = scio.loadmat(path + '/camera_depth_rev.mat')['A']
    camera_dep_gray = depth2Gray3(camera_dep)
    table_depth = cv2.imread(path + '/table_depth.png')

    # 如果txt文件的最后一行没有字符，则f.readline()会返回None
    grasps = []
    i = 0
    with open(path+'/graspLabel.txt') as f:
        while 1:
            line = f.readline()
            if not line:
                break
            strs = line.split(' ')
            row, col, angle_bin, width, depth = int(strs[0]), int(strs[1]), int(strs[2]), float(strs[3]), float(strs[4])
            angle = (angle_bin / GRASP_ANGLE_BINS) * math.pi
            width = width / 0.0016
            if depth == 0:
                print('depth = ', depth)
            i += 1
            # if i % 2 == 0:
            grasps.append([row, col, angle, width])
            
    
    im_rgb_region = camera_dep_gray.copy()
    im_rgb_line = camera_dep_gray.copy()
    im_grasp_region = drawGrasps(im_rgb_region, grasps, mode='region')
    im_grasp_line = drawGrasps(im_rgb_line, grasps, mode='line')

    print('grasps = ', len(grasps))

    cv2.imshow('im_grasp_region', im_grasp_region)
    cv2.imshow('im_grasp_line', im_grasp_line)
    cv2.imshow('table_depth', cv2.resize(table_depth, (1000, 1000)))



if __name__ == "__main__":
    # img_path = 'F:/sim_grasp/imgs/done'
    # # err_path = 'F:/sim_grasp/imgs/error'

    # file_names = os.listdir(img_path)
    # start_i = 0
    # i = start_i

    # grasps_counts = 0   # 记录抓取标签的数量
    # for file_name in file_names[start_i:]:
    #     print('filename: ', file_name, ', i = ', i)
    #     i += 1
    #     path = os.path.join(img_path, file_name)

    #     # if not os.path.exists(path+'/graspLabel.txt'):
    #     #     try:
    #     #         copytree(path, os.path.join(err_path, file_name))
    #     #         print(path)
    #     #     except:
    #     #         pass
        
    #     show_grasp(path)    # 可视化像素级抓取标签
    #     # max_depth(path)   # 打印最大深度
    #     # grasps_counts += grasps_count(path) # 获取每张图像的抓取标签数量

    #     key = cv2.waitKeyEx()
    #     if key == 49:   # 1
    #         break
    
    # print(grasps_counts*1.0 / len(file_names[start_i:]))

    # 可视化某个图片的标签
    path = 'E:/research/dataset/grasp/sim_grasp/imgs/test'
    show_grasp(path)
    cv2.waitKeyEx()