'''
Description: 
Author: wangdx
Date: 2021-01-20 20:22:48
LastEditTime: 2022-03-11 10:27:23

创建抓取数据集

'''

import os
import scipy.io as scio
import numpy as np
from shutil import Error, copyfile
from scripts.dataset.generate_graspFig import GRASP_ANGLE_BINS, HEIGHT, WIDTH

neighbor_val = 0.8


def run():
    src_path = 'D:/research/grasp_detection/sim_grasp/openSource/dataset/train'        # 抓取源数据
    dst_path = 'D:/research/grasp_detection/sim_grasp/openSource/PLGP-Dataset/train'      # 数据集路径

    folders = os.listdir(src_path)
    for folder in folders:
        print('processing ', folder)
        path = os.path.join(src_path, folder)

        if not os.path.exists(path + '/graspLabel.txt'):
            raise Error
        
        grasp_point_map = np.zeros((HEIGHT, WIDTH), np.float64)
        grasp_angle_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.float64)
        grasp_width_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.float64)
        grasp_depth_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.float64)

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
                grasp_depth_map[row, col, angle_bin] = depth
        
        H, W, bins = grasp_angle_map.shape  # 480, 640, 18

        # =================== 数据集预处理 ===================
        # 预处理方式查看 像素级仿真数据及+网络思路.txt
        grasp_pts = np.where(grasp_point_map > 0)   # 标注的抓取点
        for i in range(grasp_pts[0].shape[0]):      # 遍历标注点
            row, col = grasp_pts[0][i], grasp_pts[1][i]
            # 对于相邻的8个点，置信度设为max(原值，0.8)
            t = max(row-1, 0)
            b = min(row+1, H-1)
            l = max(col-1, 0)
            r = min(col+1, W-1)
            for r_ in range(t, b+1):
                    for c_ in range(l, r+1):
                        if grasp_point_map[r_, c_] == 0:
                            grasp_point_map[r_, c_] = neighbor_val

            # 扩展当前位置的抓取角和抓取宽度
            angle_bins = np.where(grasp_angle_map[row, col] == 1.0)[0]
            for angle_bin in angle_bins:    # 0-17
                angle_bin_1, angle_bin_2 = (angle_bin+1) % bins, (angle_bin-1+bins) % bins
                grasp_angle_map[row, col, angle_bin_1] = max(grasp_angle_map[row, col, angle_bin_1], neighbor_val)
                grasp_angle_map[row, col, angle_bin_2] = max(grasp_angle_map[row, col, angle_bin_2], neighbor_val)
                # 只更新值为0的抓取宽度
                grasp_width_map[row, col, angle_bin_1] = grasp_width_map[row, col, angle_bin] if grasp_width_map[row, col, angle_bin_1] == 0 else grasp_width_map[row, col, angle_bin_1]
                grasp_width_map[row, col, angle_bin_2] = grasp_width_map[row, col, angle_bin] if grasp_width_map[row, col, angle_bin_2] == 0 else grasp_width_map[row, col, angle_bin_2]
                # 只更新值为0的抓取深度
                grasp_depth_map[row, col, angle_bin_1] = grasp_depth_map[row, col, angle_bin] if grasp_depth_map[row, col, angle_bin_1] == 0 else grasp_depth_map[row, col, angle_bin_1]
                grasp_depth_map[row, col, angle_bin_2] = grasp_depth_map[row, col, angle_bin] if grasp_depth_map[row, col, angle_bin_2] == 0 else grasp_depth_map[row, col, angle_bin_2]
            
            # 扩展相邻位置的抓取角和抓取宽度
            angle_bins = np.where(grasp_angle_map[row, col] == 1.0)[0]  # 只扩展置信度为1的抓取角
            # 扩展相邻8个位置的抓取宽度  只更新值为0的抓取宽度
            for angle_bin in angle_bins:    # 0-17
                for r_ in range(t, b+1):
                    for c_ in range(l, r+1):
                        # 扩展抓取角
                        grasp_angle_map[r_, c_, angle_bin] = neighbor_val if grasp_angle_map[r_, c_, angle_bin] == 0 else grasp_angle_map[r_, c_, angle_bin]
                        # 扩展抓取宽度
                        grasp_width_map[r_, c_, angle_bin] = grasp_width_map[row, col, angle_bin] if grasp_width_map[r_, c_, angle_bin] == 0 else grasp_width_map[r_, c_, angle_bin]
                        # 扩展抓取深度
                        grasp_depth_map[r_, c_, angle_bin] = grasp_depth_map[row, col, angle_bin] if grasp_depth_map[r_, c_, angle_bin] == 0 else grasp_depth_map[r_, c_, angle_bin]

        # =================== 将抓取合并为一个mat文件进行保存 ===================
        # 单个文件达到88M，不使用该方案
        # grasp_point_map = np.expand_dims(grasp_point_map, 2)
        # grasp_label = np.concatenate((grasp_point_map, grasp_angle_map, grasp_width_map), axis=2)     # (480, 640, 37)
        # label_name = file_name + '_grasp.mat'
        # scio.savemat(os.path.join(dst_path, label_name), {'A':grasp_label})
        # print('saved ', label_name)

        '''
        置信度为1的抓取点，可能对应置信度为0.5和1的抓取角;
        置信度为0.5的抓取点，只对应置信度为0.5的抓取角
        '''

        # =================== 将抓取标注合并为一个txt文件进行保存 ===================
        # 遍历分3次进行写入，第一次写入抓取点置信度为1且抓取角置信度为1的抓取，第二次写入抓取点置信度为1且抓取角置信度为0.5的抓取
        # 第三次写入抓取点置信度为0.5且抓取角置信度为0.5的抓取
        # 用空格隔开
        label_name = folder + '_grasp.txt'
        f = open(os.path.join(dst_path, label_name), 'w')

        # 抓取点置信度为1
        grasp_pts = np.where(grasp_point_map == 1.0)
        for i in range(grasp_pts[0].shape[0]):
            row, col = grasp_pts[0][i], grasp_pts[1][i]
            # 抓取角置信度为1
            angle_bins = np.where(grasp_angle_map[row, col] == 1.0)[0]
            for angle_bin in angle_bins:
                width = grasp_width_map[row, col, angle_bin]
                depth = grasp_depth_map[row, col, angle_bin]
                line = str(row) + ' ' + str(col) + ' ' + str(angle_bin) +  ' ' + str(width) + ' ' + str(depth) + '\n'
                f.write(line)
        f.write('\n')

        # 抓取点置信度为1
        grasp_pts = np.where(grasp_point_map == 1.0)
        for i in range(grasp_pts[0].shape[0]):
            row, col = grasp_pts[0][i], grasp_pts[1][i]
            # 抓取角置信度为0.5
            angle_bins = np.where(grasp_angle_map[row, col] == neighbor_val)[0]
            for angle_bin in angle_bins:
                width = grasp_width_map[row, col, angle_bin]
                depth = grasp_depth_map[row, col, angle_bin]
                line = str(row) + ' ' + str(col) + ' ' + str(angle_bin) +  ' ' + str(width) + ' ' + str(depth) + '\n'
                f.write(line)
        f.write('\n')

        # 抓取点置信度为0.5
        grasp_pts = np.where(grasp_point_map == neighbor_val)
        for i in range(grasp_pts[0].shape[0]):
            row, col = grasp_pts[0][i], grasp_pts[1][i]
            # 抓取角置信度为0.5
            angle_bins = np.where(grasp_angle_map[row, col] == neighbor_val)[0]
            for angle_bin in angle_bins:
                width = grasp_width_map[row, col, angle_bin]
                depth = grasp_depth_map[row, col, angle_bin]
                line = str(row) + ' ' + str(col) + ' ' + str(angle_bin) +  ' ' + str(width) + ' ' + str(depth) + '\n'
                f.write(line)
                
        f.close()
        print('saved ', label_name)

        # =================== 复制相机RGB和深度图 ===================
        # 将深度图转为uint16，空间缩小为1/4
        depth_float64 = scio.loadmat(path + '/camera_depth.mat')['A']
        scio.savemat(os.path.join(dst_path, folder + '_depth.mat'), {'A':(depth_float64*1000).astype(np.uint16)})
        # copyfile(path + '/camera_depth.mat', os.path.join(dst_path, folder + '_depth.mat'))   # 直接复制，占空间较大
        copyfile(path + '/camera_rgb.png', os.path.join(dst_path, folder + '_rgb.png'))


if __name__ == "__main__":
    run()
