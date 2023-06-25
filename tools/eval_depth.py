'''
Description: 
Author: wangdx
Date: 2021-01-20 20:22:48
LastEditTime: 2021-11-15 15:11:35

创建抓取回归数据集
同时进行数据集预处理，预处理方式查看 像素级仿真数据及+网络思路.txt

txt文件每行表示一个抓取配置 [row, col, angle_bin, width, depth]
txt文件比mat文件占空间小？

命名格式：
{database}_{objs}_{idx}_{render}_depth.txt
database: 数据库 1-egad, 2-dex
objs: 图像中的物体数量
idx: 一个物体时，图像中包含的物体的idx；多个物体时，图像中包含的物体的最小idx。idx即物体在数据库中的索引
render: 该场景的第render次渲染。
'''

import os
import scipy.io as scio
import numpy as np
from shutil import copyfile


neighbor_val = 0.8


def run():
    src_path = 'F:/my_dense/imgs/clutter_train'        # 抓取源数据

    all_count = 0
    dep_zero_count = 0

    folders = os.listdir(src_path)
    for folder in folders:
        folder = os.path.join(src_path, folder)
        file_names = os.listdir(folder)
        for file_name in file_names:
            path = os.path.join(folder, file_name)
            print(path)

            if not os.path.exists(path + '/grasp_18/grasp_point_map.mat'):
                continue

            grasp_point_map = scio.loadmat(path + '/grasp_18/grasp_point_map.mat')['A'].astype(np.float64)   # (h, w)
            grasp_angle_map = scio.loadmat(path + '/grasp_18/grasp_angle_map.mat')['A'].astype(np.float64)   # (h, w, bin)
            grasp_width_map = scio.loadmat(path + '/grasp_18/grasp_width_map.mat')['A'].astype(np.float64)   # (h, w, bin)
            grasp_depth_map = scio.loadmat(path + '/grasp_18/grasp_depth_map.mat')['A']                    # (h, w, bin)   抓取点相对于桌面的高度
            

            # =================== 数据集预处理 ===================
            # 预处理方式查看 像素级仿真数据及+网络思路.txt
            grasp_pts = np.where(grasp_point_map > 0)   # 标注的抓取点
            for i in range(grasp_pts[0].shape[0]):      # 遍历标注点
                row, col = grasp_pts[0][i], grasp_pts[1][i]

                # 扩展当前位置的抓取角和抓取宽度
                angle_bins = np.where(grasp_angle_map[row, col] == 1.0)[0]
                for angle_bin in angle_bins:    # 0-17
                    all_count += 1
                    width = grasp_width_map[row, col, angle_bin]
                    if width == 0:
                        print('depth = ', width)
                        dep_zero_count += 1
    
    print('all_count = ', all_count)
    print('dep_zero_count = ', dep_zero_count)
           

            


if __name__ == "__main__":
    run()
