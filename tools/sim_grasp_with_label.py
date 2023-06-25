'''
Description: 
Author: wangdx
Date: 2021-09-06 22:09:19
LastEditTime: 2021-11-16 13:58:56
'''
"""
用于验证生成的抓取配置标注

运行流程：
(1) 加载物体和抓取标签
(2) 抓取物体，记录结果
(3) 重置物体位置
(4) 重复(2)和(3)，直到标签验证完毕
(5) 跳到(1)

把抓取失败的情况记录下来
"""

import pybullet as p
import pybullet_data
import time
import math
import cv2
import os
import numpy as np
import sys
import scipy.io as scio
sys.path.append(os.curdir)
from utils.simEnv import SimEnv
# from generate_graspFig import getOneGrasp
import utils.panda_sim_grasp_gripper as gripper_sim
from utils.camera import Camera


GRASP_GAP = 0.005
GRASP_DEPTH = 0.005

def run(path, database_path):
    cid = p.connect(p.GUI)  # 连接服务器
    # cid = p.connect(p.DIRECT)  # 连接服务器
    gripper = gripper_sim.GripperSimAuto(p, [0, 0, 0.3])  # 初始化panda机器人
    env = SimEnv(p, database_path, gripper.gripperId) # 初始化虚拟环境
    camera = Camera()

    print('='*100)
    
    success = 0     # 抓取成功次数
    grasp_count = 0     # 抓取总次数
    tt = 30
    
    env.loadObjsWithPose(path)  # 按照预先保存的位姿加载多物体
    env.loadGraspLabelsFromTxt(path, grasp_start_id=0)   # 加载对应的抓取标签     32
    camera_depth = scio.loadmat(path + '/camera_depth.mat')['A']
    t = 0

    # txt = open('E:/research/dataset/grasp/sim_grasp/imgs/done/fail.txt', 'a+')

    while True:
        p.stepSimulation()
        t += 1
        # if t % tt == 0:
        #     time.sleep(1./240.)

        # 获取抓取配置
        graspFig = env.getGrasp(thresh=0.175)
        if graspFig is None:
            p.disconnect()
            # txt.close()
            return success, grasp_count

        row, col, depth, grasp_angle, grasp_width = graspFig
        grasp_x, grasp_y, grasp_z = camera.img2world([col, row], camera_depth[row, col]) # [x, y, z]
        grasp_z -= depth
        print('graspFig = ', grasp_x, grasp_y, grasp_z, grasp_angle, grasp_width)
        
        dist = 0.3
        offset = 0.114    # 机械手的偏移距离
        gripper.resetPose([grasp_x, grasp_y, grasp_z+dist+offset], grasp_angle, grasp_width/2)

        # 抓取
        while True:
            p.stepSimulation()
            t += 1
            if t % tt == 0:
                time.sleep(1./240.)
            
            if gripper.step(dist):  # 机器人抓取
                t = 0
                break

        # 遍历所有物体，只要有物体位于指定的坐标范围外，就认为抓取正确
        ret = env.evalGrasp(z_thresh=0.2)
        success += 1 if ret else 0
        # if not ret:
        #     txt.write(os.path.basename(path) + ' ' + str(env.grasp_id) + '\n')  # 记录文件名和grasp_id
        grasp_count += 1

        # 重置物体
        env.resetObjsPose(path)



if __name__ == "__main__":
    success_nums = 0
    grasp_count = 0
    # data_path = 'E:/research/dataset/grasp/sim_grasp/imgs/verify_label'
    # data_path = 'E:/research/dataset/grasp/sim_grasp/imgs/test2'
    database_path = 'E:/research/dataset/grasp/sim_grasp/mesh_database'  # 注意database要和data_path使用的database保持一致
    # file_names = os.listdir(data_path)
    # file_names.sort()
    # for file_name in file_names:
    #     print('filename: ', file_name)
    #     path = os.path.join(data_path, file_name)
    #     success, all = run(path, database_path)
    #     success_nums += success
    #     grasp_count += all
    #     print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> accuracy: {}/{}={} \n'.format(success_nums, grasp_count, success_nums/grasp_count))

    success, all = run('E:/research/dataset/grasp/sim_grasp/imgs/test/2', database_path)