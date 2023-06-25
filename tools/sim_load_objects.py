'''
Description: 
Author: wangdx
Date: 2021-09-06 22:09:19
LastEditTime: 2021-12-20 11:33:12
'''
"""
按顺序加载物体，用于观察物体大小和形状

操作步骤：
1、运行后自动开始加载物体到仿真环境中。
2、按 1 开始渲染深度图。
3、等待渲染结束，按 2 开始计算抓取位姿，计算完毕后，自动开始抓取。
4、按 3 可重新加载物体。
"""

import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np
import sys
sys.path.append(os.curdir)
from utils.simEnv import SimEnv
import utils.robotiq_sim_grasp_gripper as gripper_sim


def run():
    database_path = 'E:/research/dataset/grasp/sim_grasp/mesh_database/dataset'      # 数据库路径
    listname = 'list_small.txt'
    
    cid = p.connect(p.GUI)  # 连接服务器
    gripper = gripper_sim.GripperSimAuto(p, [0, 0, 0.25])  # 初始化抓取器
    env = SimEnv(p, database_path, listname) # 初始化虚拟环境

    obj_nums = 10    # 每次加载的物体个数
    start_idx = 0   # 开始加载的物体id
    idx = start_idx
    
    while True:
        # 加载物体
        env.loadObjsInURDF(idx, obj_nums)
        # env.loadObjsWithPose1('E:/research/dataset/grasp/sim_grasp/mesh_database/visual/data/objsPose.mat')
        idx += obj_nums

        while True:
            p.stepSimulation()
            time.sleep(1./480.)
            
            # 检测按键
            keys = p.getKeyboardEvents()

            # 按3重置环境            
            if ord('3') in keys and keys[ord('3')]&p.KEY_WAS_TRIGGERED:
                env.removeObjsInURDF()
                break



if __name__ == "__main__":
    run()