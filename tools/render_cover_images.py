'''
Description: 
Author: wangdx
Date: 2021-01-18 15:26:24
LastEditTime: 2021-10-05 14:07:37
'''
"""
自动渲染用于计算抓取配置所需的图像和文件

1324 + 98 + 4562 / 2 = 2992个物体

每个场景1个物体：每个场景渲染1次，共渲染 2992

每个场景10个物体：每个场景渲染8次，共渲染 2992 / 10  * 8 = 2392  最后剩一个
重复上述过程5次，每次的随机种子不同 共??个样本

每个场景15个物体：每个场景渲染6次，共渲染 2992 / 15  * 6 = 1200
重复上述过程20次，每次的随机种子不同 共23920个样本

每个场景20个物体：每个场景渲染4次，共渲染 2992 / 20  * 4 = 598
重复上述过程40次，每次的随机种子不同 共23920个样本

每个场景25个物体：每个场景渲染3次，共渲染 2992 / 25  * 3 = 359
重复上述过程60次，每次的随机种子不同 共21540个样本

"""

import random
import pybullet as p
import time
import os
import math
import cv2
import shutil
import numpy as np
import sys
sys.path.append(os.curdir)
from utils.simEnv import SimEnv
# import panda_sim_grasp as panda_sim
# from generate_graspFig import getGrasp


def run(start_id, num, save_id, urdf_path, save_path, RENDER_NUM, seed):
    START_ID = start_id     # 物体开始的id
    END_ID = START_ID + num       # 物体结束的id  START_ID + num  记得改回去
    OBJ_NUMS = num        # 加载的物体数量
    ADD_NUM = num         # 每次更新物体时，idx的增量；当场景只有一个物体时,add_num=1; 当场景有多个物体时，add_num根据需求设置
    RENDER_NUM = RENDER_NUM      # 每个场景的渲染次数 不超过10次
    random.seed(seed)      # 随机种子

    # 初始化环境
    p.connect(p.GUI)  # 连接服务器
    env = SimEnv(p, urdf_path) # 初始化虚拟环境

    idx = START_ID
    render_n = 0
    key_1 = False
    key_2 = False
    t = 0
    
    while True:
        p.stepSimulation()
        # time.sleep(1./240.)  # 240
        t += 1
        if t == 1:
            key_2 = True
        if t == 240*2:      # 240*5
            key_1 = True
            t = 0

        # 检测按键
        if key_1:
            # 渲染图像
            save_path_ = os.path.join(save_path, '{:02d}_{:06d}_{}'.format(OBJ_NUMS, save_id, render_n))
            print('>> saving cover ...', save_id)
            env.renderImageAndCover(save_path_)
            render_n += 1
            save_id += 1
            key_1 = False
        
        # 根据物体的位置是否变化自动切换(待写)
        if key_2:
            if render_n >= RENDER_NUM:
                idx += ADD_NUM
                render_n = 0
                print('=========================== 更新物体: ', idx, render_n)
            else:
                print('=========================== 重置物体位置: ', idx, render_n)

            # 加载的物体数超过一定数量后，重新加载仿真环境
            if idx >= min(END_ID, env._urdf_nums()):
                p.disconnect()
                return save_id

            env.removeObjsInURDF()
            env.loadObjsInURDF(idx, OBJ_NUMS, render_n)   # idx: 加载物体的起始id，idx=-1时，随机加载
            key_2 = False



if __name__ == "__main__":
    # 因为使用同一个仿真环境渲染多次，会出现卡顿情况，所以每渲染一部分，就重启仿真环境
    start_id = 0        # 开始的物体id
    save_id = 9360      # 开始保存的样本id
    num = 25        # 和ADD_NUM一样
    RENDER_NUM = 3
    
    # 物体模型路径
    urdf_path = ['E:/research/dataset/grasp/sim_grasp/mesh_database/dex-net', 
                 'E:/research/dataset/grasp/sim_grasp/mesh_database/egad_eval_set', 
                 'E:/research/dataset/grasp/sim_grasp/mesh_database/egad_train_set']
    
    seeds = [i+1 for i in range(1000)]    # 1-20

    repeats = 56
    repeat_cur = 27
    while repeat_cur <= repeats:
        # 文件保存路径
        save_path = 'E:/research/dataset/grasp/sim_grasp/cover_data/' + '{:02d}_{:03d}'.format(num, repeat_cur)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        while start_id <= 2991:
            print('rerun pybullet ...')
            save_id = run(start_id, num, save_id, urdf_path, save_path, RENDER_NUM, seed=seeds[repeat_cur-1])
            start_id += num
        
        repeat_cur += 1
        start_id = 0
