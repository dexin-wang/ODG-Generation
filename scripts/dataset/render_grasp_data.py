'''
Description: 
Author: wangdx
Date: 2021-01-18 15:26:24
LastEditTime: 2022-03-22 20:19:29
'''
"""
本程序为生成数据集的主程序
自动加载物体、随机相机视角、渲染图像、计算抓取标签、保存数据
"""
import pybullet as p
import os
import time
import shutil
import sys
sys.path.append(os.curdir)
from utils.simEnv import SimEnv
from scripts.dataset.generate_graspFig import getGrasp
from tools.visual_graspFig_txt import save_graspLabel_img


def run(start_id, num, obj_nums, model_path, model_list, save_path, add_num=1, render_start=0, render_num=5, view_start=0, view_num=3, visual=False):
    """
    生成数据集

    args:
        - start_id: 用于生成数据集的物体集合在列表中的开始id
        - num: 用于生成数据集的物体集合在列表中的开始id
        - obj_nums: 每个场景中物体的数量
        - add_num: 每次更新场景时, 物体id的增量
        - render_start: 初始的场景渲染id, 用于构建样本的文件名
        - render_num: 每个场景的渲染次数
        - view_start: 初始的相机视角id
        - view_num: 每个场景的渲染次数
    """
    START_ID = start_id
    END_ID = START_ID + num
    OBJ_NUMS = obj_nums
    ADD_NUM = add_num
    RENDER_start = render_start
    RENDER_NUM = render_num
    VIEW_start = view_start
    VIEW_NUM = view_num

    # 初始化环境
    if visual:
        cid = p.connect(p.GUI)
    else:
        cid = p.connect(p.DIRECT)
    model_path = ''
    env = SimEnv(p, model_path, model_list) # 初始化虚拟环境

    idx = START_ID
    render_n = RENDER_start
    view_n = VIEW_start

    key_1 = False
    key_2 = False
    t = 0
    while True:
        p.stepSimulation()
        t += 1
        if t == 1:
            key_2 = True
        if t == 240*5:
            key_1 = True
            t = 0

        # 检测按键
        if key_1:
            # 渲染图像
            while view_n < VIEW_NUM:
                save_path_ = os.path.join(save_path, '{:02d}_{:04d}_{}_{}'.format(OBJ_NUMS, idx, render_n, view_n))
                try:
                    tim = time.time()
                    camera_rgb, camera_depth, camera_mask, parallel_depth, parallel_mask = env.renderGraspImages(save_path_, parallel=True)
                    env.save_cameraData(save_path_)
                    env.saveObjsPose(save_path_)
                    getGrasp(camera_depth, camera_mask, parallel_depth, parallel_mask, env.camera, save_path_)
                    save_graspLabel_img(camera_rgb, camera_depth, env.camera, save_path_)
                    print('耗时 = ', time.time() - tim)
                    env.movecamera(isRandom=True)
                except:
                    shutil.rmtree(save_path_)
                view_n += 1
            view_n = VIEW_start
            render_n += 1
            key_1 = False
        
        if key_2:
            env.movecamera()
            if render_n >= RENDER_NUM:
                idx += ADD_NUM
                render_n = RENDER_start
                print('=========================== 更新物体: ', idx, render_n)
                env.deleteAllURDFs()
                env.loadURDFs(idx, OBJ_NUMS)
            else:
                print('=========================== 重置物体位置: ', idx, render_n)
                if env.urdfs_obj_num() == 0:
                    env.loadURDFs(idx, OBJ_NUMS)
                else:
                    env.resetURDFsPoseRandom()
            key_2 = False

        if idx >= min(END_ID, env.urdf_list_num()):
            p.disconnect()
            break


if __name__ == "__main__":
    # 因为使用同一个仿真环境加载过多物体时，内存占用过大加载不进去，所以每渲染一部分，就重启仿真环境
    start_id = 0
    end_id = 5
    num = 5

    obj_nums = 5
    # model path
    model_path = 'D:/research/grasp_detection/sim_grasp/openSource/obj_models'
    # objects model
    model_list = 'scripts/models_list/train.txt'
    # dataset path
    save_path = 'D:/research/grasp_detection/sim_grasp/openSource/dataset'

    while start_id < end_id:
        run(start_id, num, obj_nums, model_path, model_list, save_path, visual=True)
        start_id += num