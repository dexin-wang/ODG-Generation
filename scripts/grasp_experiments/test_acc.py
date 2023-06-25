'''
Description: 
Author: wangdx
Date: 2021-09-06 22:09:19
LastEditTime: 2022-03-31 17:15:46
'''
"""
用于验证神经网络预测的抓取

运行流程：
(1) 加载物体和渲染的深度图
(2) 输入网络，获取预测抓取
(3) 相机移动至抓取点上方，计算最大的抓取深度
(4) 实施抓取
"""

import pybullet as p
import pybullet_data
import time
import math
import cv2
import os
import numpy as np
import sys
sys.path.append(os.curdir)
from utils.simEnv import SimEnv
import utils.tool as tool
from grasp_methods.grasp import GraspMethod, GraspPose


grasp_length = 0.4
input_size = 384


def run(grasp_method:GraspMethod, model_path, model_list, pose_path, friction):
    """
    model_list: 保存模型列表的txt文件
    pose_path: 保存场景模型位姿的mat文件
    friction: 机械手的摩擦系数
    """
    cid = p.connect(p.GUI)  # 连接服务器
    # cid = p.connect(p.DIRECT)  # 连接服务器
    env = SimEnv(p, model_path, model_list, load_gripper=True, friction=friction) # 初始化虚拟环境类
    grasp_method.setcamera(env.camera)

    success_grasp = 0
    sum_grasp = 0
    tt = 5

    # 按照预先保存的位姿加载多物体
    env.loadURDFsWithPose(pose_path)
    t = 0
    continue_fail = 0
    while True:
        # 等物体稳定
        for _ in range(10*1):
            p.stepSimulation()
        
        # 避免机械手挡住相机
        gp = GraspPose(pos=[2, 0, 0.2])
        env.resetGripperPose(gp)
        # 渲染图像
        im_rgb, im_dep = env.renderCameraImages()
        im_dep = tool.add_noise_depth_experiment(im_dep)

        # cv2.imshow('im_dep', tool.depth2RGB(im_dep))
        # cv2.waitKey()

        try:
            # 计算6DOF抓取(相机坐标系)
            grasppose = grasp_method.get6DOFGraspPose(im_rgb, im_dep)
            # print('graspPose_6DOF.center = ', grasppose.center)
            # print('graspPose_6DOF.rotate_mat =', grasppose.rotate_mat)
            # grasp_method.draw6DOFGraspPose(im_rgb, im_dep, grasppose)  # 绘制当前的抓取位姿
            
            # 将抓取位姿转换到世界坐标系
            T_world_camera = env.camera.rigidTransform
            grasppose.transform(T_world_camera)
            # 设置预抓取位姿 T_world_grasp1
            pregrasp_center = grasppose.center - np.array([0, 0, -grasp_length])
            pregrasppose = GraspPose(pos=pregrasp_center, rotMat=grasppose.rotate_mat, width=grasppose.width, frame='world')
            # 设置机械手位姿
            env.gripper.resetGripperPose(pregrasppose)

            # 抓取
            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t % tt == 0:
                    time.sleep(1./240.)
                
                if env.gripper.step(grasp_length):  # 机器人抓取
                    t = 0
                    break
        except:
            pass

        # 判断抓取是否成功
        sum_grasp += 1
        if env.evalGrasp1(thresh=0.2, removeGrasped=True):
            success_grasp += 1
            continue_fail = 0
            if env.urdfs_obj_num() == 0:
                p.disconnect()
                return success_grasp, sum_grasp
        else:
            continue_fail += 1
            if continue_fail == 5:
                p.disconnect()
                return success_grasp, sum_grasp
        
        
if __name__ == "__main__":
    net = 'affga'    # 网络架构 'ggcnn2', 'deeplabv3', 'grcnn', 'unet', 'segnet', 'stdc', 'danet'
    use_rgb = False  # 是否使用rgb
    use_dep = True  # 是否使用深度图
    device='cpu'    # cpu / cuda:0

    model_dir = os.path.join('./grasp_methods/ckpt', net + '_' + 'rgb'*use_rgb + 'd'*use_dep)
    model_path = os.path.join(model_dir, os.listdir(model_dir)[0])
    print('model_path = ', model_path)

    objs_num = [5, 15]       # 场景的物体数量
    friction = [0.2, 0.4, 0.6, 0.8, 1.0]   # 机械手的摩擦系数
    obj_mode = ['seen', 'novel']   # seen/novel
    
    cur_dir = os.path.dirname(__file__)    # 当前文件所在目录
    save_txt = os.path.join(os.path.dirname(__file__), 'results', '{}_{}_acc.txt'.format(net, 'rgb'*use_rgb + 'd'*use_dep)) # 结果保存文件

    poses_path = os.path.join(cur_dir, 'poses') 
    obj_model_path = 'D:/research/grasp_detection/sim_grasp/openSource/obj_models'
    model_lists_path = os.path.join(cur_dir, '../models_list')

    grasp_method = GraspMethod(net, use_rgb, use_dep, model_path, None, input_size, device=device)

    for on in objs_num:
        for om in obj_mode:
            aver_acc = 0.
            aver_clear = 0.
            for f in friction:
    
                success_grasp = 0   # 成功的抓取次数
                all_grasp = 0       # 全部的抓取次数(成功抓取的物体数量)
                all_objs_num = 0    # 需要抓取的物体数量

                pose_path = os.path.join(poses_path, om, '{:d}'.format(on))     # 场景位姿路径
                model_list_path = os.path.join(model_lists_path, 'eval_{}.txt'.format(om))  # 模型列表路径
                # 获取
                for pose_mat in os.listdir(pose_path):
                    print('pose_mat: ', pose_mat)
                    all_objs_num += int(pose_mat[:2])   # 记录场景加载的物体数量
                    # 运行仿真抓取
                    _success, _all = run(grasp_method, obj_model_path, model_list_path, os.path.join(pose_path, pose_mat), f)
                    success_grasp += _success
                    all_grasp += _all
                    print('\n>>>>>>>>>>>>>>>>>>>> Success Rate: {}/{}={}'.format(success_grasp, all_grasp, success_grasp/all_grasp))     
                    print('\n>>>>>>>>>>>>>>>>>>>> Percent Cleared: {}/{}={}'.format(success_grasp, all_objs_num, success_grasp/all_objs_num))    
                
                # 记录结果
                result1 = '{}_{:02d}objs_{}_{:.1f}_success: {:.4f} \n'.format(net, on, om, f, success_grasp/all_grasp)
                result2 = '{}_{:02d}objs_{}_{:.1f}_cleared: {:.4f} \n'.format(net, on, om, f, success_grasp/all_objs_num)
                aver_acc += success_grasp/all_grasp
                aver_clear += success_grasp/all_objs_num
                print('write ...', result1, result2)
                
                file = open(save_txt.format(net), 'a')
                file.write(result1)
                file.write(result2)
                file.close()

            aver_acc /= len(friction)
            aver_clear /= len(friction)
            file = open(save_txt.format(net), 'a')
            file.write('{}_{:02d}objs_{}_aver_acc = {:.4f}\n'.format(net, on, om, aver_acc))
            file.write('{}_{:02d}objs_{}_aver_clear = {:.4f}\n\n'.format(net, on, om, aver_clear))
            file.close()