'''
Description: 
Author: wangdx
Date: 2021-09-06 22:09:19
LastEditTime: 2022-03-31 17:09:59
'''
"""
用于验证神经网络预测的抓取AP
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
import utils.tool as tool
from grasp_methods.grasp import GraspMethod, GraspPose

grasp_length = 0.2
input_size = 384
test_num = 50   # 抓取数量

def run(grasp_method:GraspMethod, model_path, model_list, pose_path, friction):
    id = p.connect(p.GUI)  # 连接服务器
    # cid = p.connect(p.DIRECT)  # 连接服务器
    env = SimEnv(p, model_path, model_list, load_gripper=True, friction=friction) # 初始化虚拟环境类
    grasp_method.setcamera(env.camera)

    success_grasp = 0
    sum_grasp = 0
    tt = 5

    # 按照预先保存的位姿加载多物体
    env.loadURDFsWithPose(pose_path)
    t = 0
    
    # 等物体稳定
    env.sleep(10)
        
    # 渲染图像
    im_rgb, im_dep = env.renderCameraImages()
    im_dep = tool.add_noise_depth_experiment(im_dep)
    # grasp_method.drawPCL(im_rgb, im_dep)  

    # 预测抓取
    graspposes = grasp_method.get6DOFGraspPoses(im_rgb, im_dep, num=test_num)
    print('预测的抓取数量：', len(graspposes))
    # grasp_method.draw6DOFGraspPoses(im_rgb, im_dep, graspposes) # 绘制抓取位姿
    
    for i in range(test_num):
        print('='*50, 'grasp num:', i)
        # 解码抓取配置
        if i >= len(graspposes):
            sum_grasp += 1
            continue
        
        # 获取第i个抓取位姿
        grasppose = graspposes[i]
        # print('grasppose.position = ', grasppose.center)
        # print('grasppose.rotate_mat =', grasppose.rotate_mat) 

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

        # 判断抓取是否成功
        sum_grasp += 1
        if env.evalGrasp2():
            success_grasp += 1
        
        env.resetURDFsPose(pose_path) # 重置物体位置
        # 等物体稳定
        env.sleep(10)

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
    save_txt = os.path.join(os.path.dirname(__file__), 'results', '{}_{}_AP.txt'.format(net, 'rgb'*use_rgb + 'd'*use_dep)) # 结果保存文件

    poses_path = os.path.join(cur_dir, 'poses') 
    obj_model_path = 'D:/research/grasp_detection/sim_grasp/openSource/obj_models'
    model_lists_path = os.path.join(cur_dir, '../models_list')

    grasp_method = GraspMethod(net, use_rgb, use_dep, model_path, None, input_size, device=device)

    for on in objs_num:
        for om in obj_mode:
            aver_ap = 0.
            for f in friction:
    
                success = 0 # 记录成功的抓取次数
                all = 0     # 记录总抓取次数

                pose_path = os.path.join(poses_path, om, '{:d}'.format(on))     # 场景位姿路径
                model_list_path = os.path.join(model_lists_path, 'eval_{}.txt'.format(om))  # 模型列表路径
                # 获取
                for pose_mat in os.listdir(pose_path):
                    print('pose_mat: ', pose_mat)
                    # 运行仿真抓取
                    _success, _all = run(grasp_method, obj_model_path, model_list_path, os.path.join(pose_path, pose_mat), f)
                    success += _success
                    all += _all
                    print('\n>>>>>>>>>>>>>>>>>>>> AP: {}/{}={}'.format(success, all, success/all)) 
                
                # 记录结果
                result = '{}_{:02d}objs_{}_{:.1f}_AP: {:.4f} \n'.format(net, on, om, f, success/all)
                aver_ap += success/all
                print('write ...', result)
                file = open(save_txt, 'a')
                file.write(result)
                file.close()

            aver_ap /= len(friction)
            file = open(save_txt, 'a')
            file.write('{}_{:02d}objs_{}_aver_AP = {:.4f}\n\n'.format(net, on, om, aver_ap))
            file.close()
                
    
    
