'''
Description: 初始化抓取检测方法类，计算6-DOF抓取位姿
Author: wangdx
Date: 2022-03-22 19:20:48
LastEditTime: 2022-03-31 11:29:13
'''

import random
import time
import cv2
import numpy as np
import math
import os
import open3d as o3d
from skimage.draw import line
from graspnetAPI import GraspGroup
from grasp_methods.sgdn.sgdn import SGDN, drawGrasps, drawRect
from utils.camera import Camera
import utils.tool as tool
from scripts.dataset.generate_graspFig import FINGER_L1, FINGER_L2, CAMERA_HEIGHT, GRASP_MAX_W, ptsOnRotateRect
# from grasp_methods.graspNet.graspNet import GraspNetMethod
from utils.rigid_transformations import RigidTransform
from utils import transformations

GRASP_W = 0.08  # dexnet、sample方法的抓取宽度


def collision_detection(pt, dep, angle, depth_map, finger_l1, finger_l2):
    """
    碰撞检测
    pt: (row, col)
    angle: 抓取角 弧度
    depth_map: 深度图
    finger_l1 l2: 像素长度

    return:
        True: 无碰撞
        False: 有碰撞
    """
    row, col = pt

    # 两个点
    row1 = int(row - finger_l2 * math.sin(angle))
    col1 = int(col + finger_l2 * math.cos(angle))
    
    # 在截面图上绘制抓取器矩形
    # 检测截面图的矩形区域内是否有1
    rows, cols = ptsOnRotateRect([row, col], [row1, col1], finger_l1)

    if np.min(depth_map[rows, cols]) > dep:   # 无碰撞
        return True
    return False    # 有碰撞

def getGraspDepth(camera_depth, grasp_row, grasp_col, grasp_angle, grasp_width, finger_l1, finger_l2):
    """
    根据深度图像及抓取角、抓取宽度，计算最大的无碰撞抓取深度（相对于物体表面的下降深度）
    此时抓取点为深度图像的中心点
    camera_depth: 位于抓取点正上方的相机深度图
    grasp_angle：抓取角 弧度
    grasp_width：抓取宽度 像素
    finger_l1 l2: 抓取器尺寸 像素长度

    return: 抓取深度，相对于物体表面
    """
    # grasp_row = int(camera_depth.shape[0] / 2)
    # grasp_col = int(camera_depth.shape[1] / 2)
    # 首先计算抓取器两夹爪的端点
    k = math.tan(grasp_angle)

    grasp_width /= 2
    if k == 0:
        dx = grasp_width
        dy = 0
    else:
        dx = k / abs(k) * grasp_width / pow(k ** 2 + 1, 0.5)
        dy = k * dx
    
    pt1 = (int(grasp_row - dy), int(grasp_col + dx))
    pt2 = (int(grasp_row + dy), int(grasp_col - dx))

    # 下面改成，从抓取线上的最高点开始向下计算抓取深度，直到碰撞或达到最大深度
    rr, cc = line(pt1[0], pt1[1], pt2[0], pt2[1])   # 获取抓取线路上的点坐标
    min_depth = np.min(camera_depth[rr, cc])
    # print('camera_depth[grasp_row, grasp_col] = ', camera_depth[grasp_row, grasp_col])

    grasp_depth = 0.003
    while grasp_depth < 0.05:
        if not collision_detection(pt1, min_depth+grasp_depth, grasp_angle, camera_depth, finger_l1, finger_l2):
            return grasp_depth - 0.003
        if not collision_detection(pt2, min_depth+grasp_depth, grasp_angle + math.pi, camera_depth, finger_l1, finger_l2):
            return grasp_depth - 0.003
        grasp_depth += 0.003

    return grasp_depth
    # return grasp_depth - camera_depth[grasp_row, grasp_col]


class GraspPose:
    """
    一个抓取位姿

    原点为机械手的末端中心，y轴指向任一个夹爪，x轴垂直手掌，z轴与xy轴正交
    """
    def __init__(self, pos=np.zeros(3), rotMat=np.eye(3), width=0, frame='world'):
        """
        创建抓取位姿

        pos: [ndaray, 3, np.float]
            抓取点坐标
        rotMat: [ndaray, 3*3, np.float]
            旋转矩阵
        width: float
            抓取宽度, 单位m
        depth: float
            抓取深度, 单位m
        frame: str
            抓取位姿所在的坐标系
        """
        self.center = pos
        self.rotate_mat = rotMat
        self.width = width
        self.frame = frame
        self.axis = self._axis

    def from_object_pos(self, pos_obj, rotMat, width, depth, frame='world'):
        """
        从物体表面的点创建抓取位姿
        物体表面的点指从像素点投影到物体表面的点

        pos_obj: [ndaray, 3, np.float]
            位于物体表面的点坐标
        rotMat: [ndaray, 3*3, np.float]
            旋转矩阵
        width: float
            抓取宽度, 单位m
        depth: float
            抓取深度,相对于物体表面, 单位m
        frame: str
            抓取位姿所在的坐标系
        """
        self.position_face = pos_obj
        self.rotate_mat = rotMat
        self.width = width
        self.depth = depth
        self.frame = frame
        self.center = self.getCenter()  # 机械手末端中心
        self.axis = self._axis
    

    def from_endpoints(self, p1, p2, frame='world'):
        """
        从抓取的两个端点创建抓取位姿

        p1: np.ndarray shape=(3,)
        alpha: 沿y轴的旋转角 弧度
        """
        self.p1 = p1
        self.p2 = p2
        self.center = (self.p1 + self.p2) / 2
        self.width = np.linalg.norm(p1 - p2)
        self.frame = frame
        grasp_axis = p2 - p1
        self.axis = grasp_axis / np.linalg.norm(grasp_axis)
        self.rotate_mat = self.unrotated_full_axis

    @property
    def unrotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. 
        X axis points out of the gripper palm along the 0-degree approach direction, 
        Y axis points between the jaws, 
        and the Z axs is orthogonal.
        
        本函数输出的抓取旋转,没有考虑approach_angle (即沿抓取坐标系y轴的旋转)

        保证生成的抓取位姿的z轴在frame坐标系z轴上的投影小于0，即在pybullet环境中，机械手向下抓取。

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        grasp_axis_y = self.axis
        grasp_axis_x = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0]) # 使grasp的x轴位于obj的x-y平面上, 即二指机械手的平面 垂直于 桌面 进行抓取
        if np.linalg.norm(grasp_axis_x) == 0:   # 2范数
            grasp_axis_x = np.array([1,0,0])
        grasp_axis_x = grasp_axis_x / np.linalg.norm(grasp_axis_x)    # 归一化
        grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)

        if grasp_axis_z[2] > 0:
            grasp_axis_x = np.array([-grasp_axis_y[1], grasp_axis_y[0], 0]) # 使grasp的x轴位于obj的x-y平面上, 即二指机械手的平面 垂直于 桌面 进行抓取
            if np.linalg.norm(grasp_axis_x) == 0:   # 2范数
                grasp_axis_x = np.array([1,0,0])
            grasp_axis_x = grasp_axis_x / np.linalg.norm(grasp_axis_x)    # 归一化
            grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)
        
        R = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]  # 先把每个(3,)向量reshape成(3,1),再沿列方向合并
        return R


    @staticmethod
    def _get_rotation_matrix_y(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
        return R
    
    @property
    def _pts(self):
        """
        计算抓取的两个端点

        抓取中心点 - 抓取宽度乘以axis        抓取位姿y轴上的
        """
        self.p1 = self.center - (self.width / 2) * self._axis
        self.p1 = self.center + (self.width / 2) * self._axis

    @property
    def zz_axis(self):
        """:obj:`numpy.ndarray` of float: 不考虑平移，to_frame的z轴在from_frame的z轴的投影
        """
        return self.rotate_mat[2,2]

    @property
    def _axis(self):
        """
        抓取轴 / 两端点的差 / grasp坐标系的y轴在frame坐标系中的投影 / 只考虑旋转，grasp坐标系的(0,1,0)在frame坐标系中的坐标
        """
        return self.rotate_mat[:,1]

    def getCenter(self):
        """
        计算机械手末端中心的点
        """
        # 设 g为物体表面的抓取位姿，g1为实际的抓取位姿，c为相机坐标系
        # T_c_g1 = T_c_g * T_g_g1   T_c_g已知
        T_g_g1 = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0],
            [0, 0, 1, self.depth],
            [0, 0, 0, 1]
        ])
        T_c_g = tool.getTransformMat(self.position_face, self.rotate_mat)
        T_c_g1 = np.matmul(T_c_g, T_g_g1)

        center = T_c_g1[:3, 3].reshape((3,))
        return center

    @property
    def quaternion(self):
        """
        return:
            四元数
        """
        return transformations.quaternion_from_matrix(self.rotate_mat)

    @property
    def rigidTransform(self):
        """
        返回从'frame'到 grasp 的 rigidTransform
        """
        return RigidTransform(rotation=self.rotate_mat, translation=self.center, 
                              from_frame=self.frame, to_frame='grasp')
    
    def transform(self, rigidTransform:RigidTransform):
        """
        将抓取位姿转换到 rigidTransform 的 from_frame 下
        """
        assert rigidTransform.to_frame == self.frame
        T_frame_grasp = RigidTransform(rotation=self.rotate_mat, translation=self.center, from_frame=self.frame, to_frame='grasp')
        T_newframe_grasp = rigidTransform.dot(T_frame_grasp)
        self.rotate_mat = T_newframe_grasp.rotation
        self.center = T_newframe_grasp.translation
        self.frame = T_newframe_grasp.from_frame

    
    def transformPose(self, transMat):
        """
        根据输入的转换矩阵将抓取位姿转换到相应的坐标系下(左乘)
        transMat: [ndarray, (4,4), np.float]
            相机坐标系相对于其他坐标系e的转换矩阵
        """
        # T_e_g = T_e_c * T_c_g
        rotate_mat = self.rotate_mat
        # 转换实际抓取位姿
        T_c_g1 = tool.getTransformMat(self.position, rotate_mat)
        T_e_g1 = np.matmul(transMat, T_c_g1)
        self.position = T_e_g1[:3, 3].reshape((3,))
        self.rotate_mat = T_e_g1[:3, :3]
        # print('self.rotate_mat 1 = ', self.rotate_mat)

        # 转换物体表面抓取位姿
        T_c_g = tool.getTransformMat(self.position_face, rotate_mat)
        T_e_g = np.matmul(transMat, T_c_g)
        self.position_face = T_e_g[:3, 3].reshape((3,))
        self.rotate_mat = T_e_g[:3, :3]
    


class GraspPoses:
    """
    多个抓取位姿
    """
    def __init__(self):
        self.grasp_poses = []

    def append(self, gp:GraspPose):
        self.grasp_poses.append(gp)
        pass
    
    def __len__(self):
        """
        返回抓取位姿的数量
        """
        return len(self.grasp_poses)

    def __getitem__(self, idx):
        """
        返回idx索引的抓取位姿
        """
        return self.grasp_poses[idx]


class GraspMethod:
    def __init__(self, method:str,  # 抓取检测方法名称
                       use_rgb,
                       use_dep,
                       model:str,   # 模型路径
                       camera:Camera,   # 相机
                       input_size=384,  # 图像的截取尺寸
                       angle_cls=18,    # 抓取角分类数量
                       width=None,
                       device='cpu'): # 设备 cpu/cuda:?
        """
        初始化抓取检测方法
        """
        self.init_param()
        self.method = method
        self.camera = camera
        self.use_rgb = use_rgb
        self.use_dep = use_dep
        self.input_size = input_size
        self.width = width

        if method in self.sgdn:
            print('Initing ', method)
            self.grasp_method = SGDN(method, self.use_rgb, self.use_dep, model, device, angle_cls=angle_cls)    # 初始化sgdn
        elif method == self.gqcnn:
            pass
        elif method == self.fcgqcnn:
            pass
        elif method == self.graspNet:
            # 初始化graspNet
            print('Initing ', method)
            self.grasp_method = GraspNetMethod(model, device)
            pass
        else:
            print('Not init network.')

    def setcamera(self, camera:Camera):
        """
        设置 camera
        """
        self.camera = camera


    def init_param(self):
        """
        初始化超参数
        """
        self.sgdn = ['affga', 'danet', 'deeplab', 'ggcnn2', 'grcnn', 'segnet', 'stdc', 'unet']
        self.gqcnn = 'gqcnn'
        self.fcgqcnn = 'fcgqcnn'
        self.graspNet = 'graspNet'

    def getConfidentMap(self, im_rgb, im_dep):
        """
        返回SGDN预测的抓取置信度图

        img_rgb: rgb图像 np.array (h, w, 3)
        img_dep: 深度图 np.array (h, w)

        return:
            confident_map: 抓取置信度图  抓取点置信度 * 抓取角置信度 (angle_bins, h, w)
        """
        assert self.method in self.sgdn
        confident_map = self.grasp_method.confidentMap(im_rgb, im_dep, input_size=self.input_size)
        return confident_map
    

    def drawPlanerGrasps(self, im_rgb, im_dep, save_path, pose_path):
        """
        保存平面抓取位姿
        """
        graspFigs = self.grasp_method.predict(im_rgb, im_dep, input_size=self.input_size, mode='peak', thresh=0.3, peak_dist=1)
        print('抓取位姿数量：', len(graspFigs))
        # try:
        #     if len(graspFigs) < 50:
        #         graspFigs_3 = self.grasp_method.predict(im_rgb, im_dep, input_size=self.input_size, mode='peak', thresh=0.3, peak_dist=1)
        #         graspFigs += graspFigs_3[:50-len(graspFigs)]
        # except:
        #     pass

        # 将graspFigs的抓取宽度转为像素值
        for i in range(len(graspFigs)):
            row, col, _, grasp_width, _ = graspFigs[i]
            graspFigs[i][3] = self.camera.length_TO_pixels(grasp_width, im_dep[row, col])

        # 保存可视化结果
        cv2.imwrite(os.path.join(save_path, os.path.basename(pose_path)[:-4]+'_rgb.png'), im_rgb)   # rgb
        cv2.imwrite(os.path.join(save_path, os.path.basename(pose_path)[:-4]+'_dep.png'), tool.depth2RGB(im_dep))   # depth

        im_grasp_50 = im_rgb.copy()
        im_grasp_1 = im_rgb.copy()
        drawGrasps(im_grasp_50, graspFigs[:50], mode='line')  # 绘制预测结果
        drawGrasps(im_grasp_1, graspFigs[:1], mode='line')  # 绘制预测结果
        x1 = int((640 - self.input_size) / 2)
        y1 = int((480 - self.input_size) / 2)
        # rect = [x1, y1, x1 + input_size, y1 + input_size]
        # drawRect(im_grasp, rect)
        cv2.imwrite(os.path.join(save_path, os.path.basename(pose_path)[:-4]+'_grasp_top50.png'), im_grasp_50)   # top-50抓取
        cv2.imwrite(os.path.join(save_path, os.path.basename(pose_path)[:-4]+'_grasp_top1.png'), im_grasp_1)   # top-1抓取

        crop_rgb = im_rgb[y1:y1 + self.input_size, x1:x1 + self.input_size, :]
        able_map = self.grasp_method.able_map(im_rgb, im_dep, self.input_size)
        able_map = tool.depth2RGB(able_map)
        able_map_concat = cv2.addWeighted(crop_rgb, 0.5, able_map, 0.5, 0)
        cv2.imwrite(os.path.join(save_path, os.path.basename(pose_path)[:-4]+'_map.png'), able_map_concat)


    def get6DOFGraspPose(self, im_rgb, im_dep):
        """
        获取置信度最高的6DOF抓取位姿
        
        im_rgb: RGB图像
        im_dep: 深度图像 单位m

        return: 
            tuple:{'grasp_pos':, 'grasp_euler':, 'grasp_width':}
            grasp_pos: 抓取点坐标 m（世界坐标系）
            grasp_euler: 旋转欧拉角 弧度（世界坐标系）
            grasp_width: 抓取宽度 m
        """
        if self.method in self.sgdn:
            graspFigs = self.grasp_method.predict(im_rgb, im_dep, input_size=self.input_size, mode='max')	# SGDN预测

            # 解码抓取配置
            grasp_n = 0
            row, col, grasp_angle, grasp_width, grasp_depth = graspFigs[grasp_n]
            grasp_width += 0.02
            if self.width is not None:
                grasp_width = self.width

            # print('grasp_angle = ', grasp_angle)

            # 计算抓取深度
            grasp_width_pixels = self.camera.length_TO_pixels(grasp_width, im_dep[row, col])
            finger_l1_pixels = self.camera.length_TO_pixels(FINGER_L1, im_dep[row, col])
            finger_l2_pixels = self.camera.length_TO_pixels(FINGER_L2, im_dep[row, col])
            grasp_depth = getGraspDepth(im_dep, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels, finger_l2_pixels)
            grasp_depth = max(grasp_depth, 0.01)

            # 记录抓取位姿
            grasp_pos_face = self.camera.img2camera([col, row], im_dep[row, col])   # 物体表面的抓取点
            grasp_rotMat = tool.eulerAnglesToRotationMatrix([0, 0, np.pi/2-grasp_angle])   # 旋转矩阵
            gp = GraspPose()
            gp.from_object_pos(grasp_pos_face, grasp_rotMat, grasp_width, grasp_depth, frame='camera')  # 相机坐标系下

        # elif self.method == self.gqcnn:
        #     pass
        # elif self.method == self.fcgqcnn:
        #     pass

        # TODO: 需要修改
        elif self.method == self.graspNet:
            # 基于GraspNet机械手坐标系的抓取位姿(相机坐标系下，但是轴的顺序与本程序的机械手轴顺序不同)
            # grasp_center = gg[0].translation
            # rotation_matrix = gg[0].rotation_matrix
            # grasp_width = gg[0].width
            # grasp_depth = gg[0].depth
            gg = self.grasp_method.predict(im_rgb, im_dep, input_size=self.input_size, intrinsic=self.camera.IntrinsicMatrix)
            """
            graspNet的机械手坐标系 -> panda机械手坐标系: euler = [0, -math.pi/2, math.pi/2]     # TODO: 重新确认
            [0, -math.pi/2, math.pi/2]的逆旋转就是[-math.pi/2, 0, -math.pi/2]
            """
            offset = tool.eulerAnglesToRotationMatrix([-math.pi/2, 0, -math.pi/2])
            grasp_rotMat = np.matmul(gg[0].rotation_matrix, offset)

            gp = GraspPose(gg[0].translation, grasp_rotMat, gg[0].width, gg[0].depth)  # 相机坐标系下

        # 测试用例
        # grasp_pos = [0, 0, 0.1]   # 抓取点坐标（世界坐标系）
        # grasp_euler = [math.pi/4, math.pi/4, 0]   # 欧拉角
        # grasp_width = 0

        return gp

    def get6DOFGraspPose_in(self, im_rgb, im_dep, im_mask):
        """
        获取置信度最高的6DOF抓取位姿

        *** 只获取物体区域内的 ***
        
        im_rgb: RGB图像
        im_dep: 深度图像 单位m

        return: 
            tuple:{'grasp_pos':, 'grasp_euler':, 'grasp_width':}
            grasp_pos: 抓取点坐标 m（世界坐标系）
            grasp_euler: 旋转欧拉角 弧度（世界坐标系）
            grasp_width: 抓取宽度 m
        """
        if self.method in self.sgdn:
            graspFigs = self.grasp_method.predict_in(im_rgb, im_dep, im_mask, input_size=self.input_size, mode='max')	# SGDN预测

            # 解码抓取配置
            grasp_n = 0
            row, col, grasp_angle, grasp_width, grasp_depth = graspFigs[grasp_n]
            grasp_width += 0.02
            
            if self.width is not None:
                grasp_width = self.width

            print('grasp_width = ', grasp_width)

            # 计算抓取深度
            grasp_width_pixels = self.camera.length_TO_pixels(grasp_width, im_dep[row, col])
            finger_l1_pixels = self.camera.length_TO_pixels(FINGER_L1, im_dep[row, col])
            finger_l2_pixels = self.camera.length_TO_pixels(FINGER_L2, im_dep[row, col])
            grasp_depth = getGraspDepth(im_dep, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels, finger_l2_pixels)
            grasp_depth = max(grasp_depth, 0.01)

            # 记录抓取位姿
            grasp_pos_face = self.camera.img2camera([col, row], im_dep[row, col])   # 物体表面的抓取点
            grasp_rotMat = tool.eulerAnglesToRotationMatrix([0, 0, np.pi/2-grasp_angle])   # 旋转矩阵
            gp = GraspPose()
            gp.from_object_pos(grasp_pos_face, grasp_rotMat, grasp_width, grasp_depth, frame='camera')  # 相机坐标系下

        # elif self.method == self.gqcnn:
        #     pass
        # elif self.method == self.fcgqcnn:
        #     pass

        # TODO: 需要修改
        elif self.method == self.graspNet:
            # 基于GraspNet机械手坐标系的抓取位姿(相机坐标系下，但是轴的顺序与本程序的机械手轴顺序不同)
            # grasp_center = gg[0].translation
            # rotation_matrix = gg[0].rotation_matrix
            # grasp_width = gg[0].width
            # grasp_depth = gg[0].depth
            gg = self.grasp_method.predict(im_rgb, im_dep, input_size=self.input_size, intrinsic=self.camera.IntrinsicMatrix)
            """
            graspNet的机械手坐标系 -> panda机械手坐标系: euler = [0, -math.pi/2, math.pi/2]     # TODO: 重新确认
            [0, -math.pi/2, math.pi/2]的逆旋转就是[-math.pi/2, 0, -math.pi/2]
            """
            offset = tool.eulerAnglesToRotationMatrix([-math.pi/2, 0, -math.pi/2])
            grasp_rotMat = np.matmul(gg[0].rotation_matrix, offset)

            gp = GraspPose(gg[0].translation, grasp_rotMat, gg[0].width, gg[0].depth)  # 相机坐标系下

        # 测试用例
        # grasp_pos = [0, 0, 0.1]   # 抓取点坐标（世界坐标系）
        # grasp_euler = [math.pi/4, math.pi/4, 0]   # 欧拉角
        # grasp_width = 0

        return gp

    def getRandom6DOFGraspPose(self, im_dep):
        """
       将平面抓取配置转换为6DOF抓取位姿
        
        grasp_config: 平面抓取配置 (row, col, angle, width, depth) angle为弧度，width 单位为米, depth 单位为米
        im_rgb: RGB图像
        im_dep: 深度图像 单位m

        return: 
            tuple:{'grasp_pos':, 'grasp_euler':, 'grasp_width':}
            grasp_pos: 抓取点坐标 m（世界坐标系）
            grasp_euler: 旋转欧拉角 弧度（世界坐标系）
            grasp_width: 抓取宽度 m
        """

        # 随机初始化抓取配置
        row = random.randint(200, 400)
        col = random.randint(200, 500)
        grasp_angle = random.random() * np.pi
        grasp_width = 0.1
        grasp_width_pixels = 50
        grasp_depth = 0.02

        im_grasp = drawGrasps(tool.depth2RGB(im_dep), [[row, col, grasp_angle, grasp_width_pixels]])
        cv2.imshow('im_grasp', im_grasp)
        cv2.waitKey()

        # 记录抓取位姿
        grasp_pos_face = self.camera.img2camera([col, row], im_dep[row, col])   # 物体表面的抓取点
        grasp_rotMat = tool.eulerAnglesToRotationMatrix([0, 0, np.pi/2-grasp_angle])   # 旋转矩阵
        gp = GraspPose()
        gp.from_object_pos(grasp_pos_face, grasp_rotMat, grasp_width, grasp_depth, frame='camera')  # 相机坐标系下

        return gp
    
    def get6DOFGraspPose_fromPlaneConfig(self, grasp_config, im_dep):
        """
       将平面抓取配置转换为6DOF抓取位姿
        
        grasp_config: 平面抓取配置 (row, col, angle, width, depth) angle为弧度，width 单位为米, depth 单位为米
        im_rgb: RGB图像
        im_dep: 深度图像 单位m

        return: 
            tuple:{'grasp_pos':, 'grasp_euler':, 'grasp_width':}
            grasp_pos: 抓取点坐标 m（世界坐标系）
            grasp_euler: 旋转欧拉角 弧度（世界坐标系）
            grasp_width: 抓取宽度 m
        """

        # 解码抓取配置
        row, col, grasp_angle, grasp_width, grasp_depth = grasp_config
        if grasp_depth is None:
            # 计算抓取深度
            grasp_width_pixels = self.camera.length_TO_pixels(grasp_width, im_dep[row, col])
            finger_l1_pixels = self.camera.length_TO_pixels(FINGER_L1, im_dep[row, col])
            finger_l2_pixels = self.camera.length_TO_pixels(FINGER_L2, im_dep[row, col])
            grasp_depth = getGraspDepth(im_dep, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels, finger_l2_pixels)
            grasp_depth = max(grasp_depth, 0.01)

        # 记录抓取位姿
        grasp_pos_face = self.camera.img2camera([col, row], im_dep[row, col])   # 物体表面的抓取点
        grasp_rotMat = tool.eulerAnglesToRotationMatrix([0, 0, np.pi/2-grasp_angle])   # 旋转矩阵
        gp = GraspPose()
        gp.from_object_pos(grasp_pos_face, grasp_rotMat, grasp_width, grasp_depth, frame='camera')  # 相机坐标系下

        return gp
    

    def get6DOFGraspPoses(self, im_rgb, im_dep, num=50):
        """
        计算NMS后的top-n个6DOF抓取位姿
        
        im_rgb: RGB图像
        im_dep: 深度图像 单位m
        num: 输出的抓取位姿数量

        return: 
            tuple:{'grasp_pos':, 'grasp_euler':, 'grasp_width':}
            grasp_pos: 抓取点坐标 m（世界坐标系）
            grasp_euler: 旋转欧拉角 弧度（世界坐标系）
            grasp_width: 抓取宽度 m
        """
        if self.method in self.sgdn:
            graspFigs = self.grasp_method.predict(im_rgb, im_dep, input_size=self.input_size, mode='nms')	# SGDN预测

            gps = GraspPoses()
            for i in range(len(graspFigs)):
                # 解码抓取配置
                row, col, grasp_angle, grasp_width, grasp_depth = graspFigs[i]
                grasp_width += 0.02
                if self.width is not None:
                    grasp_width = self.width

                # 计算抓取深度
                grasp_width_pixels = self.camera.length_TO_pixels(grasp_width, im_dep[row, col])
                finger_l1_pixels = self.camera.length_TO_pixels(FINGER_L1, im_dep[row, col])
                finger_l2_pixels = self.camera.length_TO_pixels(FINGER_L2, im_dep[row, col])
                grasp_depth = getGraspDepth(im_dep, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels, finger_l2_pixels)
                grasp_depth = max(grasp_depth, 0.01)

                # 记录抓取位姿
                grasp_pos_face = self.camera.img2camera([col, row], im_dep[row, col])   # 物体表面的抓取点
                grasp_rotMat = tool.eulerAnglesToRotationMatrix([0, 0, np.pi/2-grasp_angle])   # 旋转矩阵
                gp = GraspPose()
                gp.from_object_pos(grasp_pos_face, grasp_rotMat, grasp_width, grasp_depth, frame='camera')  # 相机坐标系下

                gps.append(gp)
    
        # elif self.method == self.gqcnn:
        #     pass
        # elif self.method == self.fcgqcnn:
        #     pass

        # TODO: 需要修改
        elif self.method == self.graspNet:
            # 基于GraspNet机械手坐标系的抓取位姿(相机坐标系下，但是轴的顺序与本程序的机械手轴顺序不同)
            gg = self.grasp_method.predict(im_rgb, im_dep, input_size=self.input_size, intrinsic=self.camera.IntrinsicMatrix, num=num)

            gps = GraspPoses()
            for i in range(len(gg)):
                """
                graspNet的机械手坐标系 -> panda机械手坐标系: euler = [0, -math.pi/2, math.pi/2]
                [0, -math.pi/2, math.pi/2]的逆旋转就是[-math.pi/2, 0, -math.pi/2]
                """
                offset = tool.eulerAnglesToRotationMatrix([-math.pi/2, 0, -math.pi/2])
                grasp_rotMat = np.matmul(gg[i].rotation_matrix, offset)

                gp = GraspPose(gg[i].translation, grasp_rotMat, gg[i].width, gg[i].depth)  # 相机坐标系下
                gps.append(gp)

        # 测试用例
        # grasp_pos = [0, 0, 0.1]   # 抓取点坐标（世界坐标系）
        # grasp_euler = [math.pi/4, math.pi/4, 0]   # 欧拉角
        # grasp_width = 0

        return gps

    def get6DOFGraspPoses_in(self, im_rgb, im_dep, im_mask, num=50):
        """
        计算NMS后的top-n个6DOF抓取位姿
        
        im_rgb: RGB图像
        im_dep: 深度图像 单位m
        num: 输出的抓取位姿数量

        return: 
            tuple:{'grasp_pos':, 'grasp_euler':, 'grasp_width':}
            grasp_pos: 抓取点坐标 m（世界坐标系）
            grasp_euler: 旋转欧拉角 弧度（世界坐标系）
            grasp_width: 抓取宽度 m
        """
        if self.method in self.sgdn:
            graspFigs = self.grasp_method.predict_in(im_rgb, im_dep, im_mask, input_size=self.input_size, mode='nms')	# SGDN预测

            gps = GraspPoses()
            for i in range(len(graspFigs)):
                # 解码抓取配置
                row, col, grasp_angle, grasp_width, grasp_depth = graspFigs[i]
                grasp_width += 0.02
                if self.width is not None:
                    grasp_width = self.width

                # 计算抓取深度
                grasp_width_pixels = self.camera.length_TO_pixels(grasp_width, im_dep[row, col])
                finger_l1_pixels = self.camera.length_TO_pixels(FINGER_L1, im_dep[row, col])
                finger_l2_pixels = self.camera.length_TO_pixels(FINGER_L2, im_dep[row, col])
                grasp_depth = getGraspDepth(im_dep, row, col, grasp_angle, grasp_width_pixels, finger_l1_pixels, finger_l2_pixels)
                grasp_depth = max(grasp_depth, 0.01)

                # 记录抓取位姿
                grasp_pos_face = self.camera.img2camera([col, row], im_dep[row, col])   # 物体表面的抓取点
                grasp_rotMat = tool.eulerAnglesToRotationMatrix([0, 0, np.pi/2-grasp_angle])   # 旋转矩阵
                gp = GraspPose()
                gp.from_object_pos(grasp_pos_face, grasp_rotMat, grasp_width, grasp_depth, frame='camera')  # 相机坐标系下

                gps.append(gp)
    
        # elif self.method == self.gqcnn:
        #     pass
        # elif self.method == self.fcgqcnn:
        #     pass

        # TODO: 需要修改
        elif self.method == self.graspNet:
            # 基于GraspNet机械手坐标系的抓取位姿(相机坐标系下，但是轴的顺序与本程序的机械手轴顺序不同)
            gg = self.grasp_method.predict(im_rgb, im_dep, input_size=self.input_size, intrinsic=self.camera.IntrinsicMatrix, num=num)

            gps = GraspPoses()
            for i in range(len(gg)):
                """
                graspNet的机械手坐标系 -> panda机械手坐标系: euler = [0, -math.pi/2, math.pi/2]
                [0, -math.pi/2, math.pi/2]的逆旋转就是[-math.pi/2, 0, -math.pi/2]
                """
                offset = tool.eulerAnglesToRotationMatrix([-math.pi/2, 0, -math.pi/2])
                grasp_rotMat = np.matmul(gg[i].rotation_matrix, offset)

                gp = GraspPose(gg[i].translation, grasp_rotMat, gg[i].width, gg[i].depth)  # 相机坐标系下
                gps.append(gp)

        # 测试用例
        # grasp_pos = [0, 0, 0.1]   # 抓取点坐标（世界坐标系）
        # grasp_euler = [math.pi/4, math.pi/4, 0]   # 欧拉角
        # grasp_width = 0

        return gps

    
    def draw4DOFGraspPose(self):
        """
        在图像中绘制4DOF抓取位姿
        """
        # 绘制抓取配置
        # im_rgb = tool.depth2Gray3(camera_depth)
        # im_grasp = drawGrasps(im_rgb, [[row, col, grasp_angle, grasp_width_pixels]], mode='line')  # 绘制预测结果
        # x1 = int((640 - input_size) / 2)
        # y1 = int((480 - input_size) / 2)
        # rect = [x1, y1, x1 + input_size, y1 + input_size]
        # drawRect(im_grasp, rect)
        # cv2.imshow('im_grasp', im_grasp)
        # cv2.waitKey(30)
        pass

    def drawPCL(self, im_rgb, im_dep):
        """
        绘制点云
        """
        # 根据input_size生成mask
        workspace_mask = np.zeros(im_rgb.shape[:2], dtype=np.bool)
        x1 = int((im_rgb.shape[1] - self.input_size) / 2)
        y1 = int((im_rgb.shape[0] - self.input_size) / 2)
        workspace_mask[y1:y1+self.input_size, x1:x1+self.input_size] = 1

        im_rgb = np.array(im_rgb[:, :, ::-1], dtype=np.float32) / 255.0
        pcl = self.camera.create_point_cloud(im_rgb, im_dep, workspace_mask)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=800, height=600)
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(os.path.join(os.curdir, 'grasp_methods/view.json'))
        vis.add_geometry(pcl)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()
        vis.destroy_window()

    # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
    # plot_gripper_pro_max(self.translation, self.rotation_matrix, self.width, self.depth, score = self.score, color = color)
    def draw6DOFGraspPose(self, im_rgb, im_dep, gp:GraspPose):
        """
        在点云中绘制6DOF抓取位姿
        """
        # 根据input_size生成mask
        workspace_mask = np.zeros(im_rgb.shape[:2], dtype=np.bool)
        x1 = int((im_rgb.shape[1] - self.input_size) / 2)
        y1 = int((im_rgb.shape[0] - self.input_size) / 2)
        workspace_mask[y1:y1+self.input_size, x1:x1+self.input_size] = 1

        im_rgb = np.array(im_rgb[:, :, ::-1], dtype=np.float32) / 255.0
        pcl = self.camera.create_point_cloud(im_rgb, im_dep, workspace_mask)

        # 调整旋转
        # offset = tool.eulerAnglesToRotationMatrix([0, -math.pi/2, math.pi/2])
        offset = tool.eulerAnglesToRotationMatrix([0, -math.pi/2, 0])
        rotMat = np.matmul(gp.rotate_mat, offset)

        gg_array1 = np.array([1., gp.width, 0, 0])
        gg_array = np.concatenate([gg_array1, rotMat.reshape((9,)), gp.center, np.zeros((1,))])[np.newaxis, :]
        gg = GraspGroup(gg_array)
        
        grippers = gg.to_open3d_geometry_list()

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=800, height=600)
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(os.path.join(os.curdir, 'grasp_methods/view.json'))
        vis.add_geometry(pcl)
        vis.add_geometry(*grippers)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()
        vis.destroy_window()

        # 保存视角文件
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name='pcd', width=800, height=600)
        # vis.add_geometry(pcl)
        # vis.add_geometry(*grippers)
        # vis.run()  # user changes the view and press "q" to terminate
        # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters(os.path.join(os.curdir, 'grasp_methods/view.json'), param)
        # vis.destroy_window()

        # o3d.visualization.draw_geometries([pcl, *grippers])


    def draw6DOFGraspPoses(self, im_rgb, im_dep, gps:GraspPoses):
        """
        在点云中绘制多个6DOF抓取位姿
        """
        # 根据input_size生成mask
        workspace_mask = np.zeros(im_rgb.shape[:2], dtype=np.bool)
        x1 = int((im_rgb.shape[1] - self.input_size) / 2)
        y1 = int((im_rgb.shape[0] - self.input_size) / 2)
        workspace_mask[y1:y1+self.input_size, x1:x1+self.input_size] = 1

        im_rgb = np.array(im_rgb[:, :, ::-1], dtype=np.float32) / 255.0
        pcl = self.camera.create_point_cloud(im_rgb, im_dep, workspace_mask)

        gg_arrays = []
        for i in range(len(gps)):
            gp = gps[i]
            # 调整旋转
            # offset = tool.eulerAnglesToRotationMatrix([0, -math.pi/2, math.pi/2])
            offset = tool.eulerAnglesToRotationMatrix([0, -math.pi/2, 0])
            rotMat = np.matmul(gp.rotate_mat, offset)
            gg_array1 = np.array([1., gp.width, 0, 0])
            gg_arrays.append(np.concatenate([gg_array1, rotMat.reshape((9,)), gp.center, np.zeros((1,))]))

        gg = GraspGroup(np.array(gg_arrays))
        grippers = gg.to_open3d_geometry_list()
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd', width=800, height=600)
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(os.path.join(os.curdir, 'grasp_methods/view.json'))
        vis.add_geometry(pcl)
        for i in range(len(grippers)):
            vis.add_geometry(grippers[i])
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()
        vis.destroy_window()

        # o3d.visualization.draw_geometries([pcl, *grippers])


    