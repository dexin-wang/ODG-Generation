"""
虚拟环境文件
初始化虚拟环境，加载物体，渲染图像，保存图像

(待写) ！！ 保存虚拟环境状态，以便离线抓取测试
"""

import pybullet as p
import pybullet_data
import time
import math
import os
import glob
import random
import cv2
import numpy as np
import scipy.io as scio
import sys
# import scipy.stats as ss
# import skimage.transform as skt
sys.path.append(os.curdir)
from utils.mesh import Mesh
import utils.tool as tool
from utils.camera import Camera
from scripts.dataset.generate_graspFig import TABLE_IMG_HEIGHT, TABLE_IMG_WIDTH, unit, HEIGHT, WIDTH, CAMERA_HEIGHT
import utils.panda_sim_grasp_gripper as gripper_sim
from utils import transformations
from grasp_methods.grasp import GraspPose

nearPlane = 0.01
farPlane = 10
fov = 60    # 垂直视场 图像高tan(30) * 0.7 *2 = 0.8082903m
aspect = WIDTH / HEIGHT


class SimEnv(object):
    """
    虚拟环境类
    """
    def __init__(self, bullet_client, model_path, model_list=None, plane=0, load_tray=True, load_gripper=False, friction=1.0):
        """
        path: 模型路径
        """
        self.p = bullet_client
        self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[-4, 4, 10])    # 不显示控件
        self.p.setPhysicsEngineParameter(enableFileCaching=0)   # 不压缩加载的文件
        # self.p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0, 0, 0])
        self.p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.2])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -10) # 设置重力

        # 读取urdf列表
        self.urdfs_list = []
        if model_list is not None:
            with open(model_list, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if not line:
                        break
                    self.urdfs_list.append(os.path.join(model_path, line.strip('\n')))

        self.urdfs_id_bullet = []   # 存储由pybullet系统生成的模型id
        self.urdfs_id_list = []     # 存储模型在model_list文件列表中的索引
        self.urdfs_scale = []       # 记录模型的尺度
        self.urdfs_vis_xyz = []       # 记录模型的偏移坐标，在urdf中设置
        self.start_idx = 0
        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]
        
        self.trayId = None
        self.planeId = None
        self.camera = None
        if load_tray:
            self.loadTray()     # 加载托盘
        self.loadPlane(plane=plane)    # 加载地面
        self.movecamera()  # 加载相机
        self.loadRealSense()    # 加载相机模型
        if load_gripper:
            self.loadGripper(friction)


    def calViewMatrix(self, position, qua):
        '''相机视图矩阵

        position: (xyz)
        euler: (rpy)
        
        视图矩阵(ViewMatrix)，是右手系坐标系转换为左手系， 传递给OpenGL进行渲染。
        X -> Y
        Y -> -X
        Z -> Z 
        '''
        cameraEyePosition = position
        cameraQuaternion = qua
        # 四元数转旋转矩阵 3x3矩阵
        cameraRotationMatrix = self.p.getMatrixFromQuaternion(cameraQuaternion)
        cameraRotationMatrix = np.array(cameraRotationMatrix).reshape((3,3))
        # camera_focus_distance 相机对焦距离
        # 指的是在当前的姿态角下, 相机要对准摄像头前方多远距离的物体.
        # 摄像头镜头指向的方向为Z轴方向。
        
        # 计算摄像头对焦点的坐标
        # T_world2cam = [position, quaternion]
        # T_cam2target : cam是相机坐标系, target是相机拍摄点的目标坐标系
        # 与相机坐标系， 只存在Z轴方向上的平移关系, 平移距离为camera_focus_distance
        # T_world2target = T_world2cam * T_cam2target
        # pybullet.multiplyTransforms 返回的T_world2target 格式 (位置, 四元数)
        self.camera_focus_distance = 0.1
        cameraTargetTransform = self.p.multiplyTransforms(positionA=cameraEyePosition, orientationA=cameraQuaternion, positionB=[0, 0, self.camera_focus_distance], orientationB=[0,0,0,1])
        # cameraTargetTransform = pybullet.multiplyTransforms(positionA=cameraEyePosition, orientationA=cameraQuaternion, positionB=[0, 0,self.camera_focus_distance], orientationB=[0,0,0,1])        
        
        # cameraTargetPosition =[0, 0, 0] # 指向坐标系原点
        cameraTargetPosition = cameraTargetTransform[0]
        # 相机坐标系Y轴正方向, 单位向量
        cameraUpVector = cameraRotationMatrix[:, 1] * -1
        
        # viewMatrix:  我理解的视图矩阵，是相机在世界坐标系下的位姿？
        # cameraEyePosition: 相机在世界坐标系下位置
        # cameraTargetPosition: 希望相机正前方， 即Z轴朝向的位置(应该设置为Z轴的坐标)
        # cameraUpVector:  相机坐标系Y轴正方向.
        # v_z 与相机坐标系Z轴正方向同向
        # v_z = cameraTargetPosition - cameraEyePosition
        # u_y = 相机坐标系在y轴正方向在世界坐标系下的描述
        # u_y = cameraUpVector 
        # cameraTargetPosition 与cameraUpVector
        viewMatrix = self.p.computeViewMatrix(cameraEyePosition=cameraEyePosition,
                                                    cameraTargetPosition=cameraTargetPosition,
                                                    cameraUpVector=cameraUpVector)
        return viewMatrix

    def urdfs_num(self):
        """
        返回环境中已加载的物体个数(包括托盘和地面)
        return: int
        """
        return len(self.urdfs_id_bullet)
    
    def urdfs_obj_num(self):
        """
        返回环境中已加载的物体个数(不包括托盘和地面)
        return: int
        """
        return len(self.urdfs_id_bullet) - self.start_idx
    
    def urdf_list_num(self):
        """
        返回模型列表中模型的数量
        """
        return len(self.urdfs_list)

    def urdfs_path(self):
        """
        返回环境中已加载的物体的路径
        return: list
        """
        paths = []
        for idx in self.urdfs_id_list:
            paths.append(self.urdfs_list[idx])
        return paths

    def movecamera(self, pos=[0, 0, CAMERA_HEIGHT], euler=[np.pi, 0, 0], isRandom=False):
        """移动相机至指定位置 / 或移动至随机位姿(相机中轴面向世界坐标系原点)

        pos: 相机在世界坐标系中的xyz坐标 默认为[0,0,0.6]
        euler: 相机在世界坐标系中的欧拉角 (r,p,y) 默认为[pi,0,0]
        isRandom: True表示随机设置相机位姿；False表示按照给定的值设置位姿

        连续的坐标变换，以新的坐标系为基础再次变换
        """
        qua = self.p.getQuaternionFromEuler(euler)
        if isRandom:
            x_offset = random.random() * 0.4 - 0.2
            y_offset = random.random() * 0.4 - 0.2
            addPos = [x_offset, y_offset, 0]
            p_offset = math.tan(x_offset/pos[2]) * -1
            r_offset = math.tan(y_offset/pos[2])
            addEuler = [r_offset, p_offset, 0]
            
            add_qua = self.p.getQuaternionFromEuler(addEuler)
            pos, qua = self.p.multiplyTransforms(pos, qua, addPos, add_qua)
        
        # 设置camera实例
        self.camera = Camera(pos, self.p.getEulerFromQuaternion(qua))

        # 设置相机的投影矩阵和视角矩阵
        self.viewMatrix = self.calViewMatrix(pos, qua)
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        

    def sleep(self, n=100):
        t = 0
        while True:
            self.p.stepSimulation()
            t += 1
            if t == n:
                break

    def loadRealSense(self):
        """
        加载RealSense d435i 相机模型
        """
        path = 'models/realsense/d435i.urdf'
        self.p.loadURDF(path, [0, 0, CAMERA_HEIGHT+0.05], [0.707, 0.707, 0, 0], useFixedBase=True)

    def loadTray(self):
        """
        加载托盘
        """
        path = 'models/tray/tray.urdf'
        self.trayId = self.p.loadURDF(path, [0, 0, 0])
        # 记录信息
        self.urdfs_id_list += [-1]
        self.urdfs_id_bullet.append(self.trayId)
        inf = self.p.getVisualShapeData(self.trayId)[0]
        self.urdfs_scale.append(inf[3][0]) 
        self.urdfs_vis_xyz.append(inf[5])
        self.start_idx += 1
    
    def loadGripper(self, friction):
        """
        加载yumi机械手
        """
        self.gripper = gripper_sim.GripperSimAuto(p, [0, 0, 0], friction=friction)  # 初始化抓取器
        pos = np.array([0, 0, 0.7])
        r = transformations.euler_matrix(0, np.pi, 0)[:3, :3]
        grasppose = GraspPose(pos, r)
        self.gripper.resetGripperPose(grasppose)
    
    def loadPlane(self, plane=0):
        """
        加载地面
        plane: 0:蓝白网格地面，1:灰黄色地面
        """
        if plane == 0:
            path = 'plane.urdf'               # 蓝白格地面路径：D:\developers\anaconda3-5.2.0\envs\sim_grasp\Lib\site-packages\pybullet_data
        elif plane == 1:
            path = 'models/plane/plane.urdf'    # 橙白地面，生成数据集用
        elif plane < 0:
            return

        if path == 'plane.urdf':
            self.planeId = self.p.loadURDF(path, [0, 0, 0])
        else:
            self.planeId = self.p.loadURDF(path, [0, 0, 0], flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        # 记录信息
        self.urdfs_id_list += [-2]
        self.urdfs_id_bullet.append(self.planeId)
        inf = self.p.getVisualShapeData(self.planeId)[0]
        self.urdfs_scale.append(inf[3][0]) 
        self.urdfs_vis_xyz.append(inf[5])
        self.start_idx += 1

    def loadURDF(self, urdf_file:str, idx=-1, pos=None, euler=None, scale=1):
        """
        加载单个urdf模型
        当idx为负数时，加载的物体不计入模型列表

        urdf_file: urdf文件
        idx: 物体id， 等于-1时，采用file；否则加载模型列表中索引为idx的模型
        pos: 加载位置，如果为None，则随机位置
        euler: 加载欧拉角，如果为None，则随机欧拉角
        scale: 缩放倍数
        """
        # 获取物体文件
        if idx >= 0:
            urdf_file = self.urdfs_list[idx]
            self.urdfs_id_list += [idx]

        # 位置
        if pos is None:
            _p = 0.1
            pos = [random.uniform(-1 * _p, _p), random.uniform(-1 * _p, _p), random.uniform(0.1, 0.4)]

        # 方向
        if euler is None:
            euler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
        Ori = self.p.getQuaternionFromEuler(euler)

        # 加载物体
        flags = p.URDF_USE_IMPLICIT_CYLINDER
        if idx >= 0:
            mtl_files = glob.glob( os.path.join( os.path.dirname(self.urdfs_list[idx]), '*.mtl') )
        else:
            mtl_files = []
        if len(mtl_files) > 0:
            flags = flags | p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        urdf_id = self.p.loadURDF(urdf_file, pos, Ori, globalScaling=scale, flags=flags)
        print('urdf = ', urdf_file)

        if idx >= 0:
            # 记录信息
            self.urdfs_id_bullet.append(urdf_id)
            inf = self.p.getVisualShapeData(urdf_id)[0]
            self.urdfs_scale.append(inf[3][0]) 
            self.urdfs_vis_xyz.append(inf[5]) 
    
    def loadURDFs(self, start_idx, num, scale=1):
        """
        加载多个urdf模型

        idx: 从 urdf_list中第idx个模型开始加载
        num: 加载模型的个数
        scale: 加载进来模型的尺度
            float: 按照scale加载
            -1：按照 self.urdfs_scale[-1]加载；如果self.urdfs_scale为空，则按scale=1加载
        """
        assert start_idx >= 0 and start_idx < len(self.urdfs_list) and scale != 0
        
        if (start_idx + num - 1) > (len(self.urdfs_list) - 1):
            self.urdfs_id_list += list(range(start_idx, self.urdf_list_num())) + list(range(start_idx+num-self.urdf_list_num()))
            idxs = list(range(start_idx, self.urdf_list_num())) + list(range(start_idx+num-self.urdf_list_num()))
        else:
            self.urdfs_id_list += list(range(start_idx, start_idx+num))
            idxs = list(range(start_idx, start_idx+num))

        for idx in idxs:
            # 随机位置
            pos = 0.05
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), 0.3] 
            # basePosition = [0, 0, 0.1] # 固定位置

            # 随机方向
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            # baseOrientation = [0, 0, 0, 1]    # 固定方向
            
            # 确定scale
            if scale < 0:
                if len(self.urdfs_scale) > 0:
                    scale = self.urdfs_scale[-1]
                else:
                    scale = 1
            # 确定flags，当urdf目录下存在mtl时，使用mtl材质；没有则使用urdf材质
            flags = p.URDF_USE_IMPLICIT_CYLINDER
            mtl_files = glob.glob( os.path.join( os.path.dirname(self.urdfs_list[idx]), '*.mtl') )
            print('load urdf = ', self.urdfs_list[idx])
            print('mtl_files = ', mtl_files)
            if len(mtl_files) > 0:
                flags = flags | p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            # 加载物体
            urdf_id = self.p.loadURDF(self.urdfs_list[idx], basePosition, baseOrientation, flags=flags, globalScaling=scale, useFixedBase=False)
            self.urdfs_id_bullet.append(urdf_id)

            # 获取xyz和scale信息
            inf = self.p.getVisualShapeData(urdf_id)[0] # visual相对于link/joint的坐标，就是urdf中设置的xyz
            self.urdfs_scale.append(inf[3][0])
            self.urdfs_vis_xyz.append(inf[5])

            self.sleep(n=50)
         
    def loadURDFsWithPose(self, file):
        """
        加载多个urdf模型，并将位姿设置为预先保存的位姿

        file: 存储物体信息的mat文件路径 / np.ndarray,格式和mat文件一样
        *** !! 注意初始化该类时的model_list参数，需与此函数的path保存的物体索引，保存一致，否则会加载错误的物体。 !! ***
        """
        if isinstance(file, str):
            # 读取objsPose.mat文件
            if file.endswith('.mat'):
                path_ = file
            else:
                path_ = os.path.join(file, 'objsPose.mat')
            objsPose = scio.loadmat(path_)['A']
        elif isinstance(file, np.ndarray):
            objsPose = file

        num_urdf = objsPose.shape[0]
        for i in range(num_urdf):     
            urdf_path = self.urdfs_list[int(objsPose[i, 0])]
            basePosition = [objsPose[i, 1], objsPose[i, 2], objsPose[i, 3]]
            baseOrientation = [objsPose[i, 4], objsPose[i, 5], objsPose[i, 6], objsPose[i, 7]]
            # 加载物体
            # flags= p.URDF_MERGE_FIXED_LINKS or p.URDF_USE_IMPLICIT_CYLINDER
            flags= p.URDF_USE_IMPLICIT_CYLINDER
            mtl_files = glob.glob( os.path.join( os.path.dirname(urdf_path), '*.mtl') )
            if len(mtl_files) > 0:
                flags = flags | p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            # 加载物体
            urdf_id = self.p.loadURDF(urdf_path, basePosition, baseOrientation, flags=flags)
            self.urdfs_id_bullet.append(urdf_id)
            self.urdfs_id_list.append(int(objsPose[i, 0]))
            # 获取xyz和scale信息
            inf = self.p.getVisualShapeData(urdf_id)[0]
            self.urdfs_scale.append(inf[3][0]) 
            self.urdfs_vis_xyz.append(inf[5])

    def loadGraspLabelsFromMat(self, path, grasp_start_id=0):
        """
        从mat文件加载抓取标签
        """
        print('loading grasp labels ...')
        self.grasp_labels = []
        self.grasp_id = grasp_start_id   # 记录遍历到的抓取索引
        grasp_point_map = scio.loadmat(path + '/grasp_18/grasp_point_map.mat')['A'].astype(np.float64)   # (h, w)
        grasp_angle_map = scio.loadmat(path + '/grasp_18/grasp_angle_map.mat')['A'].astype(np.float64)   # (h, w, bin)
        grasp_width_map = scio.loadmat(path + '/grasp_18/grasp_width_map.mat')['A']                    # (h, w, bin)
        grasp_depth_map = scio.loadmat(path + '/grasp_18/grasp_depth_map.mat')['A']                    # (h, w, bin)
        # camera_depth = scio.loadmat(path + '/camera_depth.mat')['A']                    # (h, w)

        grasp_pts = np.where(grasp_point_map == 1.0)   # 标注的抓取点
        for i in range(grasp_pts[0].shape[0]):      # 遍历标注点
            row, col = grasp_pts[0][i], grasp_pts[1][i]
            angle_bins = np.where(grasp_angle_map[row, col] == 1.0)[0]
            for angle_bin in angle_bins:    # 0-17
                depth = grasp_depth_map[row, col, angle_bin]    # 相机离抓取点的深度
                self.grasp_labels.append([row, col, depth, angle_bin * 1.0 / 18 * math.pi, grasp_width_map[row, col, angle_bin]])
        print('loaded grasps: ', len(self.grasp_labels))

    def loadGraspLabelsFromTxt(self, path, grasp_start_id=0):
        """
        从txt文件加载抓取标签

        self.grasp_labels: [row, col, depth, angle, width]
        row, col为相机深度图像中的点坐标
        depth：抓取深度，相机到抓取点的深度
        angle：弧度
        width：单位米
        """
        print('loading grasp labels ...')
        self.grasp_labels = []
        self.grasp_id = grasp_start_id   # 记录遍历到的抓取索引

        self.angle_k = 18
        # 如果txt文件的最后一行没有字符，则f.readline()会返回None
        with open(path + '/graspLabel.txt') as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                strs = line.split(' ')
                row, col, angle_bin, width, depth = int(strs[0]), int(strs[1]), int(strs[2]), float(strs[3]), float(strs[4])
                self.grasp_labels.append([row, col, depth, angle_bin * 1.0 / 18 * math.pi, width])

        print('loaded grasps: ', len(self.grasp_labels))

    def getGrasp(self, thresh):
        """
        返回一个抓取配置 [x, y, z, angle, width]
        x y z width的单位是m, angle的单位是弧度
        其中x y z是机械手最末端中心的坐标
        """
        if self.grasp_id >= len(self.grasp_labels):
            return None
        ret = self.grasp_labels[self.grasp_id]
        print('\r grasp id:' + str(self.grasp_id), end="")
        self.grasp_id += 1
        return ret


    def evalGrasp(self, z_thresh, removeGrasped):
        """
        验证抓取是否成功
        如果某个物体的z坐标大于z_thresh，则认为抓取成功
        removeGrasped: 是否删除抓取成功的物体

        return: 
            True-抓取成功
            False-抓取失败
        """
        for idx in range(self.urdfs_obj_num()):
            idx += self.start_idx
            offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id_bullet[idx])
            if offset[2] >= z_thresh:
                if removeGrasped:
                    self.deleteURDF(idx)
                return True
        return False
    

    def evalGrasp1(self, thresh, removeGrasped):
        """
        验证抓取是否成功
        如果某个物体的xyz坐标与机械手手指的xyz坐标之差均小于thresh, 则认为抓取成功
        removeGrasped: 是否删除抓取成功的物体

        return: 
            True-抓取成功
            False-抓取失败
        """
        # 获取机械手手指坐标
        states = self.p.getLinkState(self.gripper.gripperId, 3)
        gripper_pos = states[0]
        
        for idx in range(self.urdfs_obj_num()):
            idx += self.start_idx
            obj_pos, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id_bullet[idx])
            if abs(gripper_pos[0] - obj_pos[0]) <= thresh and \
               abs(gripper_pos[1] - obj_pos[1]) <= thresh and \
               abs(gripper_pos[2] - obj_pos[2]) <= thresh:
                if removeGrasped:
                    self.deleteURDF(idx)
                return True
        return False

    def evalGrasp2(self):
        """
        验证抓取是否成功
        如果机械手两手指间距离大于0，认为抓取成功

        *** 无法删除已抓取的物体，需要删除已抓取的物体，请使用evalGrasp1 ***

        return: 
            True-抓取成功
            False-抓取失败
        """
        if self.gripper.gripperWidth > 0.001:
            return True
        else:
            return False


    def resetURDFsPose(self, file):
        """
        按照保存的位姿重置物体的位姿
        使用此函数时，环境中所有物体必须是由运行一次 loadURDFsWithPose 函数加载的
        ** 不包括地面和托盘 **

        file: 存放物体位姿的mat文件 / np.ndarray, 格式和mat文件一样
        """
        if isinstance(file, str):
            # 读取objsPose.mat文件
            if file.endswith('.mat'):
                path_ = file
            else:
                path_ = os.path.join(file, 'objsPose.mat')
            objsPose = scio.loadmat(path_)['A']
        elif isinstance(file, np.ndarray):
            objsPose = file

        for i in range(self.urdfs_num())[self.start_idx:]:
            j = i - self.start_idx
            basePosition = [objsPose[j, 1], objsPose[j, 2], objsPose[j, 3]]
            baseOrientation = [objsPose[j, 4], objsPose[j, 5], objsPose[j, 6], objsPose[j, 7]]
            self.p.resetBasePositionAndOrientation(self.urdfs_id_bullet[i], basePosition, baseOrientation)


    def resetURDFsPoseRandom(self):
        """
        随机重置物体的位姿
        ** 不包括地面和托盘 **
        """
        for i in range(self.urdfs_num())[self.start_idx:]:
            pos = 0.1
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), 0.3]
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            self.p.resetBasePositionAndOrientation(self.urdfs_id_bullet[i], basePosition, baseOrientation)
            self.sleep()


    def deleteAllURDFs(self):
        """
        移除所有objs
        ** 保留托盘和地面 **
        """
        for i in range(self.urdfs_obj_num()):
            self.p.removeBody(self.urdfs_id_bullet[self.start_idx])
            self.urdfs_id_bullet.pop(self.start_idx)
            self.urdfs_id_list.pop(self.start_idx)
            self.urdfs_scale.pop(self.start_idx)
            self.urdfs_vis_xyz.pop(self.start_idx)


    def deleteURDF(self, idx):
        """
        移除指定的obj
        idx: 模型在列表中的索引
        """
        self.p.removeBody(self.urdfs_id_bullet[idx])
        self.urdfs_id_bullet.pop(idx)
        self.urdfs_id_list.pop(idx)
        self.urdfs_scale.pop(idx)
        self.urdfs_vis_xyz.pop(idx)
    
    def save_cameraData(self, save_path):
        """
        保存相机位姿和内参至指定路径
        """
        if save_path.endswith('.mat'):
            save_path_ = save_path
        else:
            save_path_ = os.path.join(save_path, 'cameraPos.mat')
        scio.savemat(save_path_, {'in':self.camera.InMatrix,  'pos':self.camera.transMat})


    def renderGraspImages(self, save_path, parallel=True):
        """
        渲染计算抓取配置所需的图像
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # ======================== 渲染相机深度图 ========================
        print('>> 渲染相机深度图...')
        # 渲染图像
        img_camera = self.p.getCameraImage(WIDTH, HEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        rgba = img_camera[2]    # color data RGB
        dep = img_camera[3]    # depth data
        mask = img_camera[4]    # mask data

        # 获取彩色图像
        im_rgb = np.reshape(rgba, (h, w, 4))[:, :, [2, 1, 0]]
        im_rgb = im_rgb.astype(np.uint8)

        # 获取深度图像
        depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
        A = np.ones((HEIGHT, WIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((HEIGHT, WIDTH), dtype=np.float64) * farPlane
        C = np.ones((HEIGHT, WIDTH), dtype=np.float64) * (farPlane - nearPlane)
        # im_depthCamera = A / (B - C * depth)  # 单位 m
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m
        # im_depthCamera_rev = np.ones((HEIGHT, WIDTH), dtype=np.float64) * im_depthCamera.max() - im_depthCamera # 反转深度

        # 获取分割图像
        im_mask = np.reshape(mask, (h, w))
        # 保存图像
        # scio.savemat(save_path + '/camera_rgb.mat', {'A':im_rgb})
        scio.savemat(save_path + '/camera_depth.mat', {'A':im_depthCamera})
        # scio.savemat(save_path + '/camera_depth_rev.mat', {'A':im_depthCamera_rev})
        scio.savemat(save_path + '/camera_mask.mat', {'A':im_mask})

        cv2.imwrite(save_path + '/camera_rgb.png', im_rgb)
        cv2.imwrite(save_path + '/camera_depth.png', tool.depth2Gray(im_depthCamera))
        # cv2.imwrite(save_path + '/camera_depth_rev.png', tool.depth2Gray(im_depthCamera_rev))

        if not parallel:
            return im_rgb, im_depthCamera, im_mask, None, None

        # ======================== 渲染桌面深度图 ========================
        print('>> 渲染桌面深度图...')
        depth_map = np.ones((TABLE_IMG_HEIGHT, TABLE_IMG_WIDTH), dtype=np.float32) * 2
        mask_map = np.zeros((TABLE_IMG_HEIGHT, TABLE_IMG_WIDTH), dtype=np.uint8)

        for i in range(self.urdfs_num()):
            print('正在渲染... {}/{}: {}'.format(i+1, len(self.urdfs_id_bullet), self.urdfs_id_bullet[i]))

            # 获取obj当前位姿, world -> urdf
            offset, quaternion =  self.p.getBasePositionAndOrientation(self.urdfs_id_bullet[i])    # root_link坐标系相对于世界坐标系（urdf变换+base_link的惯性坐标）
            # 当物体在托盘外部，不进行渲染
            if abs(offset[0]) > 0.3 or abs(offset[1]) > 0.3:
                print('================== 超出范围 =================: ', self.urdfs_id_bullet[i])
                continue

            # 计算从 obj坐标系->URDF坐标系 的变换矩阵
            rot_obj_urdf = self.p.getMatrixFromQuaternion(self.p.getQuaternionFromEuler([0, 0, 0]))   # 欧拉角->旋转矩阵
            # mat_obj_urdf = tool.getTransfMat1(self.urdfs_vis_xyz[i], rot_obj_urdf) # 转换矩阵
            mat_obj_urdf = tool.getTransformMat1(self.urdfs_vis_xyz[i], rot_obj_urdf) # 转换矩阵
            # 计算从 urdf坐标系->世界坐标系 的变换矩阵
            rot_urdf_wld = self.p.getMatrixFromQuaternion(quaternion)  # 四元数转旋转矩阵
            # mat_urdf_wld = tool.getTransfMat1(offset, rot_urdf_wld)
            mat_urdf_wld = tool.getTransformMat1(offset, rot_urdf_wld)
            # 计算从 obj坐标系->世界坐标系 的变换矩阵
            # transMat = np.matmul(transMat, mat) # !!!! 注意乘的顺序

            # 获取obj文件列表
            objs_path = []
            infs = self.p.getVisualShapeData(self.urdfs_id_bullet[i])
            for inf in infs:
                objs_path.append(str(inf[4], 'utf-8'))

            for obj_path in objs_path:
                # 读取obj文件，并根据scale缩放
                mesh = Mesh(obj_path)
                mesh.setScale(self.urdfs_scale[i])
                # 根据旋转矩阵调整mesh顶点坐标
                # mesh.transform(transMat)
                mesh.transform(mat_obj_urdf)
                mesh.transform(mat_urdf_wld)    # 这里的变量命名与现在不一样，mat_urdf_wld 表示从world到urdf的转换矩阵
                t = time.time()
                # 渲染桌面深度图和mask
                depth_obj = mesh.renderTableImg1(self.camera)       # 主要耗时
                mask_map[depth_obj < depth_map] = self.urdfs_id_bullet[i]  # 更新mask
                depth_map = np.minimum(depth_map, depth_obj)        # 更新桌面深度图
                print('渲染耗时: {:.3f}s'.format(time.time()-t))
        
        print(depth_map.max(), depth_map.min())

        # 保存图像
        # scio.savemat(save_path + '/table_depth.mat', {'A':depth_map})
        # scio.savemat(save_path + '/table_mask.mat', {'A':mask_map})
        cv2.imwrite(save_path + '/table_depth.png', tool.depth2Gray(depth_map))

        print('>> 渲染结束')

        return im_rgb, im_depthCamera, im_mask, depth_map, mask_map


    def renderCameraImages(self):
        """
        渲染深度图像和RGB图像
        """
        # 渲染图像
        img_camera = self.p.getCameraImage(WIDTH, HEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # print('self.viewMatrix = ', self.viewMatrix)
        # print('self.projectionMatrix = ', self.projectionMatrix)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        rgba = img_camera[2]    # color data RGB
        dep = img_camera[3]    # depth data

        # 获取彩色图像
        im_rgb = np.reshape(rgba, (h, w, 4))[:, :, [2, 1, 0]]
        im_rgb = np.ascontiguousarray(im_rgb).astype(np.uint8)

        # 获取深度图像
        depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
        A = np.ones((HEIGHT, WIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((HEIGHT, WIDTH), dtype=np.float64) * farPlane
        C = np.ones((HEIGHT, WIDTH), dtype=np.float64) * (farPlane - nearPlane)
        # im_depthCamera = A / (B - C * depth)  # 单位 m
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m

        return im_rgb, im_depthCamera
    

    def renderParallelDepthImage(self, height=3000, width=3000, unit=0.0002):
        """
        通过激光碰撞检测的方式渲染平行深度图
        height: 平行深度图的高度
        width: 平行深度图的宽度
        unit: 每个像素表示的实际长度，单位m
        """
        # 获取平行深度图中所有像素点在相机坐标系中的坐标，以及沿相机z轴平移一定距离的点的坐标
        # 平行深度图上所有点在图像坐标系中的坐标
        rs, cs = np.mgrid[0:height, 0:width]
        locs_img = np.array(np.c_[cs.ravel()-(width-1)/2, rs.ravel()-(height-1)/2], dtype=np.float32)    # shape=(h*w, 2) x,y

        # 转换到相机坐标系下
        zs_from = np.zeros((locs_img.shape[0]))
        zs_to = np.ones((locs_img.shape[0]))
        locs_cam_from = np.insert(locs_img, 2, values=zs_from, axis=1) * unit
        locs_cam_to = np.insert(locs_img*unit, 2, values=zs_to, axis=1)

        # 转换到世界坐标系
        locs_wld_from = self.camera.camera2worlds(locs_cam_from)    # (h*w, 2)
        locs_wld_to = self.camera.camera2worlds(locs_cam_to)        # (h*w, 2)

        # 以locs_wld_from为起点，locs_wld_to为终点，进行raytest，获取深度
        parallelDepthImage = np.zeros((height, width), dtype=np.float32)
        r, c = 0, 0
        # 通过激光获得场景中点的位置
        for i in range(600):  # 3000/5
            if i % 100 == 0:
                print(i)
            rayFromPositions = locs_wld_from[i*15000:(i+1)*15000]
            rayToPositions = locs_wld_to[i*15000:(i+1)*15000]
            infs = p.rayTestBatch(rayFromPositions, rayToPositions, numThreads=4)
            for inf in infs:
                parallelDepthImage[r, c] = inf[2]
                c += 1
                if c == width:
                    c = 0
                    r += 1
        
        print('r = ', r, 'c = ', c)

        return parallelDepthImage
    

    def renderCameraMask(self):
        """
        渲染相机mask
        """
        # 渲染图像
        img_camera = self.p.getCameraImage(WIDTH, HEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        mask = img_camera[4]    # mask data

        # 获取分割图像
        im_mask = np.reshape(mask, (h, w))
        return im_mask


    def saveObjsPose(self, save_path):
        """
        保存物体的文件名和位姿至.mat文件，该.mat文件位于给定的文件夹内
        ** 不包括托盘和地面 **

        save_path: mat文件
        保存格式为[obj_id, pos, qua_xyzw]
        """
        print('>> 保存模型位姿...')
        poses = []
        for i in range(self.urdfs_num())[self.start_idx:]:
            offset, quaternion =  self.p.getBasePositionAndOrientation(self.urdfs_id_bullet[i])
            if abs(offset[0]) > 0.3 or abs(offset[1]) > 0.3:
                print('==== 超出范围 ==== :', self.urdfs_id_bullet[i])
                continue
            pose = np.array([[self.urdfs_id_list[i], offset[0], offset[1], offset[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]])
            poses.append(pose)

        poses = np.concatenate(tuple(poses))

        if save_path.endswith('.mat'):
            save_path_ = save_path
        else:
            save_path_ = os.path.join(save_path, 'objsPose.mat')
        scio.savemat(save_path_, {'A':poses})
        print('>> 保存结束')
    

    def objsPose(self):
        """
        返回物体位姿
        ** 不包括托盘和地面 **

        return:
            np.ndarray, shape=(n,8) 格式为[obj_id, pos, qua_xyzw]
        *** !! 注意初始化该类时的model_list参数，需与此函数的path保存的物体索引，保存一致，否则会加载错误的物体。 !! ***
        """
        poses = []
        for i in range(self.urdfs_num())[self.start_idx:]:
            offset, quaternion =  self.p.getBasePositionAndOrientation(self.urdfs_id_bullet[i])
            if abs(offset[0]) > 0.3 or abs(offset[1]) > 0.3:
                print('==== 超出范围 ==== :', self.urdfs_id_bullet[i])
                continue
            pose = np.array([[self.urdfs_id_list[i], offset[0], offset[1], offset[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]])
            poses.append(pose)

        poses = np.concatenate(tuple(poses))
        return poses

    def resetGripperPose(self, grasppose:GraspPose=None):
        """
        设置机械手位姿
        """
        if grasppose is None:
            pos = np.array([0, 0, 0.7])
            r = transformations.euler_matrix(0, np.pi, 0)[:3, :3]
            grasppose = GraspPose(pos, r)
        self.gripper.resetGripperPose(grasppose)


    # def isCover(self, contours, im_mask, im_depth, cover_thresh=5):
    #     """
    #     判断contours包围的物体是否被其他物体遮挡

    #     contours: array[[x y]] 物体的边缘点
    #     im_mask: 单通道uint8，每个点表示其所在物体的id
    #     im_depth: 单通道深度图 float64

    #     return: true表示被遮挡

    #     pt所在的物体被其他物体遮挡的条件：
    #     对于其8邻域内的其他点
    #     (1) mask_id 与 id 不同
    #     (2) 深度值更小
    #     (3) 遮挡点的个数超过阈值 5
    #     """
    #     offsets = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]  # dx dy
    #     H, W = im_mask.shape

    #     if len(contours.shape) == 1:
    #         return True

    #     cover_count = 0     # 记录遮挡点的个数
    #     for pt in contours:
    #         x, y = pt
    #         id = im_mask[y, x]
    #         dep = im_depth[y, x]

    #         for offset in offsets:
    #             _x, _y = x + offset[0], y + offset[1]
    #             # 判断是否越界
    #             if _x < 0 or _x >= W or _y < 0 or _y >= H:
    #                 continue
    #             # 判断mask_id
    #             if im_mask[_y, _x] == id:
    #                 continue
    #             # 判断深度
    #             if im_depth[_y, _x] < dep:
    #                 cover_count += 1
    #                 if cover_count >= cover_thresh:
    #                     return True
        
    #     return False


    # def getCoverMask(self, im_mask, im_depth):
    #     """
    #     根据im_mask和im_depthCamera得到遮挡标签
    #     没被遮挡的物体mask置1，其他全部为0
    #     """
    #     cover_mask = np.zeros_like(im_mask, dtype=np.uint8)

    #     # 遍历im_mask中的值
    #     ids_objs = list(np.unique(im_mask)[2:])     # 去掉地面和盒子
        
    #     for id_obj in ids_objs:
    #         # 获取二值图，对应物体的区域为255，其余为0
    #         rows, cols = np.where(im_mask == id_obj)
    #         im_mask_bin = np.zeros_like(im_mask)
    #         im_mask_bin[rows, cols] = 255

    #         contours, _ = cv2.findContours(im_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # contours: list[array]  array.shape = (n, 1, 2)
    #         contours = np.concatenate(tuple(contours), axis=0)  # (m, 1, 2)
    #         contours = np.squeeze(contours)

    #         # 判断是否被遮挡
    #         if not self.isCover(contours, im_mask, im_depth):
    #             cover_mask[rows, cols] = 1

    #     return cover_mask


    # def renderImageAndCover(self, save_path):
    #     """
    #     渲染相机深度图像，并计算遮挡检测的标签

    #     使用如下方式存储: img单位m
    #     cv2.imwrite('path.tiff', (img * 1000).astype(np.uint16))

    #     一张tiff图像的储存大小为30k
    #     生成100000张
    #     30k * 100000 = 3G = 
    #     """
    #     # 渲染图像
    #     img_camera = self.p.getCameraImage(WIDTH, HEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    #     w = img_camera[0]      # width of the image, in pixels
    #     h = img_camera[1]      # height of the image, in pixels
    #     rgba = img_camera[2]    # color data RGB
    #     dep = img_camera[3]    # depth data
    #     mask = img_camera[4]    # mask data

    #     # 获取彩色图像
    #     im_rgb = np.reshape(rgba, (h, w, 4))[:, :, [2, 1, 0]]
    #     im_rgb = im_rgb.astype(np.uint8)
    #     cv2.imwrite(save_path + 'camera_rgb.png', im_rgb)

    #     # 获取深度图像
    #     depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
    #     A = np.ones((HEIGHT, WIDTH), dtype=np.float64) * farPlane * nearPlane
    #     B = np.ones((HEIGHT, WIDTH), dtype=np.float64) * farPlane
    #     C = np.ones((HEIGHT, WIDTH), dtype=np.float64) * (farPlane - nearPlane)
    #     # im_depthCamera = A / (B - C * depth)  # 单位 m
    #     im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # 单位 m

    #     # 获取分割图像
    #     im_mask = np.reshape(mask, (h, w)).astype(np.uint8)
    #     # cv2.imshow('im_mask', im_mask*15)
    #     # cv2.waitKey()

    #     # 根据im_mask和im_depthCamera得到遮挡标签
    #     # 被遮挡的物体mask置1，其他全部为0
    #     cover_mask = self.getCoverMask(im_mask, im_depthCamera)

    #     # 保存图像
    #     # scio.savemat(save_path + '_mask.mat', {'A':cover_mask.astype(np.bool)})
    #     cv2.imwrite(save_path + '_mask.png', cover_mask)
    #     cv2.imwrite(save_path + '_depth.tiff', (im_depthCamera * 1000).astype(np.uint16))

        