import time
import numpy as np
import math
import pybullet_data
from grasp_methods.grasp import GraspPose
from utils.rigid_transformations import RigidTransform
from utils import transformations

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

jointPositions=(0.01999436579245478, 0.019977024051412193)
rp = jointPositions

class GripperSim(object):
    def __init__(self, bullet_client, position, friction):
        self.p = bullet_client
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        orn = self.p.getQuaternionFromEuler([math.pi, 0., math.pi/2])   # 机械手方向
        self.gripperId = self.p.loadURDF("models/franka_panda/panda_gripper.urdf", position, orn, useFixedBase=True, flags=flags)
        self.p.changeDynamics(self.gripperId, 2, lateralFriction=friction)
        self.p.changeDynamics(self.gripperId, 3, lateralFriction=friction)
        
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        #create a constraint to keep the fingers centered
        c = self.p.createConstraint(self.gripperId,
                          1,
                          self.gripperId,
                          2,
                          jointType=self.p.JOINT_GEAR,
                          jointAxis=[1, 0, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        # index = 0
        # for j in range(self.p.getNumJoints(self.gripperId))[1:]:
        #     self.p.changeDynamics(self.gripperId, j, linearDamping=0, angularDamping=0)
        #     info = self.p.getJointInfo(self.gripperId, j)
        #     jointName = info[1]
        #     jointType = info[2]
        #     if (jointType == self.p.JOINT_PRISMATIC):
        #         self.p.resetJointState(self.gripperId, j, jointPositions[index]) 
        #         index=index+1
        self.t = 0.
        self.setGripper(0.005)

        
        # self.t_world_base = [0, 0, -0.115]   # 世界坐标系到机械手base坐标系的平移
        # self.r_world_base = self.p.getQuaternionFromEuler([0, 0, math.pi/2]) # 世界坐标系到机械手base坐标系的四元数

        """
        加载机械手URDF时，机械手模型位于gripper坐标系原点，但不位于world坐标系原点
        gripper坐标系：机械手URDF坐标系
        grasp坐标系：原点为机械手的末端中心，y轴指向任一个夹爪，x轴垂直手掌，z轴与xy轴正交
        """
        t_grasp_gripper = [0, 0, -0.105]    # z= -0.115 时,机械手末端正好位于抓取位姿端点处，设置为-0.105使机械手超过抓取位姿端点，使抓取更稳。
        r_grasp_gripper = transformations.euler_matrix(0, 0, 0)[:3, :3]
        self.T_grasp_gripper = RigidTransform(rotation=r_grasp_gripper, translation=t_grasp_gripper, 
                                              from_frame='grasp', to_frame='gripper')

    @property
    def gripperWidth(self):
        """
        返回机械手的手指间距
        """
        finger1_pos = self.p.getJointState(self.gripperId, 1)[0]
        finger2_pos = self.p.getJointState(self.gripperId, 2)[0]
        return finger1_pos+finger2_pos


    def setGripper(self, finger_target):
        """
        设置机械手张开宽度

        传入的是每个手的闭合宽度，即抓取宽度的一半
        """
        self.p.setJointMotorControl2(self.gripperId, 1, self.p.POSITION_CONTROL, targetPosition=finger_target, force=30)
        self.p.setJointMotorControl2(self.gripperId, 2, self.p.POSITION_CONTROL, targetPosition=finger_target, force=30)


    def setArm(self, dist, maxVelocity=10):
        """设置机械手位置
        """
        self.p.setJointMotorControl2(self.gripperId, 0, self.p.POSITION_CONTROL, dist, force=50., maxVelocity=maxVelocity)
    
    
    def resetGripperPose(self, grasppose:GraspPose):
        """
        设置机械手姿态，包括位置和张开宽度

        grasppose: world坐标系下的抓取姿态
        T_world_gripper = T_world_grasp * T_grasp_gripper
        """
        assert grasppose.frame == 'world'
        T_world_gripper = grasppose.rigidTransform.dot(self.T_grasp_gripper)
        self.p.resetBasePositionAndOrientation(self.gripperId, T_world_gripper.translation, T_world_gripper.quaternion_xyzw)
        self.gripper_w = grasppose.width / 2
        self.setGripper(self.gripper_w)


    def step(self, dist):
        """
        l: float 机械手下降的深度
        gripper_w: float 抓取器张开宽度，单指侧闭合的距离，即原始的抓取宽度的一半
        """
        # 更新状态
        self.update_state()
        
        if self.state == -1:
            pass

        elif self.state == 0:
            # print('抓取位置')
            self.setArm(dist, maxVelocity=1)
            self.setGripper(self.gripper_w)
            return False

        elif self.state == 1:
            # print('闭合抓取器')
            self.setGripper(0)
            # self.setArm(dist, maxVelocity=1)
            return False
        
        elif self.state == 2:
            # print('物体上方')
            self.setGripper(0)
            self.setArm(0, maxVelocity=0.5)
            return False

        elif self.state == 103:
            self.reset()    # 重置状态
            return True

    def reset(self):
        """
        重置状态
        """
        self.state = 0
        self.state_t = 0
        self.cur_state = 0


class GripperSimAuto(GripperSim):
    def __init__(self, bullet_client, position, friction=1.0):
        """
        mode: 运行模式
            pick: 只抓取
            pick_place: 抓取物体后放置到另一个托盘里
        """
        GripperSim.__init__(self, bullet_client, position, friction)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 1, 2, 103]
        self.state_durations = [0.6, 0.4, 1, 0.1]
    
    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state = self.states[self.cur_state]
            # print("self.state =", self.state)
