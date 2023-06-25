import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
# restposes for null space
jointPositions=(0.8045609285966308, 0.525471701354679, -0.02519566900946519, -1.3925086098003587, 0.013443782914225877, 1.9178323512245277, -0.007207024243406651, 0.01999436579245478, 0.019977024051412193)
            # [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.p = bullet_client
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
        
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orn=[0, 0, 0, 1]
        self.panda = self.p.loadURDF("models/franka_panda/panda_1.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        #create a constraint to keep the fingers centered
        c = self.p.createConstraint(self.panda,
                          9,
                          self.panda,
                          10,
                          jointType=self.p.JOINT_GEAR,
                          jointAxis=[1, 0, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
        for j in range(self.p.getNumJoints(self.panda)):
            self.p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.p.getJointInfo(self.panda, j)
            #print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.p.JOINT_PRISMATIC):
                self.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1

            if (jointType == self.p.JOINT_REVOLUTE):
                self.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1
        self.t = 0.

    def calcJointLocation(self, pos, orn):
        """
        根据 pos 和 orn 计算机械臂的关节位置 
        """
        jointPoses = self.p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll, ul, jr, rp, maxNumIterations=20)
        return jointPoses


    def setArm(self, jointPoses, maxVelocity=10):
        """
        设置机械臂位置
        """
        for i in range(pandaNumDofs):   # 7
            self.p.setJointMotorControl2(self.panda, i, self.p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxVelocity)
    
    def setGripper(self, finger_target):
        """
        设置机械手位置
        """
        for i in [9,10]:
            self.p.setJointMotorControl2(self.panda, i, self.p.POSITION_CONTROL, finger_target, force=20)


    def step(self, pos, angle, gripper_w):
        """
        pos: [x, y, z] 机械手末端的位置
            x: 世界坐标系x轴，沿机械臂右侧方向
            y: 世界坐标系y轴，沿机械臂前方方向
            y: 世界坐标系z轴，沿机械臂上方方向
            单位米，原点位于托盘中心
        angle: 弧度
        gripper_w: 抓取器张开宽度
        """
        
        # 测试用
        # pos = [0.5, 0, 0.3] # 机械手位置
        # orn = self.p.getQuaternionFromEuler([math.pi, 0., math.pi / 2])   # 机械手方向
        # jointPoses = self.calcJointLocation(pos, orn)
        # print('jointPoses = ', jointPoses)
        # self.setArm(jointPoses)
        # return False

        # 更新状态
        self.update_state()
        
        pos[2] += 0.048
        if self.state == -1:
            pass

        elif self.state == 0:
            # print('恢复初始状态')
            # pos = [0, 0, 0.3] # 机械手位置
            # pos = [0.5, 0, 0.3] # 机械手位置
            pos[2] = 0.2
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            self.setGripper(gripper_w)
            return False

        # elif self.state == 101:
        #     # print('物体上方，旋转抓取器')
        #     pos = [0, 0, 0.3] # 机械手位置
        #     orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
        #     jointPoses = self.calcJointLocation(pos, orn)
        #     self.setArm(jointPoses)
        #     self.setGripper(gripper_w)
        #     return False

        # elif self.state == 102:
        #     # print('张开抓取器')
        #     self.setGripper(gripper_w)
        #     return False

        elif self.state == 1:
            # print('物体上方')
            pos[2] += 0.05
            orn = self.p.getQuaternionFromEuler([math.pi, 0., angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False

        elif self.state == 2:
            # print('抓取位置')
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses, maxVelocity=3)
            return False

        elif self.state == 3:
            # print('闭合抓取器')
            self.setGripper(0)
            return False
        
        elif self.state == 4:
            # print('物体上方(预抓取位置)')
            pos[2] += 0.05
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses, maxVelocity=0.5)
            # self.setGripper(0)
            return False
        
        elif self.state == 5:
            # print('物体上方')
            pos[2] = 0.3
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses, maxVelocity=0.5)
            # self.setGripper(0)
            return False

        # ========================= 晃动抓取器，测试抓取稳定性 =========================
        # elif self.state == 6:
        #     # print('x正方向晃动')
        #     pos[2] = 0.4
        #     pos[0] -= 0.05
        #     orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
        #     jointPoses = self.calcJointLocation(pos, orn)
        #     self.setArm(jointPoses)
        #     return False
        # elif self.state == 7:
        #     # print('x负方向晃动')
        #     pos[2] = 0.4
        #     pos[0] += 0.05
        #     orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   # 机械手方向
        #     jointPoses = self.calcJointLocation(pos, orn)
        #     self.setArm(jointPoses)
        #     return False
        # ==================================================

        # elif self.state == 8:
        #     # print('放置物体')
        #     pos = [0.5, 0, 0.3] # 机械手位置
        #     orn = self.p.getQuaternionFromEuler([math.pi,0.,math.pi / 2])   # 机械手方向
        #     jointPoses = self.calcJointLocation(pos, orn)
        #     self.setArm(jointPoses)
        #     return False

        # elif self.state == 9:
        #     # print('盒子上方')
        #     pos = [0, 0, 0.4] # 机械手位置
        #     orn = self.p.getQuaternionFromEuler([math.pi,0.,math.pi / 2])   # 机械手方向
        #     jointPoses = self.calcJointLocation(pos, orn)
        #     self.setArm(jointPoses)
        #     return False
        
        # elif self.state == 10:
        #     # print('张开抓取器')
        #     self.setGripper(0.04)
        #     return False
        
        # elif self.state == 11:
        #     # print('恢复初始状态')
        #     pos = [0.5, 0, 0.3] # 机械手位置
        #     orn = self.p.getQuaternionFromEuler([math.pi,0., math.pi / 2])   # 机械手方向
        #     jointPoses = self.calcJointLocation(pos, orn)
        #     self.setArm(jointPoses)
        #     self.reset()    # 重置状态
        #     return True

        elif self.state == 12:
            self.reset()    # 重置状态
            return True


    def reset(self):
        """
        重置状态
        """
        self.state = 0
        self.state_t = 0
        self.cur_state = 0


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        """
        0: 重置状态
        1: 物体上方(预抓取位置)，张开抓取器
        2: 抓取位置
        3: 闭合抓取器
        4: 物体上方(预抓取位置)
        5: 物体上方
        6: x正方向晃动
        7: x负方向晃动
        8: 放置物体
        9: 盒子上方
        10: 张开抓取器
        """
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        # self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.states = [0, 1, 2, 3, 4, 5, 12]
        self.state_durations = [1.0, 0.5, 2.0, 0.5, 1.0, 1.0, 0.5]
    
    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state = self.states[self.cur_state]
            # print("self.state =", self.state)
