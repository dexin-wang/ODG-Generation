'''
Description: 包含urdf类，object类和mtl类
Author: wangdx
Date: 2021-12-31 14:16:20
LastEditTime: 2022-01-14 21:35:58
'''
import os
import numpy as np


class Mtls:
    # 处理多个mtl
    def __init__(self) -> None:
        self.mtls = []
    
    def add_mtl(self, mtl):
        """
        mtl: Mtl类实例
        """
        self.mtls.append(mtl)

    def writeToFile(self, filepath):
        """
        filepath: txt路径
        """
        file = open(filepath, 'w')
        for mtl in self.mtls:
            mtl.writeToFile(file)
        file.close()
    
    def textureImgs(self):
        """
        获取所有mtl的贴图文件路径
        return: list
        """
        ret = []
        for mtl in self.mtls:
            if mtl.textureImg() is not None:
                ret.append(mtl.textureImg())
        return ret


class Mtl:
    # 处理单个mtl
    def __init__(self, name) -> None:
        self.name = name
        self.Ka = [0.0, 0.0, 0.0]
        self.Kd = [0.0, 0.0, 0.0]
        self.Ks = [0.0, 0.0, 0.0]
        self.Ns = 1.0
        self.d = 1.0
        self.map_Ka = None
        self.map_Kd = None

    def checkKd(self):
        if self.map_Ka is not None:
            self.Kd = [1.0, 1.0, 1.0]

    def writeToFile(self, file):
        """
        file: open的返回值
        """
        self.checkKd()
        file.write('newmtl ' + self.name + '\n')
        file.write('Ka ' + '{} {} {}'.format(self.Ka[0], self.Ka[1], self.Ka[2]) + '\n')
        file.write('Kd ' + '{} {} {}'.format(self.Kd[0], self.Kd[1], self.Kd[2]) + '\n')
        file.write('Ks ' + '{} {} {}'.format(self.Ks[0], self.Ks[1], self.Ks[2]) + '\n')
        file.write('Ns ' + str(self.Ns) + '\n')
        file.write('d ' + str(self.d) + '\n')
        if self.map_Ka is not None:
            file.write('map_Ka ' + self.map_Ka + '\n')
        if self.map_Kd is not None:
            file.write('map_Kd ' + self.map_Kd + '\n')
        file.write('\n')
    
    def textureImg(self):
        """
        获取贴图文件路径
        """
        return self.map_Kd





class Object:
    def __init__(self, path) -> None:
        self.path = path
        self._scale = 1
        self.inertial = [0, 0, 0]   # 重心xyz
        self.color = [0.6, 0.6, 0.6]   # 初始颜色
        self.read_obj()

    
    def read_obj(self):
        """
        读取obj文件中的点和面
        """
        with open(self.path) as file:
            self.points = []
            self.faces = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")  # f 1/1/1 5/2/1 7/3/1 3/4/1
                if strs[0] == "v":
                    self.points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "f":
                    start_id = 1
                    if strs[start_id] == '':
                        start_id = 2

                    if strs[start_id].count('//'):
                        idx1, idx2, idx3 = strs[start_id].index('//'), strs[start_id+1].index('//'), strs[start_id+2].index('//')
                        self.faces.append((int(strs[start_id][:idx1]), int(strs[start_id+1][:idx2]), int(strs[start_id+2][:idx3])))

                    elif strs[start_id].count('/'):
                        idx1, idx2, idx3 = strs[start_id].index('/'), strs[start_id+1].index('/'), strs[start_id+2].index('/')
                        self.faces.append((int(strs[start_id][:idx1]), int(strs[start_id+1][:idx2]), int(strs[start_id+2][:idx3])))

                    elif strs[start_id].count('/') == 0:
                        self.faces.append((int(strs[start_id]), int(strs[start_id+1]), int(strs[start_id+2])))
        
        self.points = np.array(self.points)     # (n, 3)
        self.faces = np.array(self.faces, dtype=np.int64)

    def setScale(self, scale):
        """
        scale: 物体缩放尺度
            float: 缩放scale倍
            -1 : 自动设置scale
        """
        assert scale == -1 or scale > 0

        if scale > 0:
            self._scale = scale
        else:
            self._scale = self.cal_scale()
        self.points = self.points * self._scale
    
    def getScale(self):
        """
        返回尺度
        """
        return self._scale

    def cal_scale(self, thresh=0.1):
        """
        计算使外接矩形框的最大边不超过thresh m 的尺度
        """
        d_x = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        d_y = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        d_z = np.max(self.points[:, 2]) - np.min(self.points[:, 2])
        scale = thresh / max(max(d_x, d_y), d_z)
        return scale
    
    def calcCenterPt(self):
        """
        计算mesh的中心点坐标 
        return: [x, y, z]
        """
        return np.mean(self.points, axis=0)
        
    def move(self, pos):
        """
        将obj移动pos
        pos: [x, y, z]
        """
        self.points[:, 0] = self.points[:, 0] + pos[0]
        self.points[:, 1] = self.points[:, 1] + pos[1]
        self.points[:, 2] = self.points[:, 2] + pos[2]

    def setInertial(self):
        """
        设置重心
        """
        self.inertial = self.calcCenterPt()
        
    def setColor(self, color):
        """
        设置颜色，
        color: [r g b] 均为0-1的浮点数
        """
        self.color[0] = format(color[0], '.2f') # 保留两位小数
        self.color[1] = format(color[1], '.2f')
        self.color[2] = format(color[2], '.2f')



class URDF:
    def __init__(self) -> None:
        self.str_head = '<?xml version="0.0" ?> \n' \
            '<robot name="name.urdf"> \n'
        self.str_end = '</robot>'
        self.base_link = '<link name="base_link"> \n' \
            '  <inertial> \n' \
            '    <origin rpy="0 0 0" xyz="0 0 0"/> \n' \
            '    <mass value="0.1"/> \n' \
            '    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/> \n' \
            '  </inertial> \n' \
            '</link> \n'
        # str_link 需要替换 linkname, objname, objcolname, _iner_xyz, _vis_xyz, _scale, _color
        self.str_link = '<link name="linkname"> \n' \
            '  <contact> \n' \
            '    <lateral_friction value="1.0"/>\n' \
            '    <rolling_friction value="0.0"/>\n' \
            '    <contact_cfm value="0.0"/>\n' \
            '    <contact_erp value="1.0"/>\n' \
            '  </contact> \n' \
            '  <inertial> \n' \
            '    <origin rpy="0 0 0" xyz="_iner_xyz"/> \n' \
            '    <mass value="0.1"/> \n' \
            '    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/> \n' \
            '  </inertial> \n' \
            '  <visual> \n' \
            '    <origin rpy="0 0 0" xyz="_vis_xyz"/> \n' \
            '    <geometry> \n' \
            '      <mesh filename="objname" scale="_scale"/> \n' \
            '    </geometry> \n' \
            '    <material name="yellow"> \n' \
            '      <color rgba="_color 1"/> \n' \
            '    </material> \n' \
            '  </visual> \n' \
            '  <collision> \n' \
            '    <origin rpy="0 0 0" xyz="_vis_xyz"/> \n' \
            '    <geometry> \n' \
            '      <mesh filename="objcolname" scale="_scale"/> \n' \
            '    </geometry> \n' \
            '  </collision> \n' \
            '</link> \n'
        # str_joint 需要替换 jointname, linkname1, linkname2
        self.str_joint = '<joint name="jointname" type="fixed"> \n' \
                '  <parent link="linkname1"/> \n' \
                '  <child link="linkname2"/> \n' \
                '  <origin rpy="0 0 0" xyz="0 0 0"/> \n' \
                '</joint> \n'

        self.objects = []
        self.links = []
        self.joints = []
        self.scale = 1
        self.xyz = [0, 0, 0]

    def add_object(self, object):
        """
        添加obj
        object: Object实例
        """
        self.objects.append(object)
    
    def cal_scale(self, thresh=0.1):
        """
        计算使所有obj的外接矩形框的最大边不超过thresh m 的尺度
        """
        for obj in self.objects:
            self.scale = min(self.scale, obj.cal_scale(thresh))
        return self.scale
    
    def setScale(self, scale, thresh=0.1):
        """
        调整所有obj的尺度，都设置为scale
        scale:
            float: 缩放scale倍
            -1 : 自动设置scale
        """
        assert scale == -1 or scale > 0

        if scale > 0:
            self.scale = scale
        else:
            self.scale = self.cal_scale(thresh)

        for obj in self.objects:
            obj.setScale(self.scale)
    
    def calCenterPt(self):
        """
        计算所有obj的中心点坐标 
        return: [x, y, z]
        """
        # 先合并所有obj的点
        pts = []
        for obj in self.objects:
            pts.append(obj.points)
        
        pts = np.concatenate(tuple(pts), axis=0)  # (N, 3)

        return np.mean(pts, axis=0)
    
    def move(self, pos):
        """
        将所有obj移动pos
        pos: [x, y, z]
        """
        self.xyz = pos
        for obj in self.objects:
            obj.move(self.xyz)
    
    def setObjInertial(self):
        """
        设置每个obj的重心
        """
        for obj in self.objects:
            obj.setInertial()

    def setColor(self, color):
        """
        设置颜色
        color: [r g b] 均为0-1的浮点数
        """
        for obj in self.objects:
            obj.setColor(color)
        

    def write(self, filepath):
        """
        写入urdf文件
        filepath: urdf文件路径

        str_link 需要替换 linkname, objname, objcolname, _iner_xyz, _vis_xyz, _scale, _color
        str_joint 需要替换 jointname, linkname1, linkname2
        """
        urdf_data = self.str_head + self.base_link
        for obj_idx in range(len(self.objects)):
            obj = self.objects[obj_idx]
            linkname = 'link_{:04d}'.format(obj_idx+1)
            objname = os.path.basename(obj.path)

            if os.path.exists(os.path.join(os.path.dirname(filepath), objname.replace('.obj', '_col.obj'))):
                objcolname = objname.replace('.obj', '_col.obj')
            else:
                objcolname = objname
            _iner_xyz = str(obj.inertial[0]) + ' ' + str(obj.inertial[1]) + ' ' + str(obj.inertial[2])
            _vis_xyz = str(self.xyz[0]) + ' ' + str(self.xyz[1]) + ' ' + str(self.xyz[2])
            _scale = str(self.scale) + ' ' + str(self.scale) + ' ' + str(self.scale)
            _color = str(obj.color[0]) + ' ' + str(obj.color[1]) + ' ' + str(obj.color[2])
            
            link_data = self.str_link.replace('linkname', linkname).replace('objname', objname).replace('objcolname', objcolname)
            link_data = link_data.replace('_iner_xyz', _iner_xyz).replace('_vis_xyz', _vis_xyz).replace('_scale', _scale).replace('_color', _color)
            urdf_data += link_data

            if obj_idx >= 1:
                linkname1 = 'link_{:04d}'.format(obj_idx)
            else:
                linkname1 = 'base_link'
            linkname2 = 'link_{:04d}'.format(obj_idx+1)
            jointname = 'joint_{:04d}'.format(obj_idx)
            joint_data = self.str_joint.replace('jointname', jointname).replace('linkname1', linkname1).replace('linkname2', linkname2)
            urdf_data += joint_data
                
        urdf_data += self.str_end
        f = open(filepath, 'w')
        f.write(urdf_data)
        f.close()

        

if __name__ == '__main__':
    """
    为path路径下的模型生成urdf

    path： obj所在的目录
    """
    
    obj_path = 'E:/research/dataset/grasp/sim_grasp/mesh_database/compare/2.obj'
    
    obj = Object(obj_path)
    urdf = URDF()
    urdf.add_object(obj)

    # 设置尺度
    urdf.setScale(1)
    # 设置重心
    urdf.setObjInertial()
    # 设置颜色
    urdf.setColor([0.5, 0.5, 0.5])

    # 写入urdf
    urdf_path = obj_path.replace('.obj', '.urdf')
    urdf.write(urdf_path)

    print('save', urdf_path)