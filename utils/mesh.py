import os
from cv2 import exp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path
from skimage.draw import polygon, line
from shapely.geometry import Polygon
from utils.camera import Camera
from scripts.dataset.generate_graspFig import GRASP_MAX_W


def clacPlane(pt1, pt2, pt3):
    """
    根据三点计算平面方程 ax+by+cz+d=0
    pts: [[x, y, z], [x, y, z], [x, y, z]]
    return: A B C   z=Ax+By+C
    """
    a = (pt2[1]-pt1[1])*(pt3[2]-pt1[2]) - (pt2[2]-pt1[2])*(pt3[1]-pt1[1])
    b = (pt2[2]-pt1[2])*(pt3[0]-pt1[0]) - (pt2[0]-pt1[0])*(pt3[2]-pt1[2])
    c = (pt2[0]-pt1[0])*(pt3[1]-pt1[1]) - (pt2[1]-pt1[1])*(pt3[0]-pt1[0])
    d = 0 - (a * pt1[0] + b * pt1[1] + c * pt1[2])

    return a, b, c, d

    # if a == 0 or b == 0 or c == 0 or d == 0:
    #     print('a = ', a)
    #     print('b = ', b)
    #     print('c = ', c)
    #     print('d = ', d)
    # return np.array([-1*a/c, -1*b/c, -1*d/c])


def polygonsOverlap(convex1, convex2):
    """
    获取两个多边形的相交区域内的点
    convex1: np.ndarray shape=(n,2) r,c 多边形的外包络点
    """
    # 获取两个多边形的相交区域的多边形的外包络点
    polygon1 = Polygon(tuple([tuple(e) for e in convex1]))
    polygon2 = Polygon(tuple([tuple(e) for e in convex2]))
    try:
        convex_rr, convex_cc = polygon1.intersection(polygon2).convex_hull.exterior.coords.xy
        assert len(convex_rr) == len(convex_cc) and len(convex_cc) >= 4

        rs, cs = [], []
        for i in range(len(convex_rr)):
            rs.append(convex_rr[i])
            cs.append(convex_cc[i])

        rr, cc = polygon(rs, cs)
        # rr, cc = polygon([convex_rr[0], convex_rr[1], convex_rr[2]], [convex_cc[0], convex_cc[1], convex_cc[2]])
        return np.column_stack((rr, cc))    # (n, 2)
    except:
        return np.column_stack((np.array([]), np.array([])))


def ptsInTriangle1(pt1, pt2, pt3):
    """
    获取pt1 pt2 pt3 组成的三角形内的坐标点
    pt1: float [r, c, dep] 本算法只使用r和c    
    
    return: np.ndarray (n, 2) r,c
    """
    rr, cc = polygon([pt1[0], pt2[0], pt3[0]], [pt1[1], pt2[1], pt3[1]])    # rr, cc: (n,)
    return np.column_stack((rr, cc))    # (n, 2)



class Mesh:
    """
    mesh 类，读取obj文件，坐标转换，生成空间深度图
    """
    def __init__(self, filename):
        """
        读取obj文件，获取v 和 f
        只用于读取EGAD数据集的obj文件

        filename: obj文件名
        """
        
        with open(filename) as file:
            self.points = []
            self.faces = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.replace('  ', ' ').replace('\n', '').split(" ")  # f 1/1/1 5/2/1 7/3/1 3/4/1
                if strs[0] == "v":
                    # print('strs = ', strs)
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
        
        self.points = np.array(self.points)
        self.faces = np.array(self.faces, dtype=np.int64)

    def write(self, filename):
        """
        写入obj文件
        只写入 point和face
        """
        f = open(filename, 'w')

        for pt in self.points:
            f.write('v %f %f %f\n' %(pt[0], pt[1], pt[2]))

        for face in self.faces:
            f.write('f %d %d %d\n' %(face[0], face[1], face[2]))

        f.close()

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

    def setOffset(self, offset:np.ndarray):
        """
        平移mesh
        offset: np.array() shape=(3,)
        """
        self.points = self.points + offset

    def min_z(self):
        """
        返回最小的z坐标
        """
        return np.min(self.points[:, 2])


    def cal_scale(self):
        """
        自适应设置scale
        使外接矩形框的中间边不超过抓取器宽度(0.07)的80% scale最大为0.001
        """
        d_x = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        d_y = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        d_z = np.max(self.points[:, 2]) - np.min(self.points[:, 2])
        ds = [d_x, d_y, d_z]
        ds.sort()
        scale = (GRASP_MAX_W - 0.01) * 0.8 / ds[1]
        if scale > 0.001:
            scale = 0.001
        
        return scale
    
    
    def cal_scale1(self):
        """
        自适应设置scale
        使外接矩形框的最大边不超过0.1m
        """
        d_x = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        d_y = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        d_z = np.max(self.points[:, 2]) - np.min(self.points[:, 2])
        scale = 0.05 / max(max(d_x, d_y), d_z)
        return scale
    

    def scale(self):
        return self._scale
    
    def calcCenterPt(self):
        """
        计算mesh的中心点坐标 
        return: [x, y, z]
        """
        return np.mean(self.points, axis=0)

    
    def transform(self, mat):
        """
        根据旋转矩阵调整顶点坐标
        """
        points = self.points.T  # 转置
        ones = np.ones((1, points.shape[1]))
        points = np.vstack((points, ones))
        # 转换
        new_points = np.matmul(mat, points)[:-1, :]
        self.points = new_points.T  # 转置  (n, 3)

    
    # TODO 优化加速
    def renderTableImg1(self, camera:Camera, size=(3000, 3000), unit=0.0002):
        """
        渲染相对于水平面的深度图和 obj mask，每个点之间的间隔为0.5mm
        size: 图像尺寸 (h, w)
        unit: 每个像素表示的实际长度，单位m
        """
        # 将全部顶点转换到相机坐标系下
        self.transform(np.linalg.inv(camera.transMat))

        # for 每一个网格
        depth_map = np.ones(size, dtype=np.float32) * 2
        for face in self.faces:
            # 将三个顶点转换到像素坐标系，以及深度值
            pt1 = self.points[face[0] - 1]  # xyz m 相机坐标系下
            pt2 = self.points[face[1] - 1]
            pt3 = self.points[face[2] - 1]

            pt1_pix = [pt1[1] / unit + (size[0]-1)/2, pt1[0] / unit + (size[1]-1)/2, pt1[2]]  # (r,c,dep) 
            pt2_pix = [pt2[1] / unit + (size[0]-1)/2, pt2[0] / unit + (size[1]-1)/2, pt2[2]]
            pt3_pix = [pt3[1] / unit + (size[0]-1)/2, pt3[0] / unit + (size[1]-1)/2, pt3[2]]

            # 获取三角形内的像素坐标 (x,y)
            # 实现方法：计算网格和图像的相交区域内的点
            convex_img = np.array([[0, 0], [0, size[1]-1], [size[0]-1, size[1]-1], [size[0]-1, 0]])
            convex_face = np.array([[pt1_pix[0], pt1_pix[1]], [pt2_pix[0], pt2_pix[1]], [pt3_pix[0], pt3_pix[1]]])
            pts_pix = polygonsOverlap(convex_img, convex_face)   # (n, 2) r,c
            if pts_pix.shape[0] == 0:
                continue

            # 计算三个顶点组成的平面方程
            plane_a, plane_b, plane_c, plane_d = clacPlane(pt1_pix, pt2_pix, pt3_pix)    # ABC  Ax+By+C=z
            plane = np.array([-1*plane_a/plane_c, -1*plane_b/plane_c, -1*plane_d/plane_c])
            # 在平面方程中计算深度值
            ones = np.ones((pts_pix.shape[0], 1))   # (n, 1)
            pts_pix_ = np.hstack((pts_pix, ones))
            depth = np.matmul(pts_pix_, plane.reshape((3,1))).reshape((-1,)) # (n,)

            # 更新深度值
            depth_map[pts_pix[:, 0], pts_pix[:, 1]] = np.minimum(depth_map[pts_pix[:, 0], pts_pix[:, 1]], depth)

        return depth_map



if __name__ == "__main__":
    p = path.Path([(0, 0), (0.001, 0), (0, 0.001)])
    ret = p.contains_points([(0.00001, 0.00001)])[0]
    print(ret)



