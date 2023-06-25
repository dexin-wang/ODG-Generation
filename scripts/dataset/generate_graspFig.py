"""
生成抓取配置
"""
import cv2
import math
import time
import copy
import random
import numpy as np
import os
import scipy.io as scio
from skimage.draw import polygon, line
import sys
sys.path.append(os.curdir)
import utils.tool as tool
from utils.camera import Camera


# 机械手参数(根据抓取器实际参数设置)
FINGER_L1 = 0.017   # 17mm
FINGER_L1_1 = FINGER_L1 + 0.005

FINGER_L2 = 0.005   # 5mm
FINGER_L2_1 = FINGER_L2 + 0.005   #  [0.002]

FINGER_L3 = 0.05

GRASP_MAX_W = 0.1     # 最大张开宽度
# GRASP_MIN_W = 0.005    # 最小张开宽度

# 抓取下降沿搜索 参数
GRASP_STRIDE = 0.0008  # 抓取下降沿的搜索步长 m 每个像素约长0.0016
GRASP_ANGLE_BINS = 18    # 抓取角分类数 共180° [12, 18]
DEATH_STRIDE = 0.005    # 深度下降步长     [0.0025, 0.005]
GRASP_DOWN_ANGLE = 90.-np.rad2deg(np.arctan(0.2))   # 摩擦系数为0.2时的下降沿角
DEATH_DOWN_THRESH = GRASP_STRIDE * math.tan(math.radians(GRASP_DOWN_ANGLE))

GRASP_GAP = 0.005   # [0.005]   抓取器与物体之间的间隔
GRASP_DEPTH = 0.005 # 5mm   抓取深度    [0.01, 0.005]
GRASP_ANGLE_RANGE = 10    # 抓取角的范围 [-10, 10]

EDGE_VAR_THRESH = 150   # 边缘方差阈值

# 桌面深度图参数
tableImgSize=(0.6, 0.6)
unit=0.0002
TABLE_IMG_HEIGHT = 3000  #int(tableImgSize[0] / unit)
TABLE_IMG_WIDTH = 3000  # int(tableImgSize[1] / unit)

HEIGHT = 480
WIDTH = 640

CAMERA_HEIGHT = 0.6 # 仿真场景相机高度


def radiansSymmetry(radians):
    """
    获取与radians对称的另一个角
    radians: 弧度
    return:  180 <= radians2 < 360
    """
    return radians + math.pi - int((radians + math.pi) // (2 * math.pi)) * 2 * math.pi

def ptsAlongAngle(pt, angle, stride, mask, obj_id, length):
    """
    获取angle方向上的点
    停止条件: 
        (1) 达到最大抓取宽度
        (2) 倒数第二个点在物体mask内 (最后一个点的angle方向上的点都不在物体mask上, 最后一个点可以不在物体mask上)
    pt: [row, col]
    angle: 弧度, [0, pi) 或 [pi, 2*pi)
    stride: 步长
    mask: 物体mask
    id: 物体id
    length: m

    return: list([x, y])
    """
    pt_x, pt_y = pt

    # 计算[-length, length]内的所有点
    length = length / unit  # 实际长度转为像素长度
    stride = stride / unit  # 实际长度转为像素长度
    dy = int(length * math.sin(angle) * -1)
    dx = int(length * math.cos(angle))
    # pt_end = []
    if abs(dy) >= abs(dx):
        # 计算直线方程
        k = dx / dy
        symbol = int(round(dy / abs(dy)))
        pts = [[pt_x, pt_y]]
        for i in range(0, dy, symbol):
            x = pt_x + int(round(k * i))
            y = pt_y + i
            if x < 0 or x >= TABLE_IMG_WIDTH or y < 0 or y >= TABLE_IMG_HEIGHT:
                continue
            if tool.distancePt(pts[-1], [x, y]) >= stride:
                pts.append([x, y])
        # return pts
    else:
        k = dy / dx
        symbol = int(round(dx / abs(dx)))
        pts = [[pt_x, pt_y]]
        for i in range(0, dx, symbol):
            x = pt_x + i
            y = pt_y + int(round(k * i))
            if x < 0 or x >= TABLE_IMG_WIDTH or y < 0 or y >= TABLE_IMG_HEIGHT:
                continue
            if tool.distancePt(pts[-1], [x, y]) >= stride:
                pts.append([x, y])
        # return pts
    
    # 筛选满足条件的pts
    if mask[pts[-1][1], pts[-1][0]] == obj_id:
        return pts

    if mask[pts[-2][1], pts[-2][0]] == obj_id and mask[pts[-1][1], pts[-1][0]] != obj_id:
        return pts

    for i in range(len(pts))[1:][::-1]:
        p = pts[i]
        p_near = pts[i-1]
        if mask[p_near[1], p_near[0]] == obj_id and mask[p[1], p[0]] != obj_id:
            return pts[:i+1]
    
    print('mask[p_near[1], p_near[0]] = ', mask[p_near[1], p_near[0]])
    print('mask[p[1], p[0]] = ', mask[p[1], p[0]])
    return []   # ??

def calcDownAngle(pt1, pt2):
    """
    计算两点高度方向的夹角
    pt1: 离中心点较近的点 [x, y, z]
    pt2: 离中心点较远的点 [x, y, z]
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dz = pt2[2] - pt1[2]
    # print()
    return math.atan(dz / ((dx ** 2 + dy ** 2) ** 0.5))

def ptsOnRect(pts):
    """
    获取矩形框上五条线上的点
    五条线分别是：四条边缘线，1条对角线
    pts: np.array, shape=(4, 2) (row, col)
    """
    rows1, cols1 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[1, 0]), int(pts[1, 1]))
    rows2, cols2 = line(int(pts[1, 0]), int(pts[1, 1]), int(pts[2, 0]), int(pts[2, 1]))
    rows3, cols3 = line(int(pts[2, 0]), int(pts[2, 1]), int(pts[3, 0]), int(pts[3, 1]))
    rows4, cols4 = line(int(pts[3, 0]), int(pts[3, 1]), int(pts[0, 0]), int(pts[0, 1]))
    rows5, cols5 = line(int(pts[0, 0]), int(pts[0, 1]), int(pts[2, 0]), int(pts[2, 1]))

    rows = np.concatenate((rows1, rows2, rows3, rows4, rows5), axis=0)
    cols = np.concatenate((cols1, cols2, cols3, cols4, cols5), axis=0)
    return rows, cols

def ptsOnRotateRect(pt1, pt2, w):
    """
    绘制矩形
    已知图像中的两个点（x1, y1）和（x2, y2），以这两个点为端点画线段，线段的宽是w。这样就在图像中画了一个矩形。
    pt1: [row, col] 
    w: 单位像素
    img: 绘制矩形的图像, 单通道
    """
    y1, x1 = pt1
    y2, x2 = pt2

    if x2 == x1:
        if y1 > y2:
            angle = math.pi / 2
        else:
            angle = 3 * math.pi / 2
    else:
        tan = (y1 - y2) / (x2 - x1)
        angle = np.arctan(tan)

    points = []
    points.append([y1 - w / 2 * np.cos(angle), x1 - w / 2 * np.sin(angle)])
    points.append([y2 - w / 2 * np.cos(angle), x2 - w / 2 * np.sin(angle)])
    points.append([y2 + w / 2 * np.cos(angle), x2 + w / 2 * np.sin(angle)])
    points.append([y1 + w / 2 * np.cos(angle), x1 + w / 2 * np.sin(angle)])
    points = np.array(points)

    # 方案1，比较精确，但耗时
    # rows, cols = polygon(points[:, 0], points[:, 1], (10000, 10000))	# 得到矩形中所有点的行和列

    # 方案2，速度快
    return ptsOnRect(points)	# 得到矩形中所有点的行和列

def drawRotateRectangle(img, pt1, pt2, w, color):
    """
    绘制矩形
    已知图像中的两个点（x1, y1）和（x2, y2），以这两个点为端点画线段，线段的宽是w。这样就在图像中画了一个矩形。
    pt1: [row, col] 
    w: 单位像素
    img: 绘制矩形的图像, 单通道
    """
    y1, x1 = pt1
    y2, x2 = pt2

    if x2 == x1:
        if y1 > y2:
            angle = math.pi / 2
        else:
            angle = 3 * math.pi / 2
    else:
        tan = (y1 - y2) / (x2 - x1)
        angle = np.arctan(tan)

    points = []
    points.append([y1 - w / 2 * np.cos(angle), x1 - w / 2 * np.sin(angle)])
    points.append([y2 - w / 2 * np.cos(angle), x2 - w / 2 * np.sin(angle)])
    points.append([y2 + w / 2 * np.cos(angle), x2 + w / 2 * np.sin(angle)])
    points.append([y1 + w / 2 * np.cos(angle), x1 + w / 2 * np.sin(angle)])
    points = np.array(points)

    # 方案1，比较精确，但耗时
    # rows, cols = polygon(points[:, 0], points[:, 1], (10000, 10000))	# 得到矩形中所有点的行和列

    # 方案2，速度快
    # rows, cols = ptsOnRect(points)	# 得到矩形中所有点的行和列
    
    # img_1 = img.copy()
    # img_1[rows, cols] = color
    # return img_1

def linear_regression(x, y): 
    """
    拟合直线
    x, y: list()
    return:  b, k  y = a0 + k*x
    """
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
 
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
 
    return np.linalg.solve(A, b)

def angle_regression(x, y):
    """
    拟合角.  
    x-y计算一次, y-x计算一次, 求哪个误差小,如果x-y误差小,直接返回angle, 如果y-x误差小, 计算沿45°对称的角
    return: 弧度
    """
    # 垂直的情况
    if x.min() == x.max():
        return math.pi / 2.
    # 水平的情况
    if y.min() == y.max():
        return 0
    # x-y
    b1, k1 = linear_regression(x, y)
    # y-x
    b2, k2 = linear_regression(y, x)
    # 计算误差
    err1 = 0
    err2 = 0
    for i in np.arange(x.shape[0]):
        err1 += abs(k1*x[i] - y[i] + b1) / math.sqrt(k1**2 + 1)
        err2 += abs(k2*y[i] - x[i] + b2) / math.sqrt(k2**2 + 1)
    if err1 <= err2:
        return (math.atan(k1) + 2*math.pi) % (2*math.pi)
    else:
        # 计算沿45°对称的角
        return (math.pi/2. - math.atan(k2) + 2*math.pi) % (2*math.pi)

def isPtIn(pt, angle, img):
    """
    判断以pt为原点，angle方向的点是否为255
    pt: [x, y]
    angle: 弧度
    img: 二值图
    """
    row, col = pt
    row_1 = int(row - 2 * math.sin(angle))
    col_1 = int(col + 2 * math.cos(angle))
    if img[row_1, col_1] == 0:
        return angle % (2 * math.pi)
    else:
        return (angle + math.pi) % (2 * math.pi)

def collision_detection(pt, dep, radians, depth_map):
    """
    碰撞检测
    pt: (row, col)
    radians: 弧度
    dep: 抓取点的平行深度值
    depth_map: 平行深度图

    return:
        True: 无碰撞
        False: 有碰撞
    """
    row, col = pt

    # if dep > 0.005:
    #     dep -= 0.005

    # 两个点
    # row_1 = int(row - 0.005 * math.sin(radians) / unit)
    row_1 = int(row - GRASP_GAP * math.sin(radians) / unit)
    # col_1 = int(col + 0.005 * math.cos(radians) / unit)
    col_1 = int(col + GRASP_GAP * math.cos(radians) / unit)
    row_2 = int(row - (GRASP_GAP + FINGER_L2_1) * math.sin(radians) / unit)
    col_2 = int(col + (GRASP_GAP + FINGER_L2_1) * math.cos(radians) / unit)
    
    # 在截面图上绘制抓取器矩形
    # 检测截面图的矩形区域内是否有1
    gripper_w = FINGER_L1_1 / unit
    rows, cols = ptsOnRotateRect([row_1, col_1], [row_2, col_2], gripper_w)

    if np.max(depth_map[rows, cols]) >= dep:   # 无碰撞
        return True
    return False    # 有碰撞
    
def calcNumsInStride(num1, num2, stride):
    """
    计算从num1 到num2 之间位于stride步长上的数，即stride的整数倍
    num1: float
    num2: float  num1 < num2
    stride: float
    """
    bottom = int(round(num1 / stride)) * stride
    if bottom < num1:
        bottom += stride
    top = int(num2 / stride) * stride

    result = []
    i = 0
    while True:
        num = bottom + stride * i
        if num < top:
            result.append(num)
            i += 1
        else:
            break
    result.append(top)

    return result
    
def calcDownPts(pts, depth_map, stride, depth_down):
    """
    根据深度步长计算下降点，并保留连续下降超过depth_down的点

    pts: 沿抓取角的连续坐标点  第一个点为候选抓取点 list([x, y])
    depth_map: 桌面深度图
    stride: 深度步长 m
    depth_down: 下降沿深度阈值 = 水平步长 * tan(80)

    return: list([x, y, dep, max_depth])   
        x, y 是像素坐标 (不准确，只是作为暂时的xy坐标)
        dep 是实际桌面深度
        只有下降点有实际值，非下降点为[0, 0, 0]
        max_depth: 从中心点到该点的最大桌面深度
    """
    max_depth = 0
    # 根据深度步长计算下降点
    pts_down = [0]
    for i in range(len(pts))[1:]:
        pt1, pt2 = pts[i-1], pts[i]
        dep1, dep2 = depth_map[pt1[1], pt1[0]], depth_map[pt2[1], pt2[0]]
        max_depth = max(max(max_depth, dep1), dep2)

        # 计算是否为下降点
        if dep2 - dep1 > depth_down:
            if pts_down[-1] is 0:
                pts_down.append([pt1[0], pt1[1], dep1, max_depth])
            # 计算pt1 pt2之间的下降点 (x, y, dep)   dep = stride的整数倍
            deps = calcNumsInStride(dep1, dep2, stride)[::-1]   # 由小到大
        
            for dep in deps:
                x = int(pt1[0] - (pt1[0] - pt2[0]) * (dep1 - dep) / (dep1 - dep2))
                y = int(pt1[1] - (pt1[1] - pt2[1]) * (dep1 - dep) / (dep1 - dep2))
                pts_down.append([x, y, dep, max_depth])
        else:
            if pts_down[-1] is not 0:
                pts_down.append(0)
    
    if pts_down[-1] is 0:
        del pts_down[-1]

    # 计算连续下降点
    pts_down_grasp = []
    start_down_pt = []
    for i in range(len(pts_down)):
        # 计算当前点与前面最靠近0的点的深度差
        pt = pts_down[i]
        if pt is 0:
            start_down_pt = pts_down[i+1]
            continue
        else:
            if pt[2] - start_down_pt[2] >= GRASP_DEPTH:
                pts_down_grasp.append(pt)
        
    # return pts_down       # 下降沿
    return pts_down_grasp   # 连续下降沿

def fitGraspEdgeAngle(pt, depth, depth_map):
    """
    拟合抓取边缘角
    pt: 抓取边缘点(抓取下降点) [row, col]
    depth: 抓取处的平行深度值
    depthMap: 平行深度图

    return: 
        r: 弧度
        pts_rest: list([x, y])
    """
    
    fit_l = int(FINGER_L1_1 / unit)    # 计算拟合角的点个数
    half_l = int(fit_l / 2)

    # (1) 以 pt 截取 2fit_l*2fit_l 的 深度图
    row, col = pt
    depth_cut = depth_map[row-fit_l : row+fit_l+1, col-fit_l : col+fit_l+1]
    _, im_thresh = cv2.threshold(depth_cut, depth, 255, cv2.THRESH_BINARY_INV)
    # _, im_thresh = cv2.threshold(depth_cut, depth+1e-4, 255, cv2.THRESH_BINARY)
    im_thresh = im_thresh.astype(np.uint8)
    # cv2.imshow('im_thresh', im_thresh)
    # cv2.waitKey()

    # (2) 检测轮廓, 获取 [row, col] 附近10个点坐标 list()
    contours, _ = cv2.findContours(im_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 获取所有轮廓中离中心点最近的点，即该点所在的contour
    contours_np = np.squeeze(np.concatenate(tuple(contours), axis=0))           # (n, 2)
    fit_pt = np.array([[fit_l, fit_l]]).repeat(contours_np.shape[0], axis=0)    # (n, 2)
    dists = contours_np - fit_pt
    dists = np.square(dists[:, 0]) + np.square(dists[:, 1])
    argmin = int(np.argmin(dists))

    pt_in_contour = contours_np[argmin].tolist()
    contourEdge_pts = []
    pt_idx = argmin

    contour_id = -1
    for contour in contours:
        # contour_pts = np.squeeze(contour).tolist()
        # contourEdge_pts = contour_pts
        pt_idx = argmin
        contour_id += 1
        argmin -= contour.shape[0]
        if argmin < 0:
            break
    
    contourEdge_pts = np.squeeze(contours[contour_id])  # (m, 2)

    # 获取 pt_in_contour 附近点坐标 list()
    pts = []
    if pt_idx - half_l < 0:
        pts = np.concatenate((contourEdge_pts[pt_idx - half_l :], contourEdge_pts[0 : pt_idx + (half_l + 1)]), axis=0)
    elif pt_idx + (half_l + 1) > contourEdge_pts.shape[0]:
        pts = np.concatenate((contourEdge_pts[pt_idx - half_l:], contourEdge_pts[0 : pt_idx + (half_l + 1) - contourEdge_pts.shape[0]]), axis=0)
    else:
        pts = contourEdge_pts[pt_idx - half_l : pt_idx + (half_l + 1)]

    # 计算这些点在原图中的坐标  [x y]
    if len(pts.shape) == 1:
        return 1001, None
    pts[:, 0] += col-fit_l  
    pts[:, 1] += row-fit_l
    pts_ret = np.copy(pts)

    # 拟合斜率 linear_regression, 根据k计算抓取角=斜率的垂线
    pts[:, 1] *= -1     # 转换坐标系
    ang = angle_regression(pts[:, 0], pts[:, 1])
    r = isPtIn(pt_in_contour, ang+math.pi/2, im_thresh)

    return r, pts_ret


class Grasp:
    """
    抓取配置类
    """
    def __init__(self, camera):
        self.camera = camera

        self.grasps = []         # 记录抓取点和抓取角
        self.grasp_widths = []   # 记录抓取宽度
        self.grasp_depths = []   # 记录抓取器下边缘距抓取宽度上最高点的深度
        # self.edgeVars = []      # 记录抓取边缘的方差值
        
        # self.detected_pt_pair = []   # 记录标注过的点对(桌面深度图坐标, 通过斜率计算的虚拟点对)

    def reinit(self):
        self.detected_pt_pair = []
    

    # 以边缘方差作为替换条件
    def recordGrasp_edgeVar(self, table_pt, depth, grasp_table_depth, angle_bin, width, edgeVar):
        """
        记录抓取配置

        param:
            table_pt: 两个抓取沿的中心点坐标 [row, col] 
            depth: table_pt 处的深度(其实是抓取方向上最高的深度) m
            grasp_depth: 抓取器下边缘的深度(相比于桌面的深度)  m
            angle_bin: 抓取角类别 [0, BINs)
            width: 抓取宽度，两个抓取沿之间的实际距离 m
            edgeVar: 两边缘的方差最大值
        """
        # 将 table_pt 映射到世界坐标系
        grasp_x = (table_pt[1] - (TABLE_IMG_WIDTH - 1) / 2) * unit
        grasp_y = -1 * (table_pt[0] - (TABLE_IMG_HEIGHT - 1) / 2) * unit
        # 映射到像素坐标系
        camera_row, camera_col = self.camera.world2img([grasp_x, grasp_y, grasp_table_depth])
        assert camera_row >= 0 and camera_row < HEIGHT
        assert camera_col >= 0 and camera_col < WIDTH

        # 查重
        if self.grasp.count([camera_row, camera_col, angle_bin]):
            idx = self.grasp.index([camera_row, camera_col, angle_bin])
            if edgeVar < self.edgeVars[idx]:
                self.edgeVars[idx] = edgeVar
                self.grasp_width[idx] = width
                self.grasp_table_depth[idx] = grasp_table_depth
        else:
            self.grasp.append([camera_row, camera_col, angle_bin])
            self.edgeVars.append(edgeVar)
            self.grasp_width.append(width)
            self.grasp_table_depth.append(grasp_table_depth) # 相机离抓取点的深度

        print ("\r record grasp:", camera_row, camera_col, angle_bin, width, end="")
        # print ("\rrecord grasp:", grasp_x, grasp_y, angle_bin, width, end="")

    # 以抓取深度作为替换条件
    def recordGrasp(self, camera_grasp_pt, grasp_depth, grasp_angle_bin, grasp_width):
        """
        记录抓取配置

        param:
            camera_grasp_pt: 相机深度图中的抓取点坐标 [row, col] 
            grasp_depth: 抓取深度, 抓取器下边缘距抓取宽度上最高点的深度
            angle_bin: 抓取角类别 [0, BINs) 逆时针
            width: 抓取宽度 m
        """
        # 查重
        if self.grasps.count([camera_grasp_pt[0], camera_grasp_pt[1], grasp_angle_bin]):
            idx = self.grasps.index([camera_grasp_pt[0], camera_grasp_pt[1], grasp_angle_bin])
            if grasp_depth > self.grasp_depths[idx]:
                self.grasp_widths[idx] = grasp_width
                self.grasp_depths[idx] = grasp_depth
        else:
            self.grasps.append([camera_grasp_pt[0], camera_grasp_pt[1], grasp_angle_bin])
            self.grasp_widths.append(grasp_width)
            self.grasp_depths.append(grasp_depth)

        print ("\r record grasp:", camera_grasp_pt[0], camera_grasp_pt[1], grasp_angle_bin, grasp_width, grasp_depth, end="")

    
    def grasp_in_world(self, table_pt):
        """
        将位于桌面深度图上的抓取点映射到世界坐标系
        """
        # 将 table_pt 映射到世界坐标系
        grasp_x = (table_pt[1] - (TABLE_IMG_WIDTH - 1) / 2) * unit
        grasp_y = -1 * (table_pt[0] - (TABLE_IMG_HEIGHT - 1) / 2) * unit

        return grasp_x, grasp_y

    def recordDetected(self, pts_down_1, pts_down_2):
        """
        记录标注过的点对(桌面深度图坐标, 通过斜率计算的虚拟点对)

        pts_down_1: [x, y, depth]   x, y是像素坐标,depth是实际深度
        """
        x1 = int(pts_down_1[0] * unit / 0.0005)      # 相对于左上角的实际坐标
        y1 = int(pts_down_1[1] * unit / 0.0005)
        x2 = int(pts_down_2[0] * unit / 0.0005)
        y2 = int(pts_down_2[1] * unit / 0.0005)

        self.detected_pt_pair.append([x1, y1, x2, y2])

    def check_in_camera(self, table_pt, depth, angle_bin, width):
        """
        抓取配置查重,基于相机深度图

        param:
            table_pt: 两个抓取沿的中心点坐标 [row, col] 
            depth: table_pt 处的深度
            angle_bin: 弧度 [0, pi)
            width: 抓取宽度，两个抓取沿之间的实际距离 m
        """
        # 将 table_pt 映射到世界坐标系
        grasp_x = (table_pt[1] - (TABLE_IMG_WIDTH - 1) / 2) * unit
        grasp_y = -1 * (table_pt[0] - (TABLE_IMG_HEIGHT - 1) / 2) * unit
        # 映射到像素坐标系
        camera_row, camera_col = self.camera.world2img([grasp_x, grasp_y, depth])

        if self.grasp.count([camera_row, camera_col, angle_bin]):
            return True
        
        return False

    def check_in_table(self, pts_down_1, pts_down_2):
        """
        抓取配置查重,基于桌面深度图
        """
        x1 = int(pts_down_1[0] * unit / 0.0005)      # 相对于左上角的实际坐标
        y1 = int(pts_down_1[1] * unit / 0.0005)
        x2 = int(pts_down_2[0] * unit / 0.0005)
        y2 = int(pts_down_2[1] * unit / 0.0005)

        if self.detected_pt_pair.count([x1, y1, x2, y2]):
            return True

        return False
    
    def save_grasp_mat(self, save_path):
        """
        生成抓取图,在计算完所有抓取配置后调用，保存为mat文件

        抓取坐标图  shape=(h, w)        int[0/1]
        抓取角度图    shape=(h, w, cls)   int[0/1] 
        抓取宽度图  shape=(h, w)        float 单位m  
        抓取深度图  shape=(h, w)        float 单位m  相机离抓取点的深度
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.grasp_point_map = np.zeros((HEIGHT, WIDTH), np.uint8)
        self.grasp_angle_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.uint8)
        self.grasp_width_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.float64)
        self.grasp_depth_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.float64)    # 相机离抓取点的深度
        # self.edgeVar_map = np.zeros((HEIGHT, WIDTH, GRASP_ANGLE_BINS), np.float64)

        for i in range(len(self.grasp)):
            camera_row, camera_col, angle_bin = self.grasp[i]
            self.grasp_point_map[camera_row, camera_col] = 1
            self.grasp_angle_map[camera_row, camera_col, angle_bin] = 1
            self.grasp_width_map[camera_row, camera_col, angle_bin] = self.grasp_width[i]
            self.grasp_depth_map[camera_row, camera_col, angle_bin] = self.camera.camera_height() - self.grasp_table_depth[i] # 相机离抓取点的深度
        
        print('生成抓取: ', len(self.grasp))
        
        scio.savemat(save_path + '/grasp_point_map.mat', {'A':self.grasp_point_map})
        scio.savemat(save_path + '/grasp_angle_map.mat', {'A':self.grasp_angle_map})
        scio.savemat(save_path + '/grasp_width_map.mat', {'A':self.grasp_width_map})
        scio.savemat(save_path + '/grasp_depth_map.mat', {'A':self.grasp_depth_map})    # 相机离抓取点的深度

    def save_grasp_txt(self, save_path):
        """
        生成抓取图,在计算完所有抓取配置后调用，保存为txt文件
        save_path: .../graspLabel.txt
        """
        f = open(save_path, 'w')
        for i in range(len(self.grasps)):
            grasp_row, grasp_col, angle_bin = self.grasps[i]
            grasp_width = self.grasp_widths[i]
            grasp_depth = self.grasp_depths[i]
            line = str(grasp_row) + ' ' + str(grasp_col) + ' ' + str(angle_bin) +  ' ' + str(grasp_width) + ' ' + str(grasp_depth) + '\n'
            f.write(line)
        f.close()
        print('生成抓取: ', len(self.grasps))


def calcPtsVar(pts, angle):
    """
    计算pts沿angle方向的方差

    pts: list([x, y])
    angle: 反向抓取弧度(朝向物体外部)

    return:
        var: 方差
        pt_touch: 抓取器与物体接触的点坐标 [x, y]
    """
    angle_v = angle + math.pi / 2
    # (1) 计算pts的平均点
    pts_ = np.array(pts)
    pts_[:, 1] *= -1
    pt_x, pt_y = np.mean(pts_[:, 0]), np.mean(pts_[:, 1])
    
    # (2) 计算一个沿angle方向较远的直线，ax+by+1=0
    y_1 = pt_y + 1000 * math.sin(angle)
    x_1 = pt_x + 1000 * math.cos(angle)
    if angle_v == math.pi / 2 or angle_v == 3 * math.pi / 2:
        a = -1 / x_1
        b = 0
    else:
        K = math.tan(angle_v)
        B = y_1 - K * x_1
        b = -1 / B
        a = -1 * K * b

    # 计算方差，同时记录距离直线最近的点
    min_dis = 1000000
    pt_touch = [0, 0]
    var = 0
    for pt in pts_:
        x, y = pt
        dis = abs(a*x + b*y + 1) / math.sqrt(a**2 + b**2)
        if dis < min_dis:
            min_dis = dis
            pt_touch = [int(x), int(-1 * y)]
        var += (dis - 1000) ** 2
    var /= pts_.shape[0]

    return var, pt_touch


def calcGripperPt(pt_edge, pt_touch, grasp_angle):
    """
    计算抓取器与物体接触时，抓取器中心点坐标
    param:
        pt_edge: 在抓取方向上 物体的边缘点 [x, y]
        pt_touch: 抓取器与物体接触的点坐标 [x, y]
        grasp_angle: 抓取角 弧度
    
    return:
        [x, y]
    """
    if pt_edge[0] == pt_touch[0] and pt_edge[1] == pt_touch[1]:
        return pt_edge

    pt_edge_ = [pt_edge[0], pt_edge[1] * -1]
    pt_touch_ = [pt_touch[0], pt_touch[1] * -1]
    # 计算距离
    L = tool.distancePt(pt_edge_, pt_touch_)
    # 计算夹角
    angle_1 = tool.calcAngleOfPts(pt_edge_, pt_touch_)
    angle_tri = abs(angle_1 - grasp_angle)
    angle_tri = min(angle_tri, 2*math.pi-angle_tri)

    if angle_tri > math.pi / 2:
        # print('angle_tri = ', tool.radians_TO_angle(angle_tri))
        return pt_edge

    # 计算前进距离
    L_ = L * math.cos(angle_tri)
    pt_gripper_x = int(pt_edge_[0] + L_ * math.cos(grasp_angle))
    pt_gripper_y = int(-1 * (pt_edge_[1] + L_ * math.sin(grasp_angle)))

    return [pt_gripper_x, pt_gripper_y]


def getGrasp(camera_depth, camera_mask, table_depth, table_mask, camera:Camera, save_path):
    """
    camera_depth: 相机深度图 np.float
    camera_mask: 相机mask np.int
    table_depth: 平行深度图 np.float
    table_mask: 平行mask np.int
    camera: Camera实例
    save_path: 抓取标签的保存路径
    """
    # 读取图像
    # 相机渲染的图像
    # camera_rgb = scio.loadmat(path + '/camera_rgb.mat')['A'] 
    # camera_depth = scio.loadmat(path + '/camera_depth.mat')['A']
    # camera_depth_rev = scio.loadmat(path + '/camera_depth_rev.mat')['A']
    # camera_mask = scio.loadmat(path + '/camera_mask.mat')['A'].astype(np.uint8)
    # 空间渲染的图像
    # table_depth = scio.loadmat(path + '/table_depth.mat')['A']
    # table_mask = scio.loadmat(path + '/table_mask.mat')['A']

    # cv2.imwrite('img/img_test/table_mask.png', table_mask*20)

    # 转三通道
    # camera_depth_gray = tool.depth2Gray(camera_depth).copy()
    # camera_depth_3c = tool.depth3C(camera_depth_gray)   # 相机深度图
    # table_depth_gray = tool.depth2Gray(table_depth).copy()
    # table_depth_3c = tool.depth3C(table_depth_gray)     # 桌面深度图

    # 初始化抓取配置类
    myGrasp = Grasp(camera)
    
    # 迭代相机深度图的物体上的点
    ids_objs = list(np.unique(camera_mask)[2:]) # 去除地面和托盘
    print('all_objs = ', np.unique(camera_mask))
    print('ids_objs = ', ids_objs)

    assert len(ids_objs) > 0

    sum_time = 0    # 记录总耗时

    # ========================= 遍历物体 =========================
    for id_obj in ids_objs:
        print('>> 计算抓取配置: ', id_obj)
        t = time.time()

        # myGrasp.reinit()

        # ========================= 遍历物体上的点 =========================
        
        # 获取物体上的点
        state = False
        rows, cols = np.where(camera_mask == id_obj)
        for i in range(rows.shape[0]):
            if i % 100 == 0:
                # 可视化    
                # print('  保存图像: ({}/{})'.format(i, rows.shape[0]))
                print('进度: ({}/{})'.format(i, rows.shape[0]))
                # cv2.imwrite('img/img_test/table_depth_3c.png', table_depth_3c)
                # cv2.waitKey()
                # state = False

            # 跳过点
            # if i < 1000:
            #     continue

            # 临时可视化图像
            # camera_depth_3c_test = camera_depth_3c.copy()
            # table_depth_3c_test = table_depth_3c.copy()

            # ===================== 1 获取相机深度图中的点 =====================
            camera_row, camera_col = rows[i], cols[i]
            # print('>> 正在检测:({}/{}) - '.format(i, rows.shape[0]), camera_row, camera_col)

            # ===================== 2 将相机深度图中的点映射为真实深度图中的三维做标签 =====================
            # 像素坐标->相机坐标系
            depth = camera_depth[camera_row, camera_col]
            coordInCamera = list(camera.img2camera([camera_col, camera_row], depth))  # [x,y,z]
            col_table = int(coordInCamera[0] / unit + (TABLE_IMG_WIDTH - 1) / 2)
            row_table = int(coordInCamera[1] / unit + (TABLE_IMG_HEIGHT - 1) / 2)

            # cv2.circle(camera_depth_3c_test, (camera_col, camera_row), 2, (0, 255, 0), -1)
            # cv2.circle(table_depth_3c_test, (col_table, row_table), 10, (0, 255, 0), -1)
            # cv2.imshow('camera_depth_3c', camera_depth_3c_test)
            # cv2.imshow('table_depth_3c', cv2.resize(table_depth_3c_test, (1000, 1000)))
            # cv2.waitKey()

            if table_mask[row_table, col_table] != id_obj:
                continue

            # ===================== 3 迭代该点的每个角度，共18个角度 =====================
            for grasp_angle_bin in range(GRASP_ANGLE_BINS):     # 18

                grasp_angle_1 = (grasp_angle_bin / GRASP_ANGLE_BINS) * math.pi    # 弧度  [0, pi)
                grasp_angle_2 = grasp_angle_1 + math.pi    # 弧度  [pi, 2*pi)

                # ===================== 5 获取抓取点两侧的下降点 =====================
                # 1、沿抓取角两侧计算深度变化
                # (1) 按步长获取抓取角两侧的在物体上的坐标点（最后一个点不在物体上）
                pts_1 = ptsAlongAngle([col_table, row_table], grasp_angle_1, GRASP_STRIDE, table_mask, id_obj, GRASP_MAX_W)
                if len(pts_1) < 2:
                    continue
                pts_2 = ptsAlongAngle([col_table, row_table], grasp_angle_2, GRASP_STRIDE, table_mask, id_obj, GRASP_MAX_W)
                if len(pts_2) < 2:
                    continue

                # (2) 计算固定深度步长的下降点
                pts_down_1 = calcDownPts(pts_1, table_depth, DEATH_STRIDE, DEATH_DOWN_THRESH)   # list([x, y, dep])   
                pts_down_2 = calcDownPts(pts_2, table_depth, DEATH_STRIDE, DEATH_DOWN_THRESH)
                if len(pts_down_1) == 0 or len(pts_down_2) == 0:
                    continue
                    
                # for pt in pts_down_1:
                #     cv2.circle(table_depth_3c_test, (pt[0], pt[1]), 1, (0, 0, 255), -1)
                # for pt in pts_down_2:
                #     cv2.circle(table_depth_3c_test, (pt[0], pt[1]), 1, (0, 0, 255), -1)

                # cv2.imwrite(save_path + '/table_depth_test.png', table_depth_3c_test)
                # cv2.waitKey()
                    
                
                # ===================== 6 迭代两侧同深度的下降点 =====================
                """
                不仅要相同深度，而且距离中心点的宽度相同（大概相同），最后以中心点作为抓取点
                """

                for pt_down_1 in pts_down_1:
                    for pt_down_2 in pts_down_2:
                        if pt_down_1[2] != pt_down_2[2]:
                            continue
                        if max(pt_down_1[3], pt_down_2[3]) - pt_down_1[2] > FINGER_L3:
                            continue
                        
                        # 与中心点的距离之差大于阈值(一个像素)，跳过
                        dist1 = tool.distancePt([pt_down_1[0], pt_down_1[1]], [col_table, row_table])
                        dist2 = tool.distancePt([pt_down_2[0], pt_down_2[1]], [col_table, row_table])
                        if abs(dist1 - dist2) > (0.002 / unit):
                            continue

                        # 距离筛选
                        if tool.distancePt([pt_down_1[0], pt_down_1[1]], [pt_down_2[0], pt_down_2[1]]) >= ((GRASP_MAX_W-2*GRASP_GAP) / unit):
                            continue
                        
                        # 1 =========================== 拟合角检测  最耗时 ===========================
                        """
                        对抓取深度上下不同深度位置取轮廓检测边缘上的点，可以计算真正的拟合角
                        """

                        # (1) 计算 抓取下降点的向外垂直法线角，弧度
                        #     和 
                        #     进行拟合的物体上的点序列; 中间的点是位于抓取方向上, 当抓取器与物体接触时，离抓取器中心点最近的坐标点
                        grasp_edge_angle_1, pts_edge_1 = fitGraspEdgeAngle([pt_down_1[1], pt_down_1[0]], pt_down_1[2], table_depth)
                        if grasp_edge_angle_1 == 1001:
                            continue
                        if abs(grasp_angle_1 - grasp_edge_angle_1) > math.radians(GRASP_ANGLE_RANGE):
                            continue

                        grasp_edge_angle_2, pts_edge_2 = fitGraspEdgeAngle([pt_down_2[1], pt_down_2[0]], pt_down_2[2], table_depth)
                        if grasp_edge_angle_2 == 1001:
                            continue
                        if abs(grasp_angle_2 - grasp_edge_angle_2) > math.radians(GRASP_ANGLE_RANGE):
                            continue
                        
                        # 抓取方向上, 离抓取器中心点最近的坐标点(在物体上)
                        pt_edge_1 = pts_edge_1[int((pts_edge_1.shape[0]-1)/2)]    # x, y
                        pt_edge_2 = pts_edge_2[int((pts_edge_2.shape[0]-1)/2)]

                        # 2 =========================== 平稳检测 ===========================
                        # (1) 计算 pts_edge_1 的平均点，使用grasp_angle_1作为斜率
                        #     同时计算抓取器与物体的接触坐标点 [x, y]
                        var1, pt_touch_1 = calcPtsVar(pts_edge_1, grasp_angle_1)
                        if var1 > EDGE_VAR_THRESH:
                            continue
                        var2, pt_touch_2 = calcPtsVar(pts_edge_2, grasp_angle_2)
                        if var2 > EDGE_VAR_THRESH:
                            continue

                        # (2) 计算 抓取器与物体接触时，抓取器中心点坐标
                        pt_gripper_1 = calcGripperPt(pt_edge_1, pt_touch_1, grasp_angle_1)
                        pt_gripper_2 = calcGripperPt(pt_edge_2, pt_touch_2, grasp_angle_2)

                        # 3 =========================== 第二次抓取点查重 ===========================
                        # 计算抓取宽度(m)
                        grasp_touch_w = tool.distancePt(pt_gripper_1, pt_gripper_2) * unit + 2 * GRASP_GAP  # 抓取宽度 m

                        # 距离筛选
                        if grasp_touch_w > GRASP_MAX_W:
                            continue
                        
                        # 4 =========================== 碰撞检测 ===========================
                        if not collision_detection((pt_gripper_1[1], pt_gripper_1[0]), pt_down_1[2], grasp_angle_1, table_depth):
                            continue
                        if not collision_detection((pt_gripper_2[1], pt_gripper_2[0]), pt_down_2[2], grasp_angle_2, table_depth):
                            continue
                        
                        # =========================== 记录抓取配置 ===========================
                        # 相对于抓取宽度上的最高点，抓取点的深度值
                        grasp_depth = max(pt_down_1[3], pt_down_2[3]) - pt_down_1[2]
                        myGrasp.recordGrasp([camera_row, camera_col], grasp_depth, grasp_angle_bin, grasp_touch_w)

                        # =========================== 可视化 ===========================
                        # table_depth_3c_test = table_depth_3c.copy()   # 临时可视化图像
                        # cv2.line(table_depth_3c, (pt_gripper_1[0], pt_gripper_1[1]), (pt_gripper_2[0], pt_gripper_2[1]), (0, 0, 255))  # 抓取连线
                        # cv2.circle(table_depth_3c, (grasp_center_table_col, grasp_center_table_row), 2, (0, 255, 0), -1) # 迭代点

                        # cv2.circle(table_depth_3c, (pt_down_1[0], pt_down_1[1]), 2, (255, 0, 0), -1) # 迭代点
                        # cv2.circle(table_depth_3c, (pt_down_2[0], pt_down_2[1]), 2, (255, 0, 0), -1) # 迭代点

                        # state = True

                        # cv2.imshow('table_depth_3c_test', cv2.resize(table_depth_3c_test, (1000, 1000)))
                        # cv2.imwrite('img/img_test/table_depth_3c_test.png', table_depth_3c_test)
                        # cv2.waitKey()


        print('物体: {}, 耗时: {:.4f}s'.format(id_obj, time.time()-t))        
        sum_time += time.time()-t
    # 保存抓取配置
    myGrasp.save_grasp_txt(save_path + '/graspLabel.txt')    # 保存为txt文件
    print('>> 保存抓取配置, 总耗时: {:.1f}s'.format(sum_time))


if __name__ == "__main__":
    # img_path = 'F:/sim_grasp2/imgs/05_4505-4550-self'
    # file_names = os.listdir(img_path)
    # file_names.sort()
    # for file_name in file_names:
    #     print('filename: ', file_name)
    #     path = os.path.join(img_path, file_name)
    #     # try:
    #     getGrasp(path)
    #     # except:
    #     #     print('计算抓取失败')
    #     #     pass

    getGrasp(path='E:/research/1-paper/sim_grasp/latex/figures/label/render')

