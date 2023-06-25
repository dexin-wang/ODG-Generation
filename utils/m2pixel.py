'''
Description: 将实际长度转化为相机深度图中的像素长度
Author: wangdx
Date: 2021-09-08 15:51:16
LastEditTime: 2021-09-08 16:04:12
根据仿真相机的内参计算
'''

import numpy as np
import math

length_m = 0.002    # 实际长度 单位m


# ============= 计算相机内参 =============
WIDTH = 640
HEIGHT = 480
fov = 60   # 垂直视场
length = 0.7
H = length * math.tan(np.deg2rad(fov/2))   # 图像第一行的中点到图像中心点的实际距离 m
W = WIDTH * H / HEIGHT     # 图像右方点的到中心点的实际距离 m
A = (HEIGHT / 2) * length / H
# 计算内参
InMatrix = np.array([[A, 0, WIDTH/2 - 0.5], [0, A, HEIGHT/2 - 0.5], [0, 0, 1]], dtype=np.float64)

# 计算图像左上角点到右下角点对应的实际长度
# 计算公式参考：https://blog.csdn.net/lyhbkz/article/details/82254069
lt = np.array([0, 0, 1]).reshape((3, 1))
Z = 0.7
lt_m = np.matmul(np.linalg.inv(InMatrix), lt) * Z
lt_m = math.sqrt(lt_m[0][0] ** 2 + lt_m[1][0] ** 2) # 从图像中心点到左上角的实际长度    m
lt_pixel = math.sqrt(240 ** 2 + 320 ** 2)           # 从图像中心点到左上角的像素长度

print(length_m / lt_m * lt_pixel)



