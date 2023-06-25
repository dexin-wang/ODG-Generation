'''
Description: 可视化单张深度图，彩色
Author: wangdx
Date: 2021-09-13 13:33:18
LastEditTime: 2021-12-27 09:36:34
'''
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import sys
sys.path.append(os.curdir)
from sgdn import inpaint
import skimage.transform as skt
import scipy.io as scio
from demo import add_noise


def depth2Gray3(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    ret = np.expand_dims(ret, 2).repeat(3, axis=2)
    return ret


def depth2Gray(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    return ret


def depth2RGB(im_depth):
    """
    将深度图转至三通道8位彩色图
    先将值为0的点去除，然后转换为彩图，然后将值为0的点设为红色
    (h, w, 3)
    im_depth: 单位 mm
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color


def realsense(im_depth):
    #六、复合多个滤波器
    colorizer = rs.colorizer()        
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()


    # im_depth = decimation.process(im_depth)
    # im_depth = depth_to_disparity.process(im_depth)
    # im_depth = spatial.process(im_depth)
    # im_depth = temporal.process(im_depth)
    # im_depth = disparity_to_depth.process(im_depth)
    # im_depth = hole_filling.process(im_depth)

    colorized_depth = np.asanyarray(colorizer.colorize(im_depth).get_data())
    return colorized_depth


def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")



# 读取图像
file = 'E:/research/grasp_detection/sim_grasp/sgdn/img/realsense/40cm_in/02camera_depth.mat'
# img = cv2.imread(file, -1).astype(np.float) / 1000.0    # m
img = scio.loadmat(file)['A'] # (H, W)

# 缩放
img = imresize(img, 0.25, interp='bilinear')
img = imresize(img, 4.0, interp='bilinear')

# 裁剪
size = 360
H, W = img.shape
l = int((W - size) / 2)
t = int((H - size) / 2)
r = l + size
b = t + size
img = img[t:b, l:r]
zeros_loc = np.where(img == 0)


# 加噪声
img = add_noise(img)


# 补全
img = inpaint(img, missing_value=0)
# resize成input_size
# scale = input_size * 1.0 / img.shape[0]
# img = imresize(img, scale, interp="bilinear")

img_c = depth2RGB(img)
img_c[zeros_loc] = (255, 255, 255)


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = '({:.3f})'.format(img[y][x])
        cv2.circle(img_c, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img_c, color, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
        cv2.imshow("imagec", img_c)


cv2.namedWindow("imagec")
cv2.setMouseCallback("imagec", on_EVENT_LBUTTONDOWN)
cv2.imshow("imagec", img_c)
cv2.imshow("image", img)

while True:
    if cv2.waitKey(30) == 27:  # q-113  esc-27
        break
