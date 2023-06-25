"""
人工添加深度图像噪声
"""
import cv2
import sys
sys.path.append(os.curdir)
import utils.tool as tool
import time
import numpy as np
import scipy.io as scio
import scipy.stats as ss
import skimage.transform as skt


grad = None


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global grad
    if event == cv2.EVENT_LBUTTONDOWN:
        print(grad[y][x])


def imresize(image, size, interp="nearest"):
    """Wrapper over `skimage.transform.resize` to mimic `scipy.misc.imresize`.
    Copied from https://github.com/BerkeleyAutomation/perception/blob/master/perception/image.py#L38.  # noqa: E501

    Since `scipy.misc.imresize` has been removed in version 1.3.*, instead use
    `skimage.transform.resize`. The "lanczos" and "cubic" interpolation methods
    are not supported by `skimage.transform.resize`, however there is now
    "biquadratic", "biquartic", and "biquintic".

    Parameters
    ----------
    image : :obj:`numpy.ndarray`
        The image to resize.

    size : int, float, or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp : :obj:`str`, optional
        Interpolation to use for re-sizing ("neartest", "bilinear",
        "biquadratic", "bicubic", "biquartic", "biquintic"). Default is
        "nearest".

    Returns
    -------
    :obj:`np.ndarray`
        The resized image.
    """
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


def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


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
    im_depth: 单位 mm或m
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color


def gaussian_noise(im_depth):
    """
    在image上添加高斯噪声，参考dex-net代码

    im_depth: 浮点型深度图，单位为米
    """
    gamma_shape = 1000.00
    gamma_scale = 1 / gamma_shape
    gaussian_process_sigma = 0.004  # 0.002
    gaussian_process_scaling_factor = 8.0

    im_height, im_width = im_depth.shape
    
    # 1
    # mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale, size=1) # 生成一个接近1的随机数，shape=(1,)
    # mult_samples = mult_samples[:, np.newaxis]
    # im_depth = im_depth * np.tile(mult_samples, [im_height, im_width])  # 把mult_samples复制扩展为和camera_depth同尺寸，然后相乘
    
    # 2
    gp_rescale_factor = gaussian_process_scaling_factor     # 4.0
    gp_sample_height = int(im_height / gp_rescale_factor)   # im_height / 4.0
    gp_sample_width = int(im_width / gp_rescale_factor)     # im_width / 4.0
    gp_num_pix = gp_sample_height * gp_sample_width     # im_height * im_width / 16.0
    gp_sigma = gaussian_process_sigma

    gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height, gp_sample_width)  # 生成(均值为0，方差为scale)的gp_num_pix个数，并reshape
    print('高斯噪声最大误差:', gp_noise.max(), gp_noise.min())
    gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")  # resize成图像尺寸，bicubic为双三次插值算法
    # gp_noise[gp_noise < 0] = 0
    # camera_depth[camera_depth > 0] += gp_noise[camera_depth > 0]
    im_depth += gp_noise

    return im_depth


def add_missing_val(im_depth):
    """
    添加缺失值
    梯度大的位置，概率大

    im_depth: 浮点型深度图，单位为米
    """
    global grad
    # 获取梯度图
    ksize = 11   # 调整ksize 7 11
    grad_X = np.abs(cv2.Sobel(im_depth, -1, 1, 0, ksize=ksize))    
    grad_Y = np.abs(cv2.Sobel(im_depth, -1, 0, 1, ksize=ksize))
    grad = cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)
    
    cv2.imshow('gradient', tool.depth2Gray(grad))
    # cv2.imwrite('E:/research/grasp_detection/sim_grasp/realsense/grad11.png', tool.depth2Gray(grad))

    # 产生缺失值的概率与梯度值呈正比
    im_height, im_width = im_depth.shape
    gp_rescale_factor = 8.0     # 调整
    gp_sample_height = int(im_height / gp_rescale_factor)   # im_height / 4.0
    gp_sample_width = int(im_width / gp_rescale_factor)     # im_width / 4.0

    # 计算梯度对应的概率，线性方程
    """
    Sobel ksize 与 g1 g2 的对应关系
    ksize = 11  --> 500.0  2000.0
    ksize = 7  --> 1  15
    """
    min_p = 0.001
    max_p = 1.0
    g1, p1 = 500.0, min_p
    g2, p2 = 1500.0, max_p
    prob_k = (p2 - p1) / (g2 - g1)
    prob_b = p2 - prob_k * g2

    prob = grad * prob_k + prob_b
    prob = np.clip(prob, 0, max_p)

    # 生成0-1随机数矩阵
    random_mat = np.random.rand(gp_sample_height, gp_sample_width)
    random_mat = imresize(random_mat, gp_rescale_factor, interp="bicubic")  # 放大
    # cv2.imshow('random_mat', tool.depth2Gray(random_mat))

    # random_mat小于prob的位置缺失
    im_depth[ random_mat < prob ] = 0

    return im_depth

    
    

def run():
    t = time.time()

    # 读取图像
    # F:/my_dense/imgs/all-15/0-99/15_000075_0
    im_depth = scio.loadmat('F:/sim_grasp/imgs/done/15_000000_0/camera_depth.mat')['A']   # (h, w) m
    cv2.imshow('im_depth', tool.depth2Gray(im_depth))

    # 添加高斯噪声
    im_depth_noise = gaussian_noise(im_depth)    
    cv2.imshow('im_depth_noise', tool.depth2Gray(im_depth_noise))
    t1 = time.time()

    cv2.namedWindow("gradient")
    cv2.setMouseCallback("gradient", on_EVENT_LBUTTONDOWN)

    # 添加缺失值
    im_depth_missing = add_missing_val(im_depth_noise)
    cv2.imshow('im_depth_missing', tool.depth2Gray(im_depth_missing))
    t2 = time.time()

    # 彩色可视化
    zeros_loc = np.where(im_depth_missing == 0)
    im_depth_missing = inpaint(im_depth_missing, missing_value=0)
    im_color = depth2RGB(im_depth_missing)
    cv2.imshow('im_color1', im_color)
    im_color[zeros_loc] = (255, 255, 255)
    cv2.imshow('im_color', im_color)

    print('高斯噪声 耗时: ', t1 - t)
    print('添加缺失值 耗时: ', t2 - t1)
    print('总 耗时: ', t2 - t)

    cv2.waitKey()


if __name__ == "__main__":
    run()