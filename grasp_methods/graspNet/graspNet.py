""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', default='./ckpt/checkpoint-rs.tar', help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()



def get_and_process_data(color, depth, workspace_mask, intrinsic, factor_depth=1):
    """
    生成网络可输入的点云和end_points
    color: RGB图像
    depth: 深度图 单位m
    workspace_mask: mask，网络只预测mask内的
    intrinsic: 相机内参 ndarray shape=(3,3)
    """
    # load data
    color = np.array(color[:, :, ::-1], dtype=np.float32) / 255.0
    # depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))   # 单位为mm
    # workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    # meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    # intrinsic = meta['intrinsic_matrix']    # (3,3)
    # factor_depth = meta['factor_depth']     # 1000

    # print('depth.max() = ', depth.max(), 'depth.min() = ', depth.min())
    # print('intrinsic = ', intrinsic)   
    # print('factor_depth = ', factor_depth)

    # generate cloud
    camera = CameraInfo(color.shape[1], color.shape[0], intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

# def get_grasps(net, end_points):
#     # Forward pass
#     with torch.no_grad():
#         end_points = net(end_points)
#         grasp_preds = pred_decode(end_points)
#     gg_array = grasp_preds[0].detach().cpu().numpy()
#     # print('gg_array.shape = ', gg_array.shape)
#     # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
#     gg = GraspGroup(gg_array)
#     return gg


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

# def demo(data_dir):
#     net = get_net()
#     end_points, cloud = get_and_process_data(data_dir)
#     gg = get_grasps(net, end_points)
#     # if cfgs.collision_thresh > 0:
#     #     gg = collision_detection(gg, np.array(cloud.points))

#     print('gg.translations = ', gg.translations)

#     vis_grasps(gg, cloud)


class GraspNetMethod:
    def __init__(self, model, device):
        """
        初始化GraspNet网络
        """
        self.net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        # Load checkpoint
        checkpoint = torch.load(model)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(model, start_epoch))
        # set model to eval mode
        self.net.eval()

    def predict(self, img_rgb, img_dep, input_size, intrinsic, num=1):
        """
        获取抓取位姿

        img_rgb: RGB图像
        img_dep: 深度图 单位m
        intrinsic: 相机内参 ndarray shape=(3,3)
        num: 输出的抓取位姿数量

        return:
            grasp_center: 抓取点（机械手闭合时末端的位置）
            rotation_matrix：旋转矩阵
        """
        # 根据input_size生成mask
        workspace_mask = np.zeros(img_rgb.shape[:2], dtype=np.bool)
        x1 = int((img_rgb.shape[1] - input_size) / 2)
        y1 = int((img_rgb.shape[0] - input_size) / 2)
        workspace_mask[y1:y1+input_size, x1:x1+input_size] = 1

        end_points, cloud = get_and_process_data(img_rgb, img_dep, workspace_mask, intrinsic)
        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        # print('gg_array.shape = ', gg_array.shape)
        # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
        gg = GraspGroup(gg_array)
        gg.nms()
        gg.sort_by_score()
        gg = gg[:num]

        # grasp_center = gg[0].translation
        # rotation_matrix = gg[0].rotation_matrix
        # grasp_width = gg[0].width
        # grasp_depth = gg[0].depth
        return gg[:num]



# if __name__=='__main__':
#     data_dir = 'doc/example_data'
#     demo(data_dir)
