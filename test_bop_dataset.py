import os
import sys
import numpy as np
import yaml
import cv2
import open3d as o3d
import json
import imageio
import matplotlib.pyplot as plt
import threading
import time
import hashlib
from gs_runner import GSRunner
from utils.graphics_utils import rgbd_to_pointcloud

import torch   
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

def center_scale_camera_poses(cam_poses):
    # TODO from pointcloud
    center = np.mean(cam_poses[:, :3, 3], axis = 0)
    scale = 1.5/0.3
    cam_poses[:, :3, 3] = (cam_poses[:, :3, 3] - center) * scale
    return cam_poses


def preprocess_data(
    rgbs, depths, masks, poses, sc_factor=1.0, translation=np.array([0.0, 0.0, 0.0])
):
    """
    @rgbs: np array (N,H,W,3)
    @depths: (N,H,W)
    @masks: (N,H,W)
    @normal_maps: (N,H,W,3)
    @poses: (N,4,4)
    """
    depths[depths < 0.1] = 0.0

    rgbs[masks == 0] = [0, 0, 0]
    depths[masks == 0] = 0.0
    masks = masks[..., None]

    rgbs = (rgbs / 255.0).astype(np.float32)
    depths *= sc_factor
    depths = depths[..., None]
    poses[:, :3, 3] += translation
    poses[:, :3, 3] *= sc_factor
    return rgbs, depths, masks, poses

def fuse_pointcloud(rgbs, depths, masks, glcam_in_obs, K):
    pcdAll = o3d.geometry.PointCloud()
    for (color, depth, mask, c2w) in zip(rgbs, depths, masks, glcam_in_obs):

        mask = mask & (depth > 0.1) & (~np.isnan(depth))

        pcd = rgbd_to_pointcloud(color, depth, K, mask, return_o3d=True)

        obj_in_cam = c2w.copy()
        obj_in_cam[:3, 1:3] *= -1
        pcd.transform(obj_in_cam)
        pcdAll += pcd

        # TODO : remove outlier
        # postprocess pointlcoud remove outlier
        # pcdAll = pcdAll.voxel_down_sample(voxel_size=0.01)
        # cl, ind = pcdAll.remove_statistical_outlier(nb_neighbors=20,
        #                                                std_ratio=1.0)

        #pcdAll = pcdAll.select_by_index(ind)


    pcdAll = pcdAll.voxel_down_sample(voxel_size=0.0005)
    cl, ind = pcdAll.remove_statistical_outlier(nb_neighbors=100,
                                                    std_ratio=1.0)
    #cl, ind = pcdAll.remove_radius_outlier(nb_points=20, radius=0.02)

    #display_inlier_outlier(pcdAll, ind)

    # POINTCLOUD in object coordinate
    pcdAll = pcdAll.select_by_index(ind)
    # world_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcdAll, world_coord])
    return pcdAll

"""
Object 01 (002_master_chef_can): [1.3360, -0.5000, 3.5105]
Object 02 (003_cracker_box): [0.5575, 1.7005, 4.8050]
Object 03 (004_sugar_box): [-0.9520, 1.4670, 4.3645]
Object 04 (005_tomato_soup_can): [-0.0240, -1.5270, 8.4035]
Object 05 (006_mustard_bottle): [1.2995, 2.4870, -11.8290]
Object 06 (007_tuna_fish_can): [-0.1565, 0.1150, 4.2625]
Object 07 (008_pudding_box): [1.1645, -4.2015, 3.1190]
Object 08 (009_gelatin_box): [1.4460, -0.5915, 3.6085]
Object 09 (010_potted_meat_can): [2.4195, 0.3075, 8.0715]
Object 10 (011_banana): [-18.6730, 12.1915, -1.4635]
Object 11 (019_pitcher_base): [5.3370, 5.8855, 25.6115]
Object 12 (021_bleach_cleanser): [4.9290, -2.4800, -13.2920]
Object 13 (024_bowl): [-0.2270, 0.7950, -2.9675]
Object 14 (025_mug): [-8.4675, -0.6995, -1.6145]
Object 15 (035_power_drill): [9.0710, 20.9360, -2.1190]
Object 16 (036_wood_block): [1.4265, -2.5305, 17.1890]
Object 17 (037_scissors): [7.0535, -28.1320, 0.0420]
Object 18 (040_large_marker): [0.0460, -2.1040, 0.3500]
Object 19 (051_large_clamp): [10.5180, -1.9640, -0.4745]
Object 20 (052_extra_large_clamp): [-0.3950, -10.4130, 0.1620]
Object 21 (061_foam_brick): [-0.0805, 0.0805, -8.2435]

"""

# load from bop dataset
dataRootDir = '/home/yjin/repos/gaussian-splatting/bop_output/bop_data/ycbv/train_pbr/000000'
#dataRootDir = '/home/datasets/BOP/ycbv/test/000048'
target_object_id = 11 

cameraInfo = json.load(open(dataRootDir + '/scene_camera.json', 'r'))

# camera K is consistent
K = np.asarray(cameraInfo['1']['cam_K'], dtype=np.float32).reshape(3, 3)
depth_scale = np.asarray(cameraInfo['1']['depth_scale'], dtype=np.float32)

scene_gt = json.load(open(dataRootDir + '/scene_gt.json', 'r'))

mm2m = True

c2ws_gt = []
c2ws = []
rgbs = []
depths = []
masks = []
frameIds = []

max_frame = 100

load_stop = False

frame_id = 0

noise = 0.02
for imgIdx, content in scene_gt.items():
    if load_stop:
        break

    for i, obj_info in enumerate(content):
        if frame_id >= max_frame:
            load_stop = True
            break

        if int(obj_info['obj_id']) == int(target_object_id):

            # object 6D Pose
            c2w = np.eye(4)
            R = np.array(obj_info['cam_R_m2c']).reshape(3,3)
            t = np.array(obj_info['cam_t_m2c'])
            c2w[:3, :3] = R
            c2w[:3, 3] = t
            if mm2m:
                c2w[:3, 3] = c2w[:3, 3] / 1000.0 

            c2w_gt = np.linalg.inv(c2w)
            c2w_gt[:3, 1:3] *= -1 # opengl

            c2w = c2w_gt.copy()
            # add translation and rotation error to camera pose
            if len(c2ws) > 0:
                c2w[:3, 3] += np.random.randn(3) * noise
                c2w[:3, :3] = c2w[:3, :3] @ cv2.Rodrigues(np.random.randn(3) * noise)[0]

            # load rgb, depth, mask
            imgId = int(imgIdx)
            # .png for ycbv test
            #color = np.asarray(cv2.imread(os.path.join(dataRootDir, 'rgb', f'{imgId:06d}.png')), dtype=np.uint8)[:,:, ::-1]
            color = np.asarray(cv2.imread(os.path.join(dataRootDir, 'rgb', f'{imgId:06d}.jpg')), dtype=np.uint8)[:,:, ::-1]

            if False:
                depth = np.asarray(cv2.imread(os.path.join(dataRootDir, 'depth', f'{imgId:06d}.png'), cv2.IMREAD_UNCHANGED), dtype=np.float64)
            else:
                depth = imageio.imread(os.path.join(dataRootDir, 'depth', f'{imgId:06d}.png'))
                depth[depth == 65535] = 0

            if mm2m:   
                depth = depth.astype(np.float64) / 1000. #/ depth_scale

            mask  = np.asarray(cv2.imread(os.path.join(dataRootDir, 'mask_visib', f'{imgId:06d}_{i:06d}.png'), cv2.IMREAD_UNCHANGED), dtype=np.uint8)

            if depth.max() == 0:
                print(f'image {imgId} has no depth map. Skip.')
                continue

            valid_mask_pixel = (mask > 0).sum()
            if  valid_mask_pixel < 1e4:
                print(f'image {imgId} has not enough valid mask. Skip.')
                continue

            # hashcode frame_id
            id_str = f"{frame_id:06d}".encode('utf-8')
            hash_object = hashlib.sha256()
            hash_object.update(id_str)

            
            if len(c2ws_gt) > 0:
                min_degrees = 180.
                for prev_c2w in c2ws_gt:
                    prev_c2w_rot = prev_c2w[:3, :3]
                    c2w_rot = c2w_gt[:3, :3]

                    rotation_error_cos = 0.5 * (np.trace(np.dot(c2w_rot.T, prev_c2w_rot)) - 1.0)
                    rotation_error_cos = min(1.0, max(-1.0, rotation_error_cos))  # Ensure value is in valid range for arccos
                    rotation_error_rad = np.arccos(rotation_error_cos)
                    rotation_error_deg = np.degrees(rotation_error_rad)
                    if rotation_error_deg < min_degrees:
                        min_degrees = rotation_error_deg
                
                if min_degrees < 5.0:
                    continue

            # # viusalize all data
            # fig = plt.figure(figsize=(19, 10), dpi=200)
            # plt.subplot(1,3,1); plt.imshow(color)
            # plt.subplot(1,3,2); plt.imshow(depth); plt.colorbar()
            # plt.subplot(1,3,3); plt.imshow(mask)
            # plt.show()

            rgbs.append(color)
            depths.append(depth)
            masks.append(mask)
            c2ws_gt.append(c2w_gt)
            c2ws.append(c2w)
            frameIds.append(hash_object.hexdigest())

            frame_id += 1
    

print(f"Total number of frames: {frame_id}")

glcam_in_obs = np.asarray(c2ws)
glcam_in_obs_gt = np.asarray(c2ws_gt)       
rgbs = np.asarray(rgbs)
depths = np.asarray(depths)
masks = np.asarray(masks)

# create initial pointcloud from RGBD
# for (color, depth, mask) in zip(rgbs, depths, masks):
#    fig = plt.figure(figsize=(20, 10), dpi=200)
#    plt.subplot(1,3,1); plt.imshow(color)
#    plt.subplot(1,3,2); plt.imshow(depth); plt.colorbar()
#    plt.subplot(1,3,3); plt.imshow(mask)
#    plt.show()


from importlib.machinery import SourceFileLoader
import argparse

# load gs_config
parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str, help="Path to experiment file")

args = parser.parse_args()

# load config
experiment = SourceFileLoader(
    os.path.basename(args.experiment), args.experiment
).load_module()

cfg = experiment.config

rgbs, depths, masks, poses = preprocess_data(rgbs, depths, masks, glcam_in_obs)

total_num_frames = len(rgbs)
print(f"Total number of frames: {total_num_frames}")
first_init_num_frames = 20

frame_id = first_init_num_frames

first_rgbs = rgbs[:first_init_num_frames, ...]
first_depths = depths[:first_init_num_frames, ...]
first_masks = masks[:first_init_num_frames, ...]
first_poses = poses[:first_init_num_frames, ...]

# get center of poses

# fuse pointcloud from rgbd
pcd = fuse_pointcloud(first_rgbs, first_depths, first_masks, first_poses, K)
world_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# create initial pointcloud from RGBD
# TODO 1. add pointcloud for gs_runner 2. understanding transformation and rescale factor works for gs_runner
first_c2w = first_poses[0].copy()
first_c2w[:3, 1:3] *= -1             # from opengl to opencv
obj_init_pose = np.linalg.inv(first_c2w)

pcd.transform(obj_init_pose)
pcd = pcd.voxel_down_sample(voxel_size=0.001)

# transit open3d pointcloud to numpy array
pcl = np.asarray(pcd.points)
pcl = np.concatenate([pcl, np.asarray(pcd.colors)], axis=1)

# create initial pointcloud in unit scale
bbox = pcd.get_oriented_bounding_box()
center = bbox.center
extent = bbox.extent.max() * 1.5 # leave some margin

N_points = 10000
points = np.random.rand(N_points, 3) * extent - extent / 2
points += center

colors = np.random.rand(N_points, 3)

pcl_random = np.concatenate([points, colors], axis=1)


# compare gt pointcloud
pcd_gt = fuse_pointcloud(rgbs[::5,...], depths[::5, ...], masks[::5, ...], glcam_in_obs_gt[::5, ...], K)
pcd_gt.transform(obj_init_pose)

pcd_gt = pcd_gt.voxel_down_sample(voxel_size=0.001)

pcl_gt = np.asarray(pcd_gt.points)
pcl_gt = np.concatenate([pcl_gt, np.asarray(pcd_gt.colors)], axis=1)

# visualize pointcloud
#o3d.visualization.draw([pcd, pcd_gt, world_coord])


import wandb
wandb_run = wandb.init(
    # Set the project where this run will be logged
    project="BundleGS",
    # Track hyperparameters and run metadata
    settings=wandb.Settings(start_method="fork"),
    mode='disabled'
)

first_poses = glcam_in_obs_gt[:first_init_num_frames, ...]

gsRunner = GSRunner(
    cfg,
    rgbs=first_rgbs,
    depths=first_depths,
    masks=first_masks,
    K=K,
    poses=first_poses,
    total_num_frames=total_num_frames,
    pointcloud=pcl_gt,
    #pointcloud_gt=pcl_gt,
    poses_gt=glcam_in_obs_gt.copy(),
    wandb_run=wandb_run,
    #run_gui=True,
)

# save trained gaussian model
gsRunner.train()
1/0
opt_pcd = o3d.geometry.PointCloud()
pcl = gsRunner.get_xyz_rgb_params()

opt_pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
opt_pcd.colors = o3d.utility.Vector3dVector(pcl[:, 3:6])

o3d.visualization.draw([opt_pcd, pcd_gt, world_coord])

for i in range(first_init_num_frames, total_num_frames):
    rgb = rgbs[i]
    rgb = rgb.reshape(1, *rgb.shape)
    depth = depths[i]
    depth = depth.reshape(1, *depth.shape)
    mask = masks[i]
    mask = mask.reshape(1, *mask.shape)
    pose = poses[: i + 1, ...]
    gsRunner.add_new_frames(rgb, depth, mask, pose)
    gsRunner.train()

opt_pcd = o3d.geometry.PointCloud()
pcl = gsRunner.get_xyz_rgb_params()

opt_pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
opt_pcd.colors = o3d.utility.Vector3dVector(pcl[:, 3:6])


o3d.visualization.draw([opt_pcd, pcd_gt, world_coord])

