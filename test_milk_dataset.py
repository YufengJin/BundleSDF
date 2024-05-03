import os
import sys
import numpy as np
import yaml
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from gs_runner import GSRunner

# from gs_runner import GaussianSplatRunner

# load all .npy
folder_path = "/home/yjin/repos/BundleSDF/final_datas"

# List all .npy files in the folder
npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

# Loop through each .npy file
for file_name in npy_files:
    # Load the NumPy array from the .npy file
    arr = np.load(os.path.join(folder_path, file_name))

    # Assign the filename as the variable name for the array
    var_name = os.path.splitext(file_name)[0]  # Remove the .npy extension
    globals()[var_name] = arr

# Now you can access the loaded arrays using their filenames as variable names
# For example, if you have a file named "example.npy", you can access its array using "example" variable

# create a transform from gl cam to gs cams
# cfg = yaml.safe_load(open('gs_runner_config.yml', 'r'))

#import wandb
#
#wandb.login()
#run = wandb.init(
#    # Set the project where this run will be logged
#    project="Gaussian Splatting Analysis",
#    name="naive-GS-bundlesdf-datasets-new-renderer",
#    # Track hyperparameters and run metadata
#    settings=wandb.Settings(start_method="fork"),
#    mode="disabled",
#)


frameIds = [f"{i:03d}" for i in range(rgbs.shape[0])]


def rgbd_to_pointcloud(rgb_image, depth_image, K):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    h, w = depth_image.shape
    y, x = np.indices((h, w))
    z = depth_image
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    # Stack the coordinates to create the point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Step 4: Associate colors with the point cloud
    colors = rgb_image.reshape(-1, 3)
    # Step 5: Create Open3D point cloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(
        colors / 255.0
    )  # Normalize colors to range [0, 1]

    return pcd


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


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


def create_pcd_from_data(rgbs, depths, masks, glcam_in_obs):
    pcdAll = o3d.geometry.PointCloud()
    for color, depth, mask, pose in zip(rgbs, depths, masks, glcam_in_obs):
        # mask erosion
        mask_uint = mask.astype(np.uint8)
        kernel = np.ones((30, 30), np.uint8)  # You can adjust the kernel size as needed

        # Perform erosion
        eroded_mask = cv2.erode(mask_uint, kernel, iterations=1)

        mask = eroded_mask.astype(np.bool_)

        depth_tmp = depth.copy()
        depth_tmp[depth_tmp < 0.1] = np.nan

        mask = np.logical_and(np.logical_not(np.isnan(depth_tmp)), mask)
        # plt.subplot(1, 3, 1); plt.imshow(mask);
        # plt.subplot(1, 3, 2); plt.imshow(mask_uint.astype(np.bool_));
        # plt.subplot(1, 3, 3); plt.imshow(mask ^ mask_uint.astype(np.bool_));
        # plt.show()

        # mask
        color[np.logical_not(mask)] = [0.0, 0.0, 0.0]
        depth[np.logical_not(mask)] = 0.0
        pcd = rgbd_to_pointcloud(color, depth, K)

        pose_o3d = pose.copy()
        pose_o3d[:3, 1:3] *= -1
        pcd.transform(pose_o3d)
        pcdAll += pcd

        # postprocess pointlcoud remove outlier
        # pcdAll = pcdAll.voxel_down_sample(voxel_size=0.01)
        # cl, ind = pcdAll.remove_statistical_outlier(nb_neighbors=20,
        #                                                std_ratio=1.0)

        # pcdAll = pcdAll.select_by_index(ind)

    pcdAll = pcdAll.voxel_down_sample(voxel_size=0.01)
    # cl, ind = pcdAll.remove_statistical_outlier(nb_neighbors=100,
    #                                                std_ratio=1.0)
    cl, ind = pcdAll.remove_radius_outlier(nb_points=20, radius=0.02)

    display_inlier_outlier(pcdAll, ind)

    pcdAll = pcdAll.select_by_index(ind)
    # o3d.visualization.draw_geometries([pcdAll])
    return pcdAll

import threading
import time

gui_lock = threading.Lock()

gui_dict = {"join": False}

def run_gui():

    # initialize pointcloud
    num_points = 1000
    points = np.random.rand(num_points, 3)
    colors = np.random.rand(num_points, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # configure the open3d visualizer size
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)

    vis.add_geometry(pcd)

    while 1:
        with gui_lock:
            join = gui_dict['join']

            if 'pointcloud' in gui_dict:
                new_pcd = gui_dict["pointcloud"]
                del gui_dict["pointcloud"]
            else:
                new_pcd = None
        if join:
            break

        if new_pcd is not None:
            pcd.points = o3d.utility.Vector3dVector(new_pcd[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(new_pcd[:, 3:6])

        time.sleep(0.05)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    visualizer.destroy_window()


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
first_init_num_frames = 5

frame_id = first_init_num_frames

first_rgbs = rgbs[:first_init_num_frames, ...]
first_depths = depths[:first_init_num_frames, ...]
first_masks = masks[:first_init_num_frames, ...]
first_poses = poses[:first_init_num_frames, ...]


gui_runner = threading.Thread(target=run_gui,)
gui_runner.start()

gsRunner = GSRunner(
    cfg,
    rgbs=first_rgbs,
    depths=first_depths,
    masks=first_masks,
    K=K,
    poses=first_poses,
    total_num_frames=total_num_frames,
)
gsRunner.train()

with gui_lock:
    gui_dict["pointcloud"] = gsRunner.get_xyz_rgb_params()


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
    with gui_lock:
        gui_dict["pointcloud"] = gsRunner.get_xyz_rgb_params()

with gui_lock:
    gui_dict['join'] = True

