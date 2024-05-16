import os, sys
_BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import io
import time
import shutil
import time
import yaml
import tqdm
import cv2
import argparse
import open3d as o3d
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
from collections import defaultdict

from datasets.bundlegs_datasets import (load_dataset_config, HO3D_v3Dataset, BOPDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.graphics_utils import rgbd_to_pointcloud
from gs_runner import GSRunner

import wandb
wandb_run = wandb.init(
    # Set the project where this run will be logged
    project="BundleGS",
    # Track hyperparameters and run metadata
    settings=wandb.Settings(start_method="fork"),
    mode='disabled'
)

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

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["ho3d_v3"]:
        return HO3D_v3Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["bop"]:
        return BOPDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def run_once(config: dict):
    cfg_bundletrack_cfg = os.path.join(_BASE_DIR, config["bundletrack_cfg"])
    cfg_bundletrack = yaml.load(open(cfg_bundletrack_cfg, 'r'), Loader=yaml.Loader)

    out_folder = os.path.join(
        config["workdir"], config["run_name"]
    )

    # reconfigure for milk bottle
    cfg_bundletrack['SPDLOG'] = int(config["debug_level"])
    cfg_bundletrack['depth_processing']["zfar"] = 1
    cfg_bundletrack['depth_processing']["percentile"] = 95
    cfg_bundletrack['erode_mask'] = 3
    cfg_bundletrack['debug_dir'] = out_folder+'/'
    cfg_bundletrack['bundle']['max_BA_frames'] = 10
    cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
    cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
    cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
    cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
    cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
    cfg_bundletrack['feature_corres']['map_points'] = True
    cfg_bundletrack['feature_corres']['resize'] = 400
    cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
    cfg_bundletrack['keyframe']['min_rot'] = 5
    cfg_bundletrack['ransac']['inlier_dist'] = 0.01
    cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
    cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
    cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
    cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
    cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
    cfg_bundletrack['p2p']['max_dist'] = 0.02
    cfg_bundletrack['p2p']['max_normal_angle'] = 45
    cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
    yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

    # use Efficient SAM
    use_segmenter = config['use_segmenter']
    if use_segmenter:
        segmenter = Segmenter()

    # dataLoader
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "data_cfg" not in dataset_config:
        data_cfg = {}
        data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        data_cfg = load_dataset_config(dataset_config["data_cfg"])

    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        relative_pose=False,
        target_object_id = dataset_config.get("target_object_id", None),
    )

    # update data frames
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)
        dataset_config['num_frames'] = num_frames

    if config['load_checkpoint']:
        #TODO check save ckpt before updating checkpoint_time_idx and save into exp
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
    else:
        checkpoint_time_idx = 0

    first_num_frames = 10

    curr_data = defaultdict(list) 
    # Load data
    for time_idx in range(checkpoint_time_idx, num_frames):
        print(f"INFO: Loading data: {time_idx}/{num_frames}(total)")

        # Load RGBD frames incrementally instead of all frames
        data = dataset[time_idx]
        mask = None
        # color: np.uint8(0, 255), depth: float (m); mask: uint8(1, 0)    , in bundlesdf color in bgr
        if len(data) == 5:
            color, depth, mask, K, gt_pose = data
        else:
            color, depth, K, gt_pose = data

        est_pose = gt_pose.copy()

        if time_idx > 0:
            # add noise on gt_pose
            est_pose[:3, 3] += (np.random.rand(3) - 0.5) * 2 * 0.02   # translation noise (-0.02, 0.02)
            est_pose[:3, :3] = gt_pose[:3, :3] @ cv2.Rodrigues((np.random.rand(3) - 0.5) * 2 * 5 / np.pi)[0]   # rotation noise (-5, 5) degree
        # convert BGR to RGB
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        if time_idx == 0:
            if mask is None:
                # get initial mask through labeling 
                if use_segmenter:
                    mask = segmenter()
                else:
                    raise("ERROR: No initial mask")

        else:
            if use_segmenter:
                mask = segmenter.update()
            else:
                mask = mask  
        
        if cfg_bundletrack['erode_mask']>0:
            kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)

        if time_idx < first_num_frames:
            curr_data['colors'].append(color)
            curr_data['depths'].append(depth)
            curr_data['masks'].append(mask)
            curr_data['Ks'].append(K)
            curr_data['gt_poses'].append(gt_pose)
            curr_data['est_poses'].append(est_pose)
            continue

        elif time_idx == first_num_frames:
            colors = curr_data['colors']
            depths = curr_data['depths']
            masks = curr_data['masks']
            Ks = curr_data['Ks']
            gt_poses = curr_data['gt_poses']
            est_poses = curr_data['est_poses']

            
            c2ws = [np.linalg.inv(pose) for pose in est_poses]
            c2ws = np.as

            colors, depths, masks, poses = preprocess_data(colors, depths, masks, glc2ws)
            # create init point cloud
            pcd = o3d.geometry.PointCloud()
            pcd_gt = o3d.geometry.PointCloud()
            for (color, depth, mask, pose, gt_pose, K) in zip(colors, depths, masks, est_poses, gt_poses, Ks):
                mask = mask & (depth > 0.1) & (~np.isnan(depth))
                sub_pcd = rgbd_to_pointcloud(color, depth, K, mask, return_o3d=True)

                c2w = np.linalg.inv(pose)
                c2w_gt = np.linalg.inv(gt_pose)

                if False:
                    c2w[:3, 1:3] = -c2w[:3, 1:3]   # opengl
                    c2w_gt[:3, 1:3] = -c2w_gt[:3, 1:3]

                sub_pcd = copy.deepcopy(sub_pcd).transform(c2w)
                pcd += sub_pcd

                sub_pcd_gt = copy.deepcopy(sub_pcd).transform(c2w_gt)
                pcd_gt += sub_pcd_gt

            # remove outliers
            assert (est_poses[0] - gt_poses[0]).mean() < 1e-6

            pcd = pcd.voxel_down_sample(voxel_size=0.001)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100,
                                                    std_ratio=1.0)
            pcd = pcd.select_by_index(ind)
            pcd = pcd.transform(np.linalg.inv(est_poses[0]))


            pcd_gt = pcd_gt.voxel_down_sample(voxel_size=0.001)
            cl, ind = pcd_gt.remove_statistical_outlier(nb_neighbors=100,
                                                    std_ratio=1.0)
            pcd_gt = pcd_gt.transform(np.linalg.inv(gt_poses[0]))
            o3d.visualization.draw_geometries([pcd, pcd_gt])

            # pcd to numpy
            pcl = np.asarray(pcd.points)
            pcl = np.concatenate([pcl, np.asarray(pcd.colors)], axis=1)

            pcl_gt = np.asarray(pcd_gt.points)
            pcl_gt = np.concatenate([pcl_gt, np.asarray(pcd_gt.colors)], axis=1)

            gsRunner = GSRunner(
                config,
                rgbs=colors,
                depths=depths,
                masks=masks,
                K=Ks[0],
                poses=first_poses,
                total_num_frames=total_num_frames,
                pointcloud=pcl_gt,
                #pointcloud_gt=pcl_gt,
                poses_gt=glcam_in_obs_gt.copy(),
                wandb_run=wandb_run,
                #run_gui=True,
            )




        


        
    
    # Object tracking and Gaussian splats generation occur simultaneously
    pass

def run_global_gs():
    # Once object tracking concludes, keyframes are selected for refining Gaussian Splats
    pass

def draw():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    # load config
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # set seed
    seed_everything(seed=experiment.config['seed'])

    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))



    run_once(experiment.config)
    

