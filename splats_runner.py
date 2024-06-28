import argparse
import os
import io
import logging
import shutil
import sys
import time
import datetime
import random
from importlib.machinery import SourceFileLoader
from collections import deque
#from Utils import ForkedPdb

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import open3d as o3d
import kornia as ki
try:
  import kaolin
except Exception as e:
  print(f"Import kaolin failed, {e}")
from torch.utils.data import DataLoader, Dataset
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_ply, params2cpu
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
    transform_to_frame,
    quat_mult,
    matrix_to_quaternion,
)
from utils.sh_helpers import RGB2SH
from Utils import *
from utils.loss_helper import l1_loss, l2_loss, log_mse_loss, psnr
from datasets.bundlegs_datasets import relative_transformation, datautils
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

# TODO 1. instead of updating camera pose and mapping iteratively, update camera pose and mapping at the same time
# TODO 2. add keyframe selection
# TODO 3. evaluate the error of camera pose optimization on gt_pose

class SDFNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=1):
        super(SDFNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # init weights
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.constant_(self.fc3.bias, 0.1)     # encourge last layer to predict positive values

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SplatsRunner:
    def __init__(
        self, cfg, rgbs, depths, masks, K, poses, total_num_frames, pcd_normalized=None, dtype=torch.float):
        """
        @param rgbs: np.ndarray, shape (N, H, W, 3), dtype float32 
        @param depths: np.ndarray, shape (N, H, W, 1), dtype float32
        @param masks: np.ndarray, shape (N, H, W, 1), dtype uint8
        @param K: np.ndarray, shape (3, 3), dtype float32
        @param poses: np.ndarray, shape (N, 4, 4), dtype float64
        """
        # preprocess data -> to tensor
        self.device = torch.device(cfg["device"])
        self.dtype = dtype
        self.total_num_frames = total_num_frames
        self.cfg = cfg
        self.K = K
        self.rgbs=rgbs
        self.depths=depths
        self.masks=masks.astype("bool")
        self.poses=poses

        self.poses[:, :3, 1:3] *= -1 # from opengl to cv
        
        self.build_octree_pts = None
        self.build_octree_cls = None

        if pcd_normalized is not None:
            self.build_octree_pts = np.asarray(pcd_normalized.points).copy()
            self.build_octree_cls = np.asarray(pcd_normalized.colors).copy()

        self.DEBUG_MODE = cfg["debug_level"]

        # downscale images
        down_scale_ratio = cfg["down_scale_ratio"]
        if down_scale_ratio!=1:
            H,W = self.rgbs[0].shape[:2]
            down_scale_ratio = int(down_scale_ratio)
            self.rgbs = self.rgbs[:, ::down_scale_ratio, ::down_scale_ratio]
            self.depths = self.depths[:, ::down_scale_ratio, ::down_scale_ratio]
            self.masks = self.masks[:, ::down_scale_ratio, ::down_scale_ratio]
            self.H, self.W = self.rgbs[0].shape[:2]

            self.K[0] *= float(self.W) / W
            self.K[1] *= float(self.H) / H
        
        self.H, self.W = rgbs[0].shape[:2]  

        self.octree_m = None
        if self.cfg['use_octree']:
            self.build_octree()

        # create spc from build_octree_pts
        self.create_splats()
        

    def create_splats(self, num_splats=10000):
        # TODO create splats from colored octree, kaolin
        model = {}
        if self.build_octree_pts is None:
            # create uniform samples points in [-1, -1, -1] to [1, 1, 1]
            logging.info("Creating splats from uniform points, no pointcloud provided")
            init_pts = np.random.uniform(-1, 1, (num_splats, 3)) * 0.7
            init_pts = torch.tensor(init_pts).float().to(self.device)
            init_cols = torch.ones((num_splats, 3)).float().to(self.device) * 0.5
        else:
            logging.info("Creating splats from provided pointcloud")
            # upsample build_octree_pts to num_splats
            num_pts = self.build_octree_pts.shape[0]
            num_pts_to_add = num_splats - num_pts 
            init_pts = self.build_octree_pts.copy()
            init_cols = self.build_octree_cls.copy()

            chunk = 1000
            while num_pts_to_add > 0:
                if num_pts_to_add < chunk:
                    chunk = num_pts_to_add
                rand_idx = np.random.randint(num_pts, size=chunk)
                new_pts = self.build_octree_pts[rand_idx] + 2 * (np.random.rand(chunk, 3) - 1) * self.cfg['sc_factor'] * self.cfg['gaussians']['init_pts_noise']
                init_pts = np.concatenate([init_pts, new_pts])
                new_cls = self.build_octree_cls[rand_idx]
                init_cols = np.concatenate([init_cols, new_cls])
                num_pts_to_add -= chunk

            init_pts = torch.tensor(init_pts).float().to(self.device)
            init_cols = torch.tensor(init_cols).float().to(self.device)

        means3D = init_pts
        rgb_colors = init_cols

        if self.cfg['gaussians']["rgb2sh"]: 
            fused_color = RGB2SH(rgb_colors)
            features = torch.zeros((fused_color.shape[0], 3, (self.cfg['gaussians']['max_sh_degree'] + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            features_dc = features[:,:,0:1].transpose(1, 2)
            features_rest = features[:,:,1:].transpose(1, 2)

        else:
            features_dc = None
            features_rest = None

        num_pts = means3D.shape[0]
        rots = torch.tile(torch.tensor([1, 0, 0, 0]), (num_pts, 1)).to(self.device).type(self.dtype)
        opacities = inverse_sigmoid(0.1 * torch.ones((num_pts, 1)).to(self.device).type(self.dtype))
        dist2 = torch.clamp_min(distCUDA2(means3D), 0.0000001)

        if self.cfg["gaussians"]["distribution"] == "isotropic":
            log_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1)
        elif self.cfg["gaussians"]["distribution"] == "anisotropic":
            log_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        elif self.cfg["gaussians"]["distribution"] == "anisotropic_2d":
            log_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1)
            log_scales = torch.cat((log_scales, log_scales, -1e5 * torch.ones_like(log_scales)), dim=1)
        else:
            raise ValueError(f"Unknown gaussian_distribution {self.cfg['gaussian_distribution']}")

        params = {
            "means3D": means3D,
            "rgb_colors": rgb_colors,
            "unnorm_rotations": rots,
            "logit_opacities": opacities,
            "log_scales": log_scales,
            "features_dc": features_dc,
            "features_rest": features_rest
        }

        variables = {
            "max_2D_radius": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "means2D_gradient_accum": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "denom": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "scene_radius": torch.tensor(2.0).cuda().float(),       #TODO understand scene_radius
        } 
        logging.info("Splats created")

        model["params"] = params
        model["variables"] = variables

        # Initialize cam pose array 
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, self.total_num_frames))
        model['cam_unnorm_rots'] = cam_rots
        model['cam_trans'] = np.zeros((1, 3, self.total_num_frames))

        for k, v in model.items():
            if k == "params" or k == "variables":
                model[k] = {kk: torch.tensor(vv).cuda().float().contiguous().requires_grad_(True) for kk, vv in v.items() if vv is not None}
             
            else:
                model[k] = torch.tensor(v).cuda().float().contiguous().requires_grad_(True)

        
        model['sdf'] = SDFNetwork().cuda().float()]
        self.model = model
        

    def build_octree(self):
        # build octree
        logging.info("Building Octree")
        if self.build_octree_pts is None:
            logging.info("No points to build octree, octree is not built")
            return

        if self.DEBUG_MODE >= 2:
            logging.info("Saving Octree Build Points")
            dir = f"{self.cfg['save_dir']}/build_octree_pts.ply"
            pcd = toOpen3dCloud(self.build_octree_pts)
            o3d.io.write_point_cloud(dir, pcd)


        pts = torch.tensor(self.build_octree_pts).cuda().float()                   # Must be within [-1,1]
        octree_smallest_voxel_size = self.cfg['octree_smallest_voxel_size']*self.cfg['sc_factor']
        finest_n_voxels = 2.0/octree_smallest_voxel_size
        max_level = int(np.ceil(np.log2(finest_n_voxels)))
        octree_smallest_voxel_size = 2.0/(2**max_level)
        logging.info(f"Octree voxel smallest_voxel_size:{octree_smallest_voxel_size} max_level:{max_level}")

        #################### Dilate
        dilate_radius = int(np.ceil(self.cfg['octree_dilate_size']/self.cfg['octree_smallest_voxel_size']))
        dilate_radius = max(1, dilate_radius)
        logging.info(f"Octree dilation octree voxel dilate_radius:{dilate_radius}")
        shifts = []
        for dx in [-1,0,1]:
          for dy in [-1,0,1]:
            for dz in [-1,0,1]:
              shifts.append([dx,dy,dz])
        shifts = torch.tensor(shifts).cuda().long()    # (27,3)
        coords = torch.floor((pts+1)/octree_smallest_voxel_size).long()  #(N,3)
        dilated_coords = coords.detach().clone()
        for iter in range(dilate_radius):
          dilated_coords = (dilated_coords[None].expand(shifts.shape[0],-1,-1) + shifts[:,None]).reshape(-1,3)
          dilated_coords = torch.unique(dilated_coords,dim=0)
        pts = (dilated_coords+0.5) * octree_smallest_voxel_size - 1
        pts = torch.clip(pts,-1,1) 

        if self.DEBUG_MODE >= 2:
            logging.info("Saving Octree Dilated Points")
            pcd = toOpen3dCloud(pts.cpu().numpy())
            dir = f"{self.cfg['save_dir']}/dilated_octree_pts.ply"
            o3d.io.write_point_cloud(dir, pcd)

        #################### create octree Manager
        assert pts.min()>=-1 and pts.max()<=1
        self.octree_m = OctreeManager(pts, max_level)

        logging.info("Octree Built")

        if self.DEBUG_MODE >= 2: 
            dir = f"{self.cfg['save_dir']}/octree_boxes_max_level.ply"
            self.octree_m.draw_boxes(level=max_level,outfile=dir)

        vox_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
        level = int(np.floor(np.log2(2.0/vox_size)))
        
        if self.DEBUG_MODE >= 2:
            dir = f"{self.cfg['save_dir']}/octree_boxes_ray_tracing_level.ply"
            self.octree_m.draw_boxes(level=level,outfile=dir)
