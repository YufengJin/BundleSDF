import argparse
import os
import io
import shutil
import sys
import time
import datetime
import random
from importlib.machinery import SourceFileLoader
from collections import deque

from Utils import *
from simple_knn._C import distCUDA2

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import open3d as o3d
import kornia as ki
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid, ConfigParams
from utils.common_utils import (
    seed_everything,
    save_params_ckpt,
    save_params,
    save_ply,
    params2cpu,
)
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
    transform_to_frame,
    l1_loss_v1,
    quat_mult,
    matrix_to_quaternion,
)
from datasets.bundlegs_datasets import relative_transformation, datautils
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from utils.graphics_utils import getWorld2View_torch, getProjectionMatrix, BasicPointCloud
from scene import Camera, GaussianModel

from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

def run_gui_thread(gui_lock, gui_dict, inital_pointcloud):
    # initialize pointcloud
    if inital_pointcloud is not None:
        points = inital_pointcloud[:, :3]
        colors = np.zeros_like(points)
        colors[:, 0] = 1
    else:
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
            join = gui_dict["join"]

            if "pointcloud" in gui_dict:
                new_pcd = gui_dict["pointcloud"]
                del gui_dict["pointcloud"]
            else:
                new_pcd = None
        if join:
            break

        if new_pcd is not None:
            if inital_pointcloud is not None:
                new_pcd = np.concatenate([inital_pointcloud, new_pcd], axis=0)
            pcd.points = o3d.utility.Vector3dVector(new_pcd[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(new_pcd[:, 3:6])

        time.sleep(0.05)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


class GSRunner:
    def __init__(
        self,
        cfg_gs,
        rgbs,
        depths,
        masks,
        K,
        poses,
        total_num_frames,
        dtype=torch.float,
        pointcloud_normalized=None,
        pointcloud_gt=None,
        poses_gt=None,
        wandb_run=None,
        run_gui=False,
    ):
        self.device = torch.device(cfg_gs["primary_device"])
        self.dtype = dtype
        self._total_num_frames = total_num_frames
        self.cfg_gs = cfg_gs
        self.run_gui = run_gui
        self.debug_level = cfg_gs["debug_level"]

        
        # TODO add downscale factor for images
        # from numpy to tensor
        colors, depths, masks = self._preprocess_images_data(rgbs, depths, masks)
        image_height, image_width = colors.shape[2], colors.shape[3]
        
        # setup camera params
        self.bg = torch.tensor([0, 0, 0]).to(self.device).type(self.dtype)
        self.image_height = image_height
        self.image_width = image_width

        fx, fy = K[0][0], K[1][1]
        self.fovx = 2 * math.atan(image_width / (2 * fx))
        self.fovy = 2 * math.atan(image_height / (2 * fy))

        poses[:, :3, 1:3] *= -1          
        poses = torch.tensor(poses).to(self.device).type(self.dtype)
        
        #self.setup_init_camera_params(K)

        # uniform sampling using octree
        normal_pts = np.asarray(pointcloud_normalized.points).copy()
        normal_cls = np.asarray(pointcloud_normalized.colors).copy()
        self.octree_m = None

        # TODO how to trully use octree
        if self.cfg_gs['use_octree']:
            pts, cols = self.build_octree(normal_pts, normal_cls)

        opts = ConfigParams(**self.cfg_gs["gaussians_model"])
        self.pipe = ConfigParams(**self.cfg_gs["pipe"])

        self.gaussians = GaussianModel(opts.sh_degree)
        self.gaussians.training_setup(opts)
        
        pcd = BasicPointCloud(pts.cpu().numpy(), cols.cpu().numpy(), None)
        self.gaussians.create_from_pcd(pcd, self.cfg_gs['sc_factor'])

        #self.create_gaussians(pts, cols)

        # intialize cam error
        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        #self.params["cam_rel_rots"] = torch.zeros((self._total_num_frames, 3)).cuda().float().contiguous().requires_grad_(True)
        #self.params["cam_rel_trans"] = torch.zeros((self._total_num_frames, 3)).cuda().float().contiguous().requires_grad_(True)

        self.gaussians_iter = 0
        self.curr_frame_id = 0

        self.datas = []

        if wandb_run is not None:
            self.wandb_run = wandb_run
            self.use_wandb = True

        if self.run_gui:
            import threading

            self.gui_lock = threading.Lock()
            self.gui_dict = {"join": False}
            gui_runner = threading.Thread(
                target=run_gui_thread,
                args=(self.gui_lock, self.gui_dict, pointcloud_gt),
            )
            gui_runner.start()

        # prepare data from training
        for color, depth, mask, c2w in zip(colors, depths, masks, poses):
            iter_time_idx = self.curr_frame_id
            R = c2w[:3, :3]
            t = c2w[:3, 3]
            viewpoint = Camera(
                iter_time_idx,
                color,
                depth,
                mask,
                self.fovx,
                self.fovy,
                R,
                t,
                self.image_height,
                self.image_width,
                device=self.device,
            )

            self.datas.append(
                {
                    "id": iter_time_idx,
                    'viewpoint': viewpoint,
                    "optimized": False
                }
            )
            self.curr_frame_id += 1

    def add_new_frame(self, rgbs, depths, masks, poses):
        colors, depths, masks = self._preprocess_images_data(rgbs, depths, masks)
        poses[:, :3, 1:3] = -poses[:, :3, 1:3]
        
        # only use the last frame
        poses = torch.tensor(poses).to(self.device).type(self.dtype)[-1, ...].unsqueeze(0)

        for color, depth, mask, c2w in zip(colors, depths, masks, poses):
            iter_time_idx = self.curr_frame_id
            R = c2w[:3, :3]
            t = c2w[:3, 3]

            viewpoint = Camera(
                iter_time_idx,
                color,
                depth,
                mask,
                self.fovx,
                self.fovy,
                R,
                t,
                self.image_height,
                self.image_width,
                device=self.device,
            )

            self.datas.append(
                {
                    "id": iter_time_idx,
                    'viewpoint': viewpoint,
                    "optimized": False
                }
            )
            self.curr_frame_id += 1

    def setup_init_camera_params(self, K, near=0.01, far=100):
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K).to(self.device).type(self.dtype)

        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]

        tanfovx = self.cam_params["image_width"] / (2 * fx)
        tanfovy = self.cam_params["image_height"] / (2 * fy)
        fovx = 2 * torch.atan(tanfovx)
        fovy = 2 * torch.atan(tanfovy)
        project_matrix = getProjectionMatrix(near, far, fovx.item(), fovy.item()).to(self.device).type(self.dtype)

        bg=torch.tensor([0, 0, 0]).to(self.device).type(self.dtype)
        scale_modifier=1.0

        self.cam_params["bg"] = bg
        self.cam_params["scale_modifier"] = scale_modifier
        self.cam_params["proj_matrix"] = project_matrix
        self.cam_params["tanfovx"] = tanfovx
        self.cam_params["tanfovy"] = tanfovy
        self.cam_params['fovx'] = fovx
        self.cam_params['fovy'] = fovy
        self.cam_params['active_sh_degree'] = self.active_sh_degree


    def create_gaussians(self, pts=None, cols=None, num_gaussians=10_000):
        if pts is None and cols is None:
            # random points from [-1, -1, -1, 1, 1, 1]
            pts = np.random.rand(num_gaussians, 3) * 2 - 1
            cols = np.ones((num_gaussians, 3)) * 0.5

        if not isinstance(pts, torch.Tensor):
            pts = torch.tensor(pts).to(self.device).type(self.dtype)
        if not isinstance(cols, torch.Tensor):
            cols = torch.tensor(cols).to(self.device).type(self.dtype)

        means3D = pts
        rgb_colors = cols

        if self.cfg_gs["rgb2sh"]: 
            fused_color = RGB2SH(rgb_colors)
            features = torch.zeros((fused_color.shape[0], 3, (self.cfg_gs['max_sh_degree'] + 1) ** 2)).float().cuda()
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

        if self.cfg_gs["gaussian_distribution"] == "isotropic":
            log_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1)
        elif self.cfg_gs["gaussian_distribution"] == "anisotropic":
            log_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        elif self.cfg_gs["gaussian_distribution"] == "anisotropic_2d":
            log_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1)
            log_scales = torch.cat((log_scales, log_scales, -1e5 * torch.ones_like(log_scales)), dim=1)
        else:
            raise ValueError(f"Unknown gaussian_distribution {self.cfg_gs['gaussian_distribution']}")

        params = {
            "means3D": means3D,
            "rgb_colors": rgb_colors,
            "unnorm_rotations": rots,
            "logit_opacities": opacities,
            "log_scales": log_scales,
            "features_dc": features_dc,
            "features_rest": features_rest
        }

        self.params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) if v is not None else None for k, v in params.items()}

        self.variables = {
            "max_2D_radius": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "means2D_gradient_accum": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "denom": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            #"timestep": torch.zeros(params["means3D"].shape[0]).cuda().float(),
        }        
        
        center_pcl = torch.mean(means3D, dim=0)
        radius = torch.norm(means3D - center_pcl, dim=1).max()
        # Initialize an estimate of scene radius for Gaussian-Splatting Densification, TODO understanding variables scene_radius
        self.variables["scene_radius"] = 2 * radius 
        
    def build_octree(self, normal_pts, normal_cls):
        if self.cfg_gs["save_octree_clouds"]:
            dir = os.path.join(self.cfg_gs['workdir'], self.cfg_gs['run_name'], "build_octree_cloud.ply")
            pcd = toOpen3dCloud(normal_pts, normal_cls)
            o3d.io.write_point_cloud(dir, pcd)
        pts = (
            torch.tensor(normal_pts).cuda().float()
        )  # Must be within [-1,1]
        cls = torch.tensor(normal_cls).cuda().float()

        octree_smallest_voxel_size = (
            self.cfg_gs["octree_smallest_voxel_size"] * self.cfg_gs["sc_factor"]
        )
        finest_n_voxels = 2.0 / octree_smallest_voxel_size
        max_level = int(np.ceil(np.log2(finest_n_voxels)))
        octree_smallest_voxel_size = 2.0 / (2**max_level)

        #################### Dilate
        dilate_radius = int(
            np.ceil(
                self.cfg_gs["octree_dilate_size"] / self.cfg_gs["octree_smallest_voxel_size"]
            )
        )
        dilate_radius = max(1, dilate_radius)
        logging.info(f"Octree voxel dilate_radius:{dilate_radius}")
        shifts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    shifts.append([dx, dy, dz])
        shifts = torch.tensor(shifts).cuda().long()  # (27,3)
        coords = torch.floor((pts + 1) / octree_smallest_voxel_size).long()  # (N,3)

        pts = (coords + 0.5) * octree_smallest_voxel_size - 1
        pts = torch.clip(pts, -1, 1)
        if self.cfg_gs["save_octree_clouds"]:
            dir = os.path.join(self.cfg_gs['workdir'], self.cfg_gs['run_name'], "build_octree_coords.ply")
            pcd = toOpen3dCloud(pts.cpu().numpy(), cls.cpu().numpy())
            o3d.io.write_point_cloud(dir, pcd)

        return pts, cls
        # # TODO duplicate colors

        # dilated_coords = coords.detach().clone()
        # for iter in range(dilate_radius):
        #     dilated_coords = (
        #         dilated_coords[None].expand(shifts.shape[0], -1, -1) + shifts[:, None]
        #     ).reshape(-1, 3)
        #     dilated_coords = torch.unique(dilated_coords, dim=0)
        # pts = (dilated_coords + 0.5) * octree_smallest_voxel_size - 1
        # pts = torch.clip(pts, -1, 1)

        # if self.cfg_gs["save_octree_clouds"]:
        #     pcd = toOpen3dCloud(pts.data.cpu().numpy())
        #     dir = os.path.join(self.cfg_gs['workdir'], self.cfg_gs['run_name'], "build_octree_dilated.ply")
        #     o3d.io.write_point_cloud(dir, pcd)
        # ####################

        # assert pts.min() >= -1 and pts.max() <= 1
        # return pts, cls

    def _preprocess_images_data(self, colors, depths, masks):
        assert len(colors.shape) == len(depths.shape) == len(masks.shape) == 4
        # color = datautils.channels_first(color)
        # mask = mask.astype(np.float32)
        # mask = datautils.channels_first(mask)
        # depth = datautils.channels_first(depth)
        if colors.dtype == "uint8":
            colors = colors.astype(np.float32) / 255.0

        if masks.dtype == "uint8" and masks.max() == 255:
            masks = masks / 255

        # to tensor
        colors = torch.from_numpy(colors).to(self.device).type(self.dtype)
        depths = torch.from_numpy(depths).to(self.device).type(self.dtype)
        masks = torch.from_numpy(masks).to(self.device).type(self.dtype)

        # from B,H,W,C -> B,C,W,H
        colors = colors.permute(0, 3, 1, 2)
        depths = depths.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)

        return (colors, depths, masks)

    def train_v1(self):
        # maping and tracking seperately
        lr_dict = self.cfg_gs['train']["lrs"]

        while len(self.datas) > 0:
            curr_data = self.datas.pop(0)
            time_idx = curr_data["id"]
            if time_idx > 0:
                if self.cfg_gs['add_new_gaussians']:
                    print(f"INFO: Adding new gaussians for frame {time_idx}")
                    self.add_new_gaussians(curr_data)         

                    

        