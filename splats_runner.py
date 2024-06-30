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
import torch.nn.functional as F
from tqdm import tqdm
import open3d as o3d
import kornia as ki
try:
  import kaolin
except Exception as e:
  print(f"Import kaolin failed, {e}")
from torch.utils.data import Dataset
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_ply, params2cpu
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.splats_helper import (setup_camera, 
                                 transform_to_frame, 
                                 transformed_params2depthplussilhouette, 
                                 transformed_params2rendervar,
                                 densify)
from utils.sh_helpers import RGB2SH
from Utils import *
from utils.loss_helper import l1_loss, l2_loss, calc_ssim

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

class DataReader(Dataset):
    def __init__(self, data_dict):
        data_lens = [len(v) for v in data_dict.values()]
        assert len(set(data_lens)) == 1, "All values in data_dict should have the same length"
        self.lens = data_lens[0]
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return self.lens 

    def __getitem__(self, idx):
        return {
            k: self.data_dict[k][idx] for k in self.keys
        } 

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
        self._N_splats_iters = 0   # number of splats iterations
        self._N_epoch = 0
        

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
        
        self.down_scale_ratio = down_scale_ratio
        self.H, self.W = self.rgbs[0].shape[:2]  

        self.octree_m = None
        
        # build octree
        if self.cfg['use_octree']:
            self.build_octree()

        # create gaussian splats and sdf network
        self.create_splats()

        # create optimizer
        self.create_optimizer()

        # convert rgbs, depths, masks, pose to tensors
        self.rgbs = npimage2tensor(self.rgbs).to(self.device)
        self.depths = npimage2tensor(self.depths).to(self.device)
        self.masks = npimage2tensor(self.masks).to(self.device)
        self.poses = torch.tensor(self.poses).to(self.device).float()
        self.w2cs = torch.inverse(self.poses).float()           # world to camera
        self.K = torch.tensor(self.K).to(self.device).float()

        data = {
            'rgbs': self.rgbs,
            'depths': self.depths,
            'masks': self.masks,
            'w2cs': self.w2cs,
        }

        self.data_reader = DataReader(data)
        logging.info("DataReader created")
        logging.info("SplatsRunner initialized")
        
    def add_new_frames(self, rgbs, depths, masks, poses):
        """
        Add new frames to the runner
        """
        poses[:, :3, 1:3] *= -1 # from opengl to cv

        if self.down_scale_ratio!=1:
            rgbs = rgbs[:, ::self.down_scale_ratio, ::self.down_scale_ratio]
            depths = depths[:, ::self.down_scale_ratio, ::self.down_scale_ratio]
            masks = masks[:, ::self.down_scale_ratio, ::self.down_scale_ratio]

        rgbs = npimage2tensor(rgbs).to(self.device)
        depths = npimage2tensor(depths).to(self.device)
        masks = npimage2tensor(masks).to(self.device)

        # update full poses  TODO probably the estimation fromm BA worse than 6dsplats
        self.poses = torch.tensor(poses).to(self.device).float()
        self.w2cs = torch.inverse(self.poses).float()           # world to camera

        # update rgbs, depths, masks
        self.rgbs = torch.cat([self.rgbs, rgbs], dim=0)
        self.depths = torch.cat([self.depths, depths], dim=0)
        self.masks = torch.cat([self.masks, masks], dim=0)

        logging.info("New frames added") 
        data = {
            'rgbs': self.rgbs,
            'depths': self.depths,
            'masks': self.masks,
            'w2cs': self.w2cs,
        }

        self.data_reader = DataReader(data)
        self._N_epoch = 0
        

    def create_optimizer(self):
        lrs = self.cfg['optimizer']['lrs']

        params = []
        for k in self.model.keys():
            if k == 'params':
                params += [{'params': v, 'name': k, 'lr': lrs[k]} for k, v in self.model[k].items()]
            elif k == 'sdf':
                params += [{'params': self.model[k].parameters(), 'name': k, 'lr': lrs[k]}]
            elif k == 'variables':
                continue
            else:
                params += [{'params': self.model[k], 'name': k, 'lr': lrs[k]}]

        self.optimizer = torch.optim.Adam(params, betas=(0.9, 0.999),weight_decay=0,eps=1e-15)
        
        # save initial param group
        self.init_param_group = self.optimizer.param_groups.copy()
    
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

        params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items() if v is not None}
        model["params"] = params
        model["variables"] = variables

        # Initialize cam pose array 
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, self.total_num_frames))
        cam_trans = np.zeros((1, 3, self.total_num_frames))

        model['cam_unnorm_rots'] = torch.nn.Parameter(torch.tensor(cam_rots).cuda().float().contiguous().requires_grad_(True))
        model['cam_trans'] = torch.nn.Parameter(torch.tensor(cam_trans).cuda().float().contiguous().requires_grad_(True)) 

        model['sdf'] = SDFNetwork().cuda().float()
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


    def train(self):
        """
        Train the model
        """
        epochs = self.cfg['train']['num_epochs']
        batch_size = self.cfg['train']['batch_size']

        while self._N_epoch < epochs:
            if self._N_epoch % (epochs // 10) == 0:
                logging.info(f"Epoch {self._N_epoch}/{epochs}")
            # batch data
            indices = np.random.permutation(len(self.data_reader))[:batch_size]
            # TODO first frame must be in the batch
            batch_data = self.data_reader[indices]

            self.train_batch(batch_data, indices)
            self._N_epoch += 1


    def train_once_worker(self, curr_data, time_idx, dssim_weight=0.2):
        im_gt = curr_data['im_gt']
        depth_gt = curr_data['depth_gt']
        mask_gt = curr_data['mask_gt']
        w2c = curr_data['w2c']
        
        loss_weights = self.cfg['train']['loss_weights']

        loss = torch.tensor(0.0).to(self.device)
        # transform the gaussian splats to the current frame
        transformed_gaussians, rel_w2c = transform_to_frame(
                    self.model, time_idx, gaussians_grad=True, camera_grad=True
                )
        
        # Initialize Render Variables
        rendervar = transformed_params2rendervar(self.model['params'], transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(
            self.model['params'], w2c, transformed_gaussians
        )
                    # RGB Rendering
        rendervar["means2D"].retain_grad()

        cam = setup_camera(self.W, self.H, self.K, w2c)
        (
            im,
            radius,
            _,
        ) = Renderer(
            raster_settings=cam
        )(**rendervar)

        self.model['variables']["means2D"] = rendervar[
            "means2D"
        ]  # Gradient only accum from colour render for densification

        # Depth & Silhouette Rendering
        (
            depth_sil,
            _,
            _,
        ) = Renderer(
            raster_settings=cam
        )(**depth_sil_rendervar)

        depth = depth_sil[0, :, :].unsqueeze(0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = silhouette > self.cfg["train"]["sil_thres"]

        # overlap mask
        nan_mask = (~torch.isnan(depth))
        valid_mask = mask_gt.bool() 
        valid_mask = valid_mask & presence_sil_mask & nan_mask

        gt_edge = None
        edge = None

        if self.cfg['train']['use_edge_loss']: 
            # canny edge detection
            gt_gray = ki.color.rgb_to_grayscale(im_gt.unsqueeze(0))
            gt_edge = ki.filters.sobel(gt_gray).squeeze(0)

            gray = ki.color.rgb_to_grayscale(im.unsqueeze(0))

            # sobel edges
            edge = ki.filters.sobel(gray).squeeze(0)
            edge_loss = torch.abs(gt_edge - edge).mean()

            loss += loss_weights["edge"] * edge_loss
        
        if self.cfg['train']['use_depth_loss']:
            depth_loss = l2_loss(depth, depth_gt, valid_mask, reduction="mean")
            loss += loss_weights["depth"] * depth_loss

        if self.cfg['train']['use_silhouette_loss']:
            silhouette_loss = torch.abs(silhouette.float() - mask_gt).mean()
            loss += loss_weights["silhouette"] * silhouette_loss

        if self.cfg['train']['use_im_loss']:
            # color loss
            color_mask = torch.tile(valid_mask, (3, 1, 1)).detach()
            
            rgb_loss = torch.abs(im_gt - im)[color_mask].mean()
            im_loss = (1-dssim_weight) * rgb_loss + dssim_weight * (1.0 - calc_ssim(im, im_gt))
            loss += loss_weights["im"] * im_loss

        seen = radius > 0
        variables = self.model["variables"]
        variables["max_2D_radius"][seen] = torch.max(
            radius[seen], variables["max_2D_radius"][seen]
        )
        variables["seen"] = seen

        if self.DEBUG_MODE >=2 and self._N_epoch % 20 == 0:
# evaluaaussian depth rendering
            fig, ax = plt.subplots(4, 4, figsize=(12, 12))
            weighted_render_im = im * color_mask
            weighted_im = im_gt * color_mask
            weighted_render_depth = depth * valid_mask
            weighted_depth = depth_gt * valid_mask
            weighted_render_candy = edge 
            weighted_candy = gt_edge
            viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)

            diff_rgb = (
                torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
            )
            diff_depth = (
                torch.abs(weighted_render_depth - weighted_depth)
                .mean(dim=0)
                .detach()
                .cpu()
            )

            diff_candy = (
                torch.abs(weighted_render_candy - weighted_candy)
                .mean(dim=0)
                .detach()
                .cpu()
            )
            diff_sil = (
                torch.abs(presence_sil_mask.float() - mask_gt.float())
                .squeeze(0)
                .detach()
                .cpu()
            )
            ax[0, 0].imshow(viz_img)
            ax[0, 0].set_title("Weighted GT RGB")
            viz_render_img = torch.clip(
                weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1
            )
            ax[1, 0].imshow(viz_render_img)
            ax[1, 0].set_title("Weighted Rendered RGB")
            ax_im = ax[0, 1].imshow(weighted_depth[0].detach().cpu())
            cbar = fig.colorbar(ax_im, ax=ax[0, 1])
            ax[0, 1].set_title("Weighted GT Depth")
            ax_im = ax[1, 1].imshow(
                weighted_render_depth[0].detach().cpu())
            cbar = fig.colorbar(ax_im, ax=ax[1, 1])
            ax[1, 1].set_title("Weighted Rendered Depth")
            ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
            ax[0, 2].set_title(f"Diff RGB, Loss: {im_loss.item()}")
            ax_im = ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
            cbar = fig.colorbar(ax_im, ax=ax[1, 2])
            ax[1, 2].set_title(f"Diff Depth, Loss: {depth_loss.item()}")
            ax_im = ax[0, 3].imshow(silhouette.detach().squeeze().cpu())
            cbar = fig.colorbar(ax_im, ax=ax[0, 3])
            ax[0, 3].set_title("Silhouette Mask")
            ax[1, 3].imshow(valid_mask[0].detach().cpu(), cmap="gray")
            ax[1, 3].set_title("Loss Mask")
            ax[2, 0].imshow(mask_gt.detach().squeeze().cpu())
            ax[2, 0].set_title("gt Mask")
            ax[2, 2].imshow(
                (im.permute(1, 2, 0).detach().squeeze().cpu().numpy() * 255).astype(
                    np.uint8
                )
            )
            ax[2, 2].set_title("Rendered RGB")
            ax[2, 3].imshow(im_gt.permute(1, 2, 0).detach().squeeze().cpu())
            ax[2, 3].set_title("GT RGB")
            ax[3, 0].imshow(
                torch.clamp(
                    weighted_candy.permute(1, 2, 0).detach().squeeze().cpu(), 0.0, 1.0
                ),
                cmap="jet",
            )
            ax[3, 0].set_title("Weighted GT canny")
            # A3d colorbar
            ax[3, 1].imshow(
                torch.clamp(
                    weighted_render_candy.permute(1, 2, 0).detach().squeeze().cpu(),
                    0.0,
                    1.0,
                ),
                cmap="jet",
            )
            ax[3, 1].set_title("Weighted rendered canny")
            ax[3, 2].imshow(
                torch.clamp(diff_candy.detach().squeeze().cpu(), 0.0, 1.0), cmap="jet"
            )
            ax[3, 2].set_title(f"Diff Edge: {edge_loss.item()}")
            ax[3, 3].imshow(diff_sil, cmap="jet")
            ax[3, 3].set_title(f"Silhouette Loss: {silhouette_loss.item()}")
            # Turn off axis
            for i in range(2):
                for j in range(4):
                    ax[i, j].axis("off")
            # Set Title
            suptitle = f"frame_id: {time_idx}"
            fig.suptitle(suptitle, fontsize=16)
            # Figure Tight Layout
            fig.tight_layout()
            plot_dir = os.path.join(self.cfg['save_dir'], "train_plots")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plt.savefig(
                os.path.join(plot_dir, f"epoch_{self._N_epoch}_frame_{time_idx:06d}.png"), bbox_inches="tight", dpi=180
            )
            plt.close()
        return loss

    def train_batch(self, batch_data, indices):
        """
        Train the model on a batch
        """
        loss_weights = self.cfg['train']['loss_weights']

        batch_loss = torch.tensor(0.0).to(self.device)
        ims, depths, masks, w2cs = batch_data['rgbs'], batch_data['depths'], batch_data['masks'], batch_data['w2cs']
        for im_gt, depth_gt, mask_gt, w2c, time_idx in zip(ims, depths, masks, w2cs, indices):
            curr_data = {
                'im_gt': im_gt,
                'depth_gt': depth_gt,
                'mask_gt': mask_gt,
                'w2c': w2c,
            }
            loss = self.train_once_worker(curr_data, time_idx)
            if time_idx == 0:
                # first frame
                loss *= 10 

            batch_loss += loss
            
        batch_loss/=len(indices)

        # optimize
        batch_loss.backward()

        if self.cfg['gaussians']['distribution'] == "aniostropic_2d":
            self.params["log_scales"].grad[...,2] = torch.tensor(0.0).to(self.device) 

        self.optimizer.step()
        self._N_splats_iters += 1

        with torch.no_grad():
            # gaussian densification
            self.params, self.variables = densify(
                    self.model['params'],
                    self.model['variables'],
                    self.optimizer,
                    self._N_splats_iters,
                    self.cfg['train']["densification"],
                    debug_level=self.DEBUG_MODE,
                )

        self.optimizer.zero_grad(set_to_none=True)

        # save intermediate splats
        if self.DEBUG_MODE >= 2 and self._N_epoch % 100 == 0:
            params = params2cpu(self.model['params'])
            means = params['means3D']
            scales = params['log_scales']
            rotations = params['unnorm_rotations']
            rgbs = params['rgb_colors']
            opacities = params['logit_opacities']
            ply_path = os.path.join(self.cfg['save_dir'], f"splats_iter_{self._N_splats_iters}.ply")
            save_ply(ply_path, means, scales, rotations, rgbs, opacities)

            
