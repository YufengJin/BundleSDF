import argparse
import os
import io
import shutil
import sys
import time
import datetime
import random
import torch.nn as nn
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
from utils.gs_helpers import visualize_camera_poses
from scene import Camera, GaussianModel
from utils.gaussian_renderer import render
from utils.sh_utils import SH2RGB
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from utils.slam_helpers import get_depth_and_silhouette

def run_gui_thread(gui_lock, gui_dict):
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

            if "image" in gui_dict:
                new_image = gui_dict["image"]
                del gui_dict["image"]
            else:
                new_image = None

        if join:
            break

        if new_pcd is not None:
            pcd.points = o3d.utility.Vector3dVector(new_pcd[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(new_pcd[:, 3:6])

        # if new_image is not None:
        #     cv2.imshow("Diff", new_image)

        time.sleep(0.05)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    cv2.destroyWindow("Diff")
    vis.destroy_window()

def c2w_to_RT(c2w):
    # c2w must in colmap format
    assert isinstance(c2w, np.ndarray) 
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    return R, T

def c2w_to_RT_torch(c2w):
    assert isinstance(c2w, torch.Tensor)
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3,:3].t()  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    return R, T

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
        self.bg = torch.tensor([0, 0, 0]).to(self.device).type(self.dtype)    # black background
        self.image_height = image_height
        self.image_width = image_width

        fx, fy = K[0][0], K[1][1]
        self.fovx = 2 * math.atan(image_width / (2 * fx))
        self.fovy = 2 * math.atan(image_height / (2 * fy))

        poses[:, :3, 1:3] *= -1      # from opengl to colmap
        poses = torch.tensor(poses).to(self.device).type(self.dtype)

        # uniform sampling using octree
        if pointcloud_normalized is not None:
            normal_pts = np.asarray(pointcloud_normalized.points).copy()
            normal_cls = np.asarray(pointcloud_normalized.colors).copy()
            self.octree_m = None

            # TODO how to trully use octree
            if self.cfg_gs['use_octree']:
                pts, cols = self.build_octree(normal_pts, normal_cls)
        
            else:
                pts = torch.tensor(normal_pts).cuda().float()
                cols = torch.tensor(normal_cls).cuda().float()
        
        else:
            # create random pointcloud in unit cube [-1, 1]
            pts = torch.rand((10000, 3)).cuda().float() * 2 - 1
            cols = torch.ones((10000, 3)).cuda().float() * 0.5

        self.opt = ConfigParams(**self.cfg_gs["gaussians_model"])
        self.pipe = ConfigParams(**self.cfg_gs["pipe"])

        self.gaussians = GaussianModel(self.opt.sh_degree)
        pcd = BasicPointCloud(pts.cpu().numpy(), cols.cpu().numpy(), None)
        self.gaussians.create_from_pcd(pcd, self.cfg_gs['sc_factor'])

        self.sdf_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.sdf_net_opt = torch.optim.Adam(self.sdf_net.parameters(), lr=1e-3)

        self.gaussians.training_setup(self.opt)

        # TODO create sdf funtion 

        self.gaussians_iter = 1
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
                args=(self.gui_lock, self.gui_dict),
            )
            gui_runner.start()

        # prepare data from training
        for color, depth, mask, c2w in zip(colors, depths, masks, poses):
            iter_time_idx = self.curr_frame_id
            R, t = c2w_to_RT_torch(c2w)
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
            R, t = c2w_to_RT_torch(c2w)
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
    
    def get_loss(self, viewpoint, color, depth=None, opacity=None, iter=None, tracking=True, dssim_weight=0.2):
        assert viewpoint.color is not None and viewpoint.mask is not None
        if depth is not None:
            if len(depth.shape) == 2:
                depth = depth.unsqueeze(0)
            assert viewpoint.depth is not None
        
        loss = torch.tensor(0.0).to(self.device)
        
        if not tracking:
            loss_weights = self.cfg_gs["train"]["loss_weights"]['mapping']
        else:
            loss_weights = self.cfg_gs["train"]["loss_weights"]['tracking']

        presence_sil_mask = opacity > self.cfg_gs["train"]["sil_thres"]
        nan_mask = (~torch.isnan(depth))

        mask_gt = viewpoint.mask.bool()
        gt_im = viewpoint.color

        mask = mask_gt & presence_sil_mask & nan_mask

        # canny edge detection
        gt_gray = ki.color.rgb_to_grayscale(gt_im.unsqueeze(0))

        # sobel edges
        gt_edge = ki.filters.sobel(gt_gray).squeeze(0)
        # canny edges TODO ablation test
        # gt_edge= ki.filters.canny(gt_gray)[0]

        gray = ki.color.rgb_to_grayscale(color.unsqueeze(0))

        # sobel edges
        edge = ki.filters.sobel(gray).squeeze(0)

        # edge loss
        if not tracking:
            edge_loss = torch.tensor(0.0).to(self.device) 
        else:
            edge_loss = torch.abs(gt_edge - edge).sum()

        loss += loss_weights['edge'] * edge_loss
        
        if depth is not None:   
            # Depth loss
            if not tracking:
                depth_loss = torch.abs(viewpoint.depth - depth)[mask].mean()
            else:
                depth_loss = torch.abs(viewpoint.depth - depth)[mask].sum()

            loss += loss_weights['depth'] * depth_loss

        # silhouette loss
        if not tracking:
            silhouette_loss = torch.abs(opacity.float() - viewpoint.mask).mean()
        else:
            silhouette_loss = torch.tensor(0.0).to(self.device)

        loss += loss_weights['silhouette'] * silhouette_loss

        # color loss
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()

        rgbl1 = torch.abs(gt_im - color)[color_mask].mean()
        im_loss = (1-dssim_weight) * rgbl1 + dssim_weight * (1.0 - calc_ssim(color, gt_im))
        loss += loss_weights['im'] * im_loss

        # visualize debugging images
        if self.debug_level > 1 and iter % 1 == 0:
            # evaluate gaussian depth rendering
            magnified_diff_depth = torch.abs(viewpoint.depth - depth) * (mask_gt & presence_sil_mask & nan_mask)
            depth_dist = magnified_diff_depth.mean()

            fig, ax = plt.subplots(4, 4, figsize=(12, 12))
            weighted_render_im = color * color_mask
            weighted_im = viewpoint.color * color_mask
            weighted_render_depth = depth * mask
            weighted_depth = viewpoint.depth * mask
            weighted_render_candy = (edge) 
            weighted_candy = (gt_edge)
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
                torch.abs(presence_sil_mask.float() - viewpoint.mask)
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
            ax[1, 2].set_title(f"Diff Depth, Loss: {depth_loss.item()}, \nMean Depth Dist: {depth_dist:.6f}")
            ax_im = ax[0, 3].imshow(opacity.detach().squeeze().cpu())
            cbar = fig.colorbar(ax_im, ax=ax[0, 3])
            ax[0, 3].set_title("Silhouette Mask")
            ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
            ax[1, 3].set_title("Loss Mask")
            ax[2, 0].imshow(viewpoint.mask.detach().squeeze().cpu())
            ax[2, 0].set_title("gt Mask")
            ax_im = ax[2, 1].imshow(magnified_diff_depth.detach().squeeze().cpu(), cmap="jet", vmin=0, vmax=0.02)
            cbar = fig.colorbar(ax_im, ax=ax[2, 1])
            ax[2, 1].set_title("Masked Diff Depth")
            ax[2, 2].imshow(
                (color.permute(1, 2, 0).detach().squeeze().cpu().numpy() * 255).astype(
                    np.uint8
                )
            )
            ax[2, 2].set_title("Rendered RGB")
            ax[2, 3].imshow(gt_im.permute(1, 2, 0).detach().squeeze().cpu())
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
            if tracking:
                suptitle = f"frame_id: {viewpoint.uid} Tracking Iteration: {iter}"
            else:
                suptitle = f"frame_id: {viewpoint.uid} Mapping Iteration: {iter}"
            fig.suptitle(suptitle, fontsize=16)
            # Figure Tight Layout
            fig.tight_layout()
            plot_dir = os.path.join(self.cfg_gs['workdir'], self.cfg_gs['run_name'],'gs_loss_plots') 
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            current_time = datetime.datetime.now()
            filename = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(
                os.path.join(plot_dir, f"{filename}.png"), bbox_inches="tight", dpi=180
            )
            plt.close()

        return loss

    def train_v2(self):
        batch_size = self.cfg_gs['train']["batch_size"]
        lr_dict = self.cfg_gs['train']["lrs"]

        # TODO select keyframes + latest frames + first frame
        if len(self.datas) < batch_size:
            indices = list(range(len(self.queued_data_for_train)))
            random.shuffle(indices)

            # shuffle the data
            batch = [self.datas[i] for i in indices]
        else:
            indices = [0, -1]
            indices = list(range(1, batch_size-1)) + indices
            random.shuffle(indices)
            # random sample batch_size data
            batch = [self.datas[i] for i in indices]

        cam_opt_params = []
        for curr_data in batch:
            time_idx = curr_data["id"]
            viewpoint = curr_data["viewpoint"]
            if time_idx > 0:
                    # intialize the camera optimizer
                cam_opt_params.append(
                    {
                        "params": [viewpoint.cam_rot_delta],
                        "lr": 0.01, 
                        "name": "rot_{}".format(viewpoint.uid),
                    }
                )
                cam_opt_params.append(
                    {
                        "params": [viewpoint.cam_trans_delta],
                        "lr": 0.001,
                        "name": "trans_{}".format(viewpoint.uid),
                    }
                )

        print(f"INFO: Camera Optimizer Params Initialized: {cam_opt_params}")
        pose_optimizer = torch.optim.Adam(cam_opt_params)

        self.gaussians.update_learning_rate(self.gaussians_iter)

        if self.gaussians_iter % 1000 == 0:
            self.gaussians.oneupSHdegree()

        batch_loss = torch.tensor(0.0).to(self.device)
        for iter in range(self.cfg_gs['train']['batch_iters']):
            for curr_data in batch:
                time_idx = curr_data["id"]
                viewpoint = curr_data["viewpoint"]
                for _ in range(100):
                    render_pkg = render(viewpoint, self.gaussians, self.pipe, self.bg)
                    image, viewspace_point_tensor, visibility_filter, radii, depth, opacity= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["opacity"]

                    w2c = viewpoint.world_view_transform.t()

                    # hachy way to render depth and silhouette
                    pts_3d = self.gaussians.get_xyz
                    depth_silhouette = get_depth_and_silhouette(pts_3d, w2c)

                    render_pkg = render(viewpoint, self.gaussians, self.pipe, self.bg, override_color=depth_silhouette)
                    depth_sil = render_pkg["render"]

                    loss = self.get_loss(viewpoint, image, depth_sil[0], depth_sil[1], iter, tracking=True)

                    depth_gt = viewpoint.depth
                    #TODO add sdf loss for gaussians
                    # #  sdf loss
                    # xyz = self.gaussians.get_xyz

                    # predicted_sdf = self.sdf_net(xyz)
                    # breakpoint()
                    pose_optimizer.zero_grad()
                    loss.backward()
   
                    # optimize camera params
                    with torch.no_grad():
                        pose_optimizer.step()
                        for curr_data in batch:
                            viewpoint = curr_data["viewpoint"]
                            converged = viewpoint.update()

                if self.run_gui:
                    gt_image_np = viewpoint.color.detach().cpu().numpy().transpose(1, 2, 0)
                    image_np    = image.detach().cpu().numpy().transpose(1, 2, 0)
                    gt_image_np = (gt_image_np * 255).astype(np.uint8)
                    image_np = (image_np * 255).astype(np.uint8)
                    images = np.hstack((gt_image_np, image_np))
                    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

                    text = f"Mapping Iteration: {iter} on frame {time_idx} Loss: {loss.item():.6f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_color = (255, 255, 255)  # White color
                    text_position = (50, 50)
                    cv2.putText(images, text, text_position, font, font_scale, text_color, font_thickness)

                    xyz = self.gaussians.get_xyz.detach().cpu().numpy()
                    features = self.gaussians.get_features.detach().cpu().numpy()
                    rgb = SH2RGB(features)[:, 0, :]
                    pcl = np.concatenate((xyz, rgb), axis=1)
                    with self.gui_lock:
                        self.gui_dict["image"] = images
                        self.gui_dict["pointcloud"] = pcl


                continue
                with torch.no_grad():
                    # Densification
                    if self.gaussians_iter < self.opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if self.gaussians_iter > self.opt.densify_from_iter and self.gaussians_iter % self.opt.densification_interval == 0:
                            size_threshold = 20 if self.gaussians_iter > self.opt.opacity_reset_interval else None
                            self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cfg_gs['sc_factor'] , size_threshold)

                        if self.gaussians_iter % self.opt.opacity_reset_interval == 0 or (all(self.bg == torch.tensor([1, 1, 1]).to(self.device)) and self.gaussians_iter == self.opt.densify_from_iter):
                            self.gaussians.reset_opacity()

                    # Optimizer step, TODO: ends at the end of training
                    if self.gaussians_iter < 30_000:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)

                self.gaussians_iter += 1 

    def train_v1(self):
        # maping and tracking seperately
        lr_dict = self.cfg_gs['train']["lrs"]

        while len(self.datas) > 0:
            curr_data = self.datas.pop(0)
            time_idx = curr_data["id"]
            viewpoint = curr_data["viewpoint"]
            if time_idx > 0:
                # intialize the camera optimizer
                opt_params = []
                opt_params.append(
                    {
                        "params": [viewpoint.cam_rot_delta],
                        "lr": 0.01, 
                        "name": "rot_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.cam_trans_delta],
                        "lr": 0.001,
                        "name": "trans_{}".format(viewpoint.uid),
                    }
                )

                pose_optimizer = torch.optim.Adam(opt_params)

                best_R, best_t = viewpoint.R, viewpoint.T
                prev_loss = 1e6
                print(f"INFO: TRACKING STARTED FOR FRAME {time_idx}")
                for iter in range(self.cfg_gs['train']['tracking_iters']):
                    render_pkg = render(viewpoint, self.gaussians, self.pipe, self.bg)
                    image, viewspace_point_tensor, visibility_filter, radii, depth, opacity= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["opacity"]
                    
                    loss = self.get_loss(viewpoint, image, depth, opacity, iter, tracking=True)
                    pose_optimizer.zero_grad()
                    loss.backward()
                    
                    if self.run_gui:
                        gt_image_np = viewpoint.color.detach().cpu().numpy().transpose(1, 2, 0)
                        image_np    = image.detach().cpu().numpy().transpose(1, 2, 0)
                        gt_image_np = (gt_image_np * 255).astype(np.uint8)
                        image_np = (image_np * 255).astype(np.uint8)
                        images = np.hstack((gt_image_np, image_np))
                        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

                        text = f"Tracking Iteration: {iter} on frame {time_idx} Loss: {loss.item():.6f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        font_thickness = 2
                        text_color = (255, 255, 255)  # White color
                        text_position = (50, 50)
                        cv2.putText(images, text, text_position, font, font_scale, text_color, font_thickness)

                        xyz = self.gaussians.get_xyz.detach().cpu().numpy()
                        features = self.gaussians.get_features.detach().cpu().numpy()
                        rgb = SH2RGB(features)[:, 0, :]
                        pcl = np.concatenate((xyz, rgb), axis=1)
                        with self.gui_lock:
                            self.gui_dict["image"] = images
                            self.gui_dict["pointcloud"] = pcl

                    print(f"INFO: TRACKING ITERATION {iter} LOSS: {loss.item()}")

                    # with torch.no_grad():
                    #     pose_optimizer.step()
                    #     viewpoint.update()
                    #     converged = viewpoint.update()
                        
                    #     if loss < prev_loss:
                    #         best_R, best_t = viewpoint.R, viewpoint.T
                    #         prev_loss = loss

                    #     # if converged:
                    #     #     break


            print(f"INFO: MAPPING STARTED FOR FRAME {time_idx}")    
            for iter in range(self.cfg_gs['train']['mapping_iters']):
                self.gaussians.update_learning_rate(self.gaussians_iter)

                if self.gaussians_iter % 1000 == 0:
                    self.gaussians.oneupSHdegree()

                render_pkg = render(viewpoint, self.gaussians, self.pipe, self.bg)
                image, viewspace_point_tensor, visibility_filter, radii, depth, opacity= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["opacity"]
                
                w2c = viewpoint.world_view_transform.t()

                # hachy way to render depth and silhouette
                pts_3d = self.gaussians.get_xyz
                depth_silhouette = get_depth_and_silhouette(pts_3d, w2c)

                render_pkg = render(viewpoint, self.gaussians, self.pipe, self.bg, override_color=depth_silhouette)
                silhouette = render_pkg["render"]

                loss = self.get_loss(viewpoint, image, depth, silhouette[1], iter, tracking=False)
                loss.backward()
                #print(f"INFO: MAPPING ITERATION {iter} LOSS: {loss.item()}")

                if self.run_gui:
                    gt_image_np = viewpoint.color.detach().cpu().numpy().transpose(1, 2, 0)
                    image_np    = image.detach().cpu().numpy().transpose(1, 2, 0)
                    gt_image_np = (gt_image_np * 255).astype(np.uint8)
                    image_np = (image_np * 255).astype(np.uint8)
                    images = np.hstack((gt_image_np, image_np))
                    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

                    text = f"Mapping Iteration: {iter} on frame {time_idx} Loss: {loss.item():.6f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_color = (255, 255, 255)  # White color
                    text_position = (50, 50)
                    cv2.putText(images, text, text_position, font, font_scale, text_color, font_thickness)

                    xyz = self.gaussians.get_xyz.detach().cpu().numpy()
                    features = self.gaussians.get_features.detach().cpu().numpy()
                    rgb = SH2RGB(features)[:, 0, :]
                    pcl = np.concatenate((xyz, rgb), axis=1)
                    with self.gui_lock:
                        self.gui_dict["image"] = images
                        self.gui_dict["pointcloud"] = pcl
                
                with torch.no_grad():
                    # Densification
                    if self.gaussians_iter < self.opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if self.gaussians_iter > self.opt.densify_from_iter and self.gaussians_iter % self.opt.densification_interval == 0:
                            size_threshold = 20 if self.gaussians_iter > self.opt.opacity_reset_interval else None
                            self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cfg_gs['sc_factor'] , size_threshold)

                        if self.gaussians_iter % self.opt.opacity_reset_interval == 0 or (all(self.bg == torch.tensor([1, 1, 1]).to(self.device)) and self.gaussians_iter == self.opt.densify_from_iter):
                            self.gaussians.reset_opacity()

                    # Optimizer step, TODO: ends at the end of training
                    if self.gaussians_iter < 30_000:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)

                self.gaussians_iter += 1