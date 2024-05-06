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
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import open3d as o3d
import kornia as ki
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
    transform_to_frame,
    l1_loss_v1,
    matrix_to_quaternion,
)
from datasets.bundlegs_datasets import relative_transformation, datautils
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

# TODO 1. instead of updating camera pose and mapping iteratively, update camera pose and mapping at the same time
# TODO 2. add keyframe selection
# TODO 3. evaluate the error of camera pose optimization on gt_pose
VIS_LOSS_IMAGE = True

def visualize_camera_poses(c2ws):
    if isinstance(c2ws, torch.Tensor):
        c2ws = c2ws.detach().cpu().numpy()
    assert c2ws.shape[1:] == (4, 4) and len(c2ws.shape) == 3
    # Visualize World Coordinate Frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
    camFrames = o3d.geometry.TriangleMesh()
    for c2w in c2ws:
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(c2w)
        camFrames += cam_frame
        
    o3d.visualization.draw_geometries([world_frame, camFrames])

def visualize_param_info(params):
    pcl = o3d.geometry.PointCloud()
    points = params["means3D"].detach().cpu().numpy()
    pcl.points = o3d.utility.Vector3dVector(points)

    if "rgb_colors" in params:
        colors = params["rgb_colors"].detach().cpu().numpy()
        pcl.colors = o3d.utility.Vector3dVector(colors)


    if "logit_opacities" in params:
        # Get the color map by name:
        cm = plt.get_cmap("gist_rainbow")  # purple high, red small

        opacities = params["logit_opacities"].detach().squeeze(1).cpu().numpy()
        # exp
        opacities = np.exp(opacities)
        opa_colors = cm(opacities)

        pcl_opa = o3d.geometry.PointCloud()

        # Set the point cloud data
        pcl_opa.points = o3d.utility.Vector3dVector(points)
        pcl_opa.colors = o3d.utility.Vector3dVector(opa_colors[..., :3])

    else:
        pcl_opa = None
    
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    if pcl_opa is not None:
        o3d.visualization.draw([world_frame, pcl, pcl_opa])
    else:
        o3d.visualization.draw([world_frame, pcl])


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def get_pointcloud(
    color,
    depth,
    mask,
    intrinsics,
    w2c,
    transform_pts=True,
    compute_mean_sq_dist=False,
    mean_sq_dist_method="projective",
):
    def remove_far_pixels(depth, threshold_factor=3.0):
        # Calculate mean and standard deviation of depth values
        mean_depth = np.mean(depth[depth > 1e-5])
        std_depth = np.std(depth[depth > 1e-5])

        # Set a threshold based on mean and standard deviation
        threshold = mean_depth + threshold_factor * std_depth

        # Threshold depth image
        depth_mask = depth < threshold
        filtered_depth = depth.copy()
        filtered_depth[depth_mask] = 0  # Set far pixels to 0

        return depth_mask, filtered_depth

    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width).cuda().float(),
        torch.arange(height).cuda().float(),
        indexing="xy",
    )
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY) / 2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(
        -1, 3
    )  # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # remove depth outlier
    masked_depth = depth[0] * mask.reshape(height, width)
    depth_mask, _ = remove_far_pixels(masked_depth.cpu().numpy())

    mask = mask.bool().cuda() & torch.tensor(depth_mask.reshape(-1)).cuda()
    point_cld = point_cld[mask]
    if compute_mean_sq_dist:
        mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

def add_new_gaussians(
    params, variables, curr_data, sil_thres, depth_thres, time_idx, mean_sq_dist_method, gaussian_distribution
):
    # add new gaussians with non-presence mask
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(
        params, time_idx, gaussians_grad=False, camera_grad=False
    )

    depth_sil_rendervar = transformed_params2depthplussilhouette(
        params, curr_data["w2c"], transformed_gaussians
    )
    (
        depth_sil,
        _,
        _,
    ) = Renderer(
        raster_settings=curr_data["cam"]
    )(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = silhouette < sil_thres
    
    # Check for new foreground objects by using GT depth
    gt_mask = curr_data["mask"][0, :, :]
    gt_depth = curr_data["depth"][0, :, :]
    # remove nan in depth
    gt_depth = torch.nan_to_num_(gt_depth, nan=0.0)
    render_depth = depth_sil[0, :, :]

    mask = gt_mask.bool() & (gt_depth > 0)
    # Filter out invalid depth values, remove nan in depth

    non_presence_depth_mask = (render_depth-gt_depth) > depth_thres
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask & mask
    
    # TODO how to use non_presence_depth_mask
    #non_presence_mask = non_presence_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)
    # TODO add new gaussian
    if False:
        plt.subplot(2, 4, 1)
        plt.imshow(silhouette.detach().cpu().numpy())
        plt.title("silhouette")
        plt.subplot(2, 4, 2)
        plt.imshow(mask.squeeze().cpu().numpy())
        plt.title("gt_mask")
        plt.subplot(2, 4, 3)
        plt.imshow(gt_depth.detach().cpu().numpy())
        plt.colorbar()
        plt.title("gt depth")
        plt.subplot(2, 4, 4)
        plt.imshow(render_depth.detach().cpu().numpy())
        plt.colorbar()
        plt.title("rendered depth")
        plt.subplot(2, 4, 5)
        plt.imshow(
            gt_depth.detach().cpu().numpy() - render_depth.detach().cpu().numpy()
        )
        plt.colorbar()
        plt.title("depth_loss")
        plt.subplot(2, 4, 6)
        plt.imshow(non_presence_mask.detach().cpu().numpy().reshape(*mask.shape))
        plt.colorbar()
        plt.title("non presence mask")
        plt.subplot(2, 4, 7)
        plt.imshow(non_presence_depth_mask.detach().cpu().numpy())
        plt.colorbar()
        plt.title("non_presence_depth_mask")
        plt.subplot(2, 4, 8)
        plt.imshow(non_presence_sil_mask.detach().cpu().numpy())
        plt.colorbar()
        plt.title("non_presence_sil_mask")

        plt.show()

    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #    size=0.5, origin=[0, 0, 0]
    # )

    # points = params["means3D"].detach().cpu().numpy().copy()
    # colors = params["rgb_colors"].detach().cpu().numpy().copy()
    # # Create a point cloud object
    # pcl_prev = o3d.geometry.PointCloud()

    # # Set the point cloud data
    # pcl_prev.points = o3d.utility.Vector3dVector(points)
    # pcl_prev.colors = o3d.utility.Vector3dVector(colors)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(
            params["cam_unnorm_rots"][..., time_idx].detach()
        )
        curr_cam_tran = params["cam_trans"][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        new_pt_cld, mean3_sq_dist = get_pointcloud(
            curr_data["im"],
            curr_data["depth"],
            non_presence_mask,
            curr_data["intrinsics"],
            curr_w2c,
            compute_mean_sq_dist=True,
            mean_sq_dist_method=mean_sq_dist_method,
        )
        new_params = initialize_new_params(
            new_pt_cld, mean3_sq_dist, gaussian_distribution
        )
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(
                torch.cat((params[k], v), dim=0).requires_grad_(True)
            )
        num_pts = params["means3D"].shape[0]
        variables["means2D_gradient_accum"] = torch.zeros(
            num_pts, device="cuda"
        ).float()
        variables["denom"] = torch.zeros(num_pts, device="cuda").float()
        variables["max_2D_radius"] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = (
            time_idx * torch.ones(new_pt_cld.shape[0], device="cuda").float()
        )
        variables["timestep"] = torch.cat(
            (variables["timestep"], new_timestep), dim=0
        )

    # try:
    #    points = new_params["means3D"].detach().cpu().numpy().copy()
    #    colors = new_params["rgb_colors"].detach().cpu().numpy().copy()
    #    # Create a point cloud object
    #    pcl = o3d.geometry.PointCloud()

    #    # Set the point cloud data
    #    pcl.points = o3d.utility.Vector3dVector(points)
    #    pcl.colors = o3d.utility.Vector3dVector(colors)

    #    # Visualize the point cloud
    #    o3d.visualization.draw([coordinate_frame, pcl_prev, pcl])
    # except:
    #    print("Visualization of points fails")

def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        "means3D": means3D,
        "rgb_colors": new_pt_cld[:, 3:6],
        "unnorm_rotations": unnorm_rots,
        "logit_opacities": logit_opacities,
        "log_scales": log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(
                torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
            )
        else:
            params[k] = torch.nn.Parameter(
                v.cuda().float().contiguous().requires_grad_(True)
            )

    return params


def initialize_optimizer_sep(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [
        {"params": [v], "name": k, "lr": lrs[k]} for k, v in params.items()
    ]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

class GSRunner:
    def __init__(
        self, cfg_gs, rgbs, depths, masks, K, poses, total_num_frames, dtype=torch.float, offset=None, scale=None, pointcloud=None, poses_gt=None, wandb_run=None):
        # build_octree_pcd=pcd_normalized,):                    # TODO from pcd add new gaussians
        # TODO check poses c2w or obs_in_cam, z axis
        # preprocess data -> to tensor
        self.device = torch.device(cfg_gs["primary_device"])
        self.dtype = dtype
        self._total_num_frames = total_num_frames
        self.cfg_gs = cfg_gs

        self._offset = offset
        self._scale = scale

        colors, depths, masks = self._preprocess_images_data(rgbs, depths, masks)

        self._fisrt_c2w = poses[0].copy()

        #ForkedPdb().set_trace()
        poses_gt = self._preprocess_poses(poses_gt)
        self._w2c_gt = torch.linalg.inv(poses_gt)

        poses = self._preprocess_poses(poses)

        intrinsics = torch.eye(4).to(self.device)
        intrinsics[:3, :3] = torch.tensor(K)

        first_data = (colors[0], depths[0], masks[0], intrinsics, poses[0])
        
        # intialize pointcloud if provided
        if pointcloud is not None:
            assert isinstance(pointcloud, np.ndarray) and pointcloud.shape[1] == 6
            
            self._initialize_first_timestep(
                first_data,
                total_num_frames,
                cfg_gs["scene_radius_depth_ratio"],
                cfg_gs["mean_sq_dist_method"],
                gaussian_distribution=cfg_gs["gaussian_distribution"],
                pointcloud=pointcloud,
            )
        else:
            self._initialize_first_timestep(
                first_data,
                total_num_frames,
                cfg_gs["scene_radius_depth_ratio"],
                cfg_gs["mean_sq_dist_method"],
                gaussian_distribution=cfg_gs["gaussian_distribution"],
            )
        # Initialize list to keep track of Keyframes
        self.keyframe_list = []
        self.keyframe_time_indices = []

        # Init Variables to keep track of ground truth poses and runtimes
        self.gt_w2c_all_frames = []
        self.curr_frame_id = 0
        self.tracking_iter_time_sum = 0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0
        self.mapping_iter_time_count = 0
        self.tracking_frame_time_sum = 0
        self.tracking_frame_time_count = 0
        self.mapping_frame_time_sum = 0
        self.mapping_frame_time_count = 0

            
        self.queued_data_for_train = []
        # for training_together
        # self.queued_data_for_train = deque()

        # prepare data from training
        for color, depth, mask, pose in zip(colors, depths, masks, poses):
            w2c = torch.linalg.inv(pose)
            # TODO check B,C,W,H
            color = color.permute(2, 0, 1)  # /255.
            # color = color.permute(2, 0, 1/255.
            depth = depth.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            self.gt_w2c_all_frames.append(w2c)
            iter_time_idx = self.curr_frame_id

            self.queued_data_for_train.append(
                {
                    "cam": self.cam,
                    "im": color,
                    "depth": depth,
                    "mask": mask,
                    "id": iter_time_idx,
                    "intrinsics": self.intrinsics,
                    "w2c": w2c,
                    "first_c2w": self.first_frame_w2c,
                    "iter_gt_w2c_list": self.gt_w2c_all_frames.copy(),
                    "seen": False,
                    "optimized": False
                }
            )
            self.curr_frame_id += 1

    
    def _preprocess_poses(self, poses):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        # opencv c2w to opengl
        gl_poses = poses.copy()
        gl_poses[:, :3, 1:3] = -gl_poses[:, :3, 1:3]
        gl_poses = torch.from_numpy(gl_poses).to(self.device).type(self.dtype)
        gl_pose_ret = relative_transformation(
            gl_poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            gl_poses,
            orthogonal_rotations=False,
        )
        # first_c2w = gl_poses[0].unsqueeze(0)
        # gl_pose_ret = torch.matmul(first_c2w, gl_pose_ret)
         
        return gl_pose_ret
    
    def _preprocess_images_data(self, colors, depths, masks):
        assert len(colors.shape) == len(depths.shape) == len(masks.shape) == 4
        # from B,H,W,C -> B,C,W,H
        # color = datautils.channels_first(color)
        # mask = mask.astype(np.float32)
        # mask = datautils.channels_first(mask)
        # depth = datautils.channels_first(depth)
        if colors.dtype == 'uint8':
            colors = colors.astype(np.float32) / 255.

        if masks.dtype == 'uint8' and masks.max() == 255:
            masks = masks / 255

        # to tensor
        colors = torch.from_numpy(colors).to(self.device).type(self.dtype)
        depths = torch.from_numpy(depths).to(self.device).type(self.dtype)
        masks = torch.from_numpy(masks).to(self.device).type(self.dtype)

        return (colors, depths, masks)

    def _initialize_first_timestep(
        self,
        first_data,
        num_frames,
        scene_radius_depth_ratio,
        mean_sq_dist_method,
        gaussian_distribution=None,
        pointcloud=None,        
    ):
        color, depth, mask, intrinsics, pose = first_data

        # color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        color = color.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask = mask.permute(2, 0, 1)

        # Process Camera Parameters
        intrinsics = intrinsics[:3, :3]
        w2c = torch.linalg.inv(pose)

        # Setup Camera
        cam = setup_camera(
            color.shape[2],
            color.shape[1],
            intrinsics.cpu().numpy(),
            w2c.detach().cpu().numpy(),
        )

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        mask_ = (depth > 0) & mask.bool()  # Mask out invalid depth values
        mask = mask.reshape(-1)

        if pointcloud is not None:
            init_pt_cld = torch.from_numpy(pointcloud).to(self.device).type(self.dtype)
            depth_z = init_pt_cld[:, 2]
            mean3_sq_dist = depth_z / ((intrinsics[0, 0] + intrinsics[1, 1]) / 2)
            mean3_sq_dist = mean3_sq_dist**2

        else:
            init_pt_cld, mean3_sq_dist = get_pointcloud(
                color,
                depth,
                mask,
                intrinsics,
                w2c,
                compute_mean_sq_dist=True,
                mean_sq_dist_method=mean_sq_dist_method,
            )

        params, variables = self._initialize_params(
            init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution
        )

        # Initialize an estimate of scene radius for Gaussian-Splatting Densification, TODO understanding variables scene_radius
        variables["scene_radius"] = torch.max(depth) / scene_radius_depth_ratio

        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = w2c  # relative w2c np.eye(4)
        self.cam = cam

    def _initialize_params(
        self, init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution
    ):
        num_pts = init_pt_cld.shape[0]
        means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]
        unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 4]
        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(
                torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)
            )
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(
                torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3)
            )
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
        params = {
            "means3D": means3D,
            "rgb_colors": init_pt_cld[:, 3:6],
            "unnorm_rotations": unnorm_rots,
            "logit_opacities": logit_opacities,
            "log_scales": log_scales,
        }

        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
        params["cam_unnorm_rots"] = cam_rots
        params["cam_trans"] = np.zeros((1, 3, num_frames))

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(
                    torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
                )
            else:
                params[k] = torch.nn.Parameter(
                    v.cuda().float().contiguous().requires_grad_(True)
                )

        variables = {
            "max_2D_radius": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "means2D_gradient_accum": torch.zeros(params["means3D"].shape[0])
            .cuda()
            .float(),
            "denom": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "timestep": torch.zeros(params["means3D"].shape[0]).cuda().float(),
        }

        return params, variables

    def add_new_frames(self, rgbs, depths, masks, poses):
        # TODO need to redesign the add function
        # new_pcd=pcd_normalized,)

        colors, depths, masks = self._preprocess_images_data(rgbs, depths, masks)

        #ForkedPdb().set_trace()
        # get latest poss
        poses = self._preprocess_poses(poses)[-1, ...].unsqueeze(0)

        # prepare data from training
        for color, depth, mask, pose in zip(colors, depths, masks, poses):
            w2c = torch.linalg.inv(pose)
            color = color.permute(2, 0, 1)  # /255.
            depth = depth.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            self.gt_w2c_all_frames.append(w2c)
            iter_time_idx = self.curr_frame_id

            self.queued_data_for_train.append(
                {
                    "cam": self.cam,          # TODO remove it from dict
                    "im": color,
                    "depth": depth,
                    "mask": mask,
                    "id": iter_time_idx,
                    "intrinsics": self.intrinsics,
                    "w2c": w2c,
                    "first_c2w": self.first_frame_w2c,
                    "iter_gt_w2c_list": self.gt_w2c_all_frames.copy(),      # only for trajectory prediction
                    "seen": False,
                    "optimized": False
                }
            )
            self.curr_frame_id += 1

    def get_loss(
        self,
        params,
        curr_data,
        variables,
        iter_time_idx,
        loss_weights,
        use_sil_for_loss,
        sil_thres,
        use_l1,
        ignore_outlier_depth_loss,
        tracking=False,
        do_ba=False,
        training_iter=None,
    ):
        def erosion(bool_image, kernel_size=3):
            # Convert boolean image to a tensor with dtype torch.float32
            bool_image_tensor = bool_image.float()

            # Define a kernel for erosion
            kernel = torch.ones(kernel_size, kernel_size).to(bool_image.device)

            # Apply erosion using convolution
            eroded_image = F.conv2d(
                bool_image_tensor.unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size // 2,
            )

            # Threshold to convert back to boolean image
            eroded_image = (eroded_image >= kernel_size**2).squeeze(0).squeeze(0)

            return eroded_image
            # Initialize Loss Dictionary

        losses = {}

        if tracking:
            # Get current frame Gaussians, where only the camera pose gets gradient
            transformed_gaussians = transform_to_frame(
                params, iter_time_idx, gaussians_grad=False, camera_grad=True
            )
        else:
            if do_ba:
                # Get current frame Gaussians, where both camera pose and Gaussians get gradient
                transformed_gaussians = transform_to_frame(
                    params, iter_time_idx, gaussians_grad=True, camera_grad=True
                )
            else:
                # Get current frame Gaussians, where only the Gaussians get gradient
                transformed_gaussians = transform_to_frame(
                    params, iter_time_idx, gaussians_grad=True, camera_grad=False
                )

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(
            params, curr_data["w2c"], transformed_gaussians
        )

        # RGB Rendering
        rendervar["means2D"].retain_grad()
        (
            im,
            radius,
            _,
        ) = Renderer(
            raster_settings=curr_data["cam"]
        )(**rendervar)
        variables["means2D"] = rendervar[
            "means2D"
        ]  # Gradient only accum from colour render for densification

        # Depth & Silhouette Rendering
        (
            depth_sil,
            _,
            _,
        ) = Renderer(
            raster_settings=curr_data["cam"]
        )(**depth_sil_rendervar)


        depth = depth_sil[0, :, :].unsqueeze(0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = silhouette > sil_thres
        depth_sq = depth_sil[2, :, :].unsqueeze(0)
        uncertainty = depth_sq - depth**2
        uncertainty = uncertainty.detach()
        # M    ask with valid depth values (accounts for outlier depth values)
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        """
        #if ignore_outlier_depth_loss:
        #    depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        #    mask = (depth_error < 10*depth_error.median())
        #    mask = mask & (curr_data['depth'] > 0)
        #else:
        #    mask = (curr_data['depth'] > 0)
        """
        mask_gt = curr_data["mask"].bool()
        gt_im = curr_data["im"] * mask_gt

        # Mask with presence silhouette mask (accounts for empty space)
        if tracking:
            mask = mask_gt | presence_sil_mask
            mask = ki.morphology.dilation(
                mask.float().unsqueeze(0), torch.ones(10, 10).cuda()
            )
            mask = mask.detach().squeeze(0).bool()
        else:
            mask = mask_gt & presence_sil_mask & nan_mask
            mask = mask.detach()

        # canny edge detection
        gt_gray = ki.color.rgb_to_grayscale(curr_data["im"].unsqueeze(0))

        # sobel edges
        gt_edge = ki.filters.sobel(gt_gray).squeeze(0)
        # canny edges
        # gt_edge= ki.filters.canny(gt_gray)[0]

        gray = ki.color.rgb_to_grayscale(im.unsqueeze(0))

        # sobel edges
        edge = ki.filters.sobel(gray).squeeze(0)
        # edge edges
        # edge= ki.filters.canny(gray)[0]

        # losses['edge'] = torch.abs(gt_edge - edge)[mask.unsqueeze(0)].sum()
        if tracking:
            losses["edge"] = torch.abs(gt_edge - edge)[mask].sum()  # sum()
        else:
            losses["edge"] = torch.abs(gt_edge - edge)[mask].mean()

        # Depth loss
        if tracking:
            losses["depth"] = torch.abs(curr_data["depth"] - depth)[mask].sum()
        else:
            losses["depth"] = torch.abs(curr_data["depth"] - depth)[mask].mean()

        if tracking:
            losses["silhouette"] = torch.abs(silhouette.float() - curr_data["mask"]).sum()
        else:
            losses["silhouette"] = torch.abs(silhouette.float() - curr_data["mask"]).sum()


        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        if tracking:
            rgbl1 = torch.abs(gt_im - im)[color_mask].sum()
        else:
            rgbl1 = torch.abs(gt_im - im)[color_mask].mean()

        # rgbl1 = torch.abs(gt_im - im).sum()
        losses["im"] = 0.8 * rgbl1 + 0.2 * (1.0 - calc_ssim(im, gt_im))

        if VIS_LOSS_IMAGE and (training_iter) % 5 == 0:
            # define a function which returns an image as numpy array from figure

            fig, ax = plt.subplots(4, 4, figsize=(12, 12))
            weighted_render_im = im * color_mask
            weighted_im = curr_data["im"] * color_mask
            weighted_render_depth = depth * mask
            weighted_depth = curr_data["depth"] * mask
            weighted_render_candy = (edge * mask)
            weighted_candy = (gt_edge * mask)
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
                torch.abs(presence_sil_mask.float() - curr_data["mask"])
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
            ax[1, 1].imshow(
                weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6
            )
            ax[1, 1].set_title("Weighted Rendered Depth")
            ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
            ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
            ax_im = ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
            cbar = fig.colorbar(ax_im, ax=ax[1, 2])
            ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
            ax_im = ax[0, 3].imshow(silhouette.detach().squeeze().cpu())
            cbar = fig.colorbar(ax_im, ax=ax[0, 3])
            ax[0, 3].set_title("Silhouette Mask")
            ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
            ax[1, 3].set_title("Loss Mask")
            ax[2, 0].imshow(curr_data["mask"].detach().squeeze().cpu())
            ax[2, 0].set_title("gt Mask")
            ax_im = ax[2, 1].imshow(depth_sq.detach().squeeze().cpu())
            # Add colorbar
            cbar = fig.colorbar(ax_im, ax=ax[2, 1])
            ax[2, 1].set_title("uncertainty mask")
            ax[2, 2].imshow(
                (im.permute(1, 2, 0).detach().squeeze().cpu().numpy() * 255).astype(
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
            ax[3, 2].set_title(f"Diff Edge: {torch.round(losses['edge'])}")
            ax[3, 3].imshow(diff_sil, cmap="jet")
            ax[3, 3].set_title(f"Silhouette Loss: {float(losses['silhouette'])}")
            # Turn off axis
            for i in range(2):
                for j in range(4):
                    ax[i, j].axis("off")
            # Set Title
            if tracking:
                suptitle = f"frame_id: {curr_data['id']}, Tracking Iteration: {training_iter}"
            else:
                suptitle = f"frame_id: {curr_data['id']}, Mapping Iteration: {training_iter}"
            fig.suptitle(suptitle, fontsize=16)
            # Figure Tight Layout
            fig.tight_layout()
            plot_dir = "/home/yjin/repos/BundleSDF/gs_debug_imgs"
            os.makedirs(plot_dir, exist_ok=True)
            current_time = datetime.datetime.now()
            filename = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            #plt.show()
            plt.savefig(
                os.path.join(plot_dir, f"{filename}.png"), bbox_inches="tight", dpi=180
            )
            plt.close()

            # plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
            # cv2.imshow('Diff Images', plot_img)
            # cv2.waitKey(0)
            ## Save Tracking Loss Viz
            # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
            # os.makedirs(save_plot_dir, exist_ok=True)
            # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % training_iter), bbox_inches='tight')
            # plt.close()

        weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}

        #TODO wandb the loss
        msg = f'INFO: Tracking iter {training_iter}\n' if tracking else f'INFO: Mapping iter {training_iter}\n'
        for k, v in losses.items():
            msg += f"\t{k} loss: {v.item():.6f} weighted loss: {v.item() * loss_weights[k]:.6f}\n"


        loss = sum(weighted_losses.values())
        msg += f"total loss: {loss.item()}"
        #print(msg)

        seen = radius > 0
        variables["max_2D_radius"][seen] = torch.max(
            radius[seen], variables["max_2D_radius"][seen]
        )
        variables["seen"] = seen
        weighted_losses["loss"] = loss

        return loss, variables, weighted_losses


 
    def initialize_optimizer(self, lr_dict):
        lrs = lr_dict
        param_groups = [
            {"params": [v], "name": k, "lr": lrs[k]} for k, v in self.params.items()
        ]
        return torch.optim.Adam(param_groups)

    def get_xyz_rgb_params(self):
        points = self.params["means3D"].detach().cpu().numpy()
        colors = self.params["rgb_colors"].detach().cpu().numpy()
        center = points.mean(axis=0)
        points = points - center
        # Create a point cloud object
        return np.concatenate((points, colors), axis=1)

    def get_optimized_cam_poses(self):
        opt_cam_poses = []
        with torch.no_grad():
            # Get the current estimated rotation & translation
            for time_idx in range(self.curr_frame_id):
                cam_rot = F.normalize(
                    self.params["cam_unnorm_rots"][..., time_idx].detach()
                )
                cam_tran = self.params["cam_trans"][..., time_idx].detach()
                w2c = torch.eye(4).cuda().float()
                w2c[:3, :3] = build_rotation(cam_rot)
                w2c[:3, 3] = cam_tran

                opt_cam_poses.append(self._fisrt_c2w @ w2c.cpu().numpy())
                
        return np.asarray(opt_cam_poses)

    def train(self):
        # TODO N-iter training, set batch and update optimizer once per batch
        batch_size = self.cfg_gs['train']["batch_size"]
        lr_dict = self.cfg_gs['train']["lrs"]
        batch_iters = self.cfg_gs['train']["batch_iters"]
        progress_bar = tqdm(total=self.cfg_gs['train']["num_epochs"])
        # intialize optimizer
        for epoch in range(self.cfg_gs['train']["num_epochs"]):
            if len(self.queued_data_for_train) < batch_size:
                indices = list(range(len(self.queued_data_for_train)))
                random.shuffle(indices)

                # shuffle the data
                batch_data = [self.queued_data_for_train[i] for i in indices]
            else:
                indices = list(range(batch_size))
                random.shuffle(indices)
                # random sample batch_size data
                batch_data = [self.queued_data_for_train[i] for i in indices]
            
            # update camera rotation and translation in params 
            with torch.no_grad():
                for curr_data in batch_data:
                    time_idx = curr_data["id"]
                    rel_w2c = curr_data["w2c"]
                    if time_idx > 0:
                        # update initial pose relative to frame 0
                        rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                        rel_w2c_tran = rel_w2c[:3, 3].detach()
                        # Update the camera parameters
                        self.params["cam_unnorm_rots"][..., time_idx] = rel_w2c_rot_quat
                        self.params["cam_trans"][..., time_idx] = rel_w2c_tran

                        if not curr_data["seen"] and self.cfg_gs['add_new_gaussians']:
                            print(f"INFO: Adding new gaussians for frame {time_idx}")
                            pcl_num = self.params["means3D"].shape[0]
                            # add new gaussians
                            add_new_gaussians(
                                self.params,
                                self.variables,
                                curr_data,
                                self.cfg_gs['add_gaussian_dict']['sil_thres'],
                                self.cfg_gs['add_gaussian_dict']['depth_thres'],
                                time_idx,
                                self.cfg_gs["mean_sq_dist_method"],
                                self.cfg_gs["gaussian_distribution"],
                            )         
                            curr_data['seen'] = True
                            print(f"INFO: Adding new gaussians done, the number of gaussians added: {self.params['means3D'].shape[0] - pcl_num}")

            optimizer = self.initialize_optimizer(lr_dict)
            for iter in range(batch_iters):
                loss, losses = self.train_once(batch_data, iter)
                loss.backward()

                print(f"INFO: mean of opcaity of params: {torch.mean(self.params['logit_opacities'])}, \
                      max of opacity of params: {torch.max(self.params['logit_opacities'])}, \
                        min of opacity of params: {torch.min(self.params['logit_opacities'])}")
                      
                optimizer.step()

                # TODO save keyframes depending on the losses

                with torch.no_grad():
                    # Prune Gaussians
                    if self.cfg_gs['train']["prune_gaussians"]:
                        print(f"INFO: number of gaussians before pruning: {self.params['means3D'].shape[0]}")
                        pcl_num = self.params["means3D"].shape[0]
                        self.params, self.variables = prune_gaussians(
                            self.params,
                            self.variables,
                            optimizer,
                            iter,
                            self.cfg_gs['train']["pruning_dict"],
                        )
                        print(f"INFO: Gaussian Pruning Done. the number of gaussians pruned: {pcl_num - self.params['means3D'].shape[0]}, the remaining number of gaussians: {self.params['means3D'].shape[0]}")

                    # Gaussian-Splatting's Gradient-based Densification
                    if self.cfg_gs['train']["use_gaussian_splatting_densification"]: 
                        self.params, self.variables = densify(
                            self.params,
                            self.variables,
                            optimizer,
                            iter,
                            self.cfg_gs['train']["densify_dict"],
                        )
                optimizer.zero_grad(set_to_none=True)

                wandb.log(losses)

                # prune gaussians

                # denseify gaussians
            progress_bar.update(1)

    def train_once(self, batch_data, training_iter, dssim_weight=0.2):
        losses = {k: torch.tensor(0.0).to(self.device) for k in ["edge", "depth", "silhouette", "im"]}

        for curr_data in batch_data:
            iter_time_idx = curr_data["id"]
            
            # transform the gaussians to the current frame
            transformed_gaussians = transform_to_frame(
                self.params, iter_time_idx, gaussians_grad=True, camera_grad=True
            )

            # Initialize Render Variables
            rendervar = transformed_params2rendervar(self.params, transformed_gaussians)
            depth_sil_rendervar = transformed_params2depthplussilhouette(
                self.params, curr_data["first_c2w"], transformed_gaussians
            )
            # RGB Rendering
            rendervar["means2D"].retain_grad()
            (
                im,
                radius,
                _,
            ) = Renderer(
                raster_settings=self.cam
            )(**rendervar)
            self.variables["means2D"] = rendervar[
                "means2D"
            ]  # Gradient only accum from colour render for densification

            # Depth & Silhouette Rendering
            (
                depth_sil,
                _,
                _,
            ) = Renderer(
                raster_settings=self.cam
            )(**depth_sil_rendervar)

            depth = depth_sil[0, :, :].unsqueeze(0)
            silhouette = depth_sil[1, :, :]
            presence_sil_mask = silhouette > self.cfg_gs["train"]["sil_thres"]
            depth_sq = depth_sil[2, :, :].unsqueeze(0)
            uncertainty = depth_sq - depth**2
            uncertainty = uncertainty.detach()
            # M    ask with valid depth values (accounts for outlier depth values)
            nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))

            mask_gt = curr_data["mask"].bool()
            gt_im = curr_data["im"] * mask_gt

            mask = mask_gt | presence_sil_mask & nan_mask

            # canny edge detection
            gt_gray = ki.color.rgb_to_grayscale(curr_data["im"].unsqueeze(0))

            # sobel edges
            gt_edge = ki.filters.sobel(gt_gray).squeeze(0)
            # canny edges
            # gt_edge= ki.filters.canny(gt_gray)[0]

            gray = ki.color.rgb_to_grayscale(im.unsqueeze(0))

            # sobel edges
            edge = ki.filters.sobel(gray).squeeze(0)
            # edge edges

            losses["edge"] += torch.abs(gt_edge - edge)[mask].mean()

            # Depth loss
            losses["depth"] += torch.abs(curr_data["depth"] - depth)[mask].mean()

            # silhouette loss
            losses["silhouette"] += torch.abs(silhouette.float() - curr_data["mask"]).mean()

            # color loss
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()

            rgbl1 = torch.abs(gt_im - im)[color_mask].mean()
            losses["im"] += (1-dssim_weight) * rgbl1 + dssim_weight * (1.0 - calc_ssim(im, gt_im))

            wandb.log(losses)
            # visualize debugging images
            if VIS_LOSS_IMAGE and (training_iter) % 5 == 0:
                # define a function which returns an image as numpy array from figure
                fig, ax = plt.subplots(4, 4, figsize=(12, 12))
                weighted_render_im = im * color_mask
                weighted_im = curr_data["im"] * color_mask
                weighted_render_depth = depth * mask
                weighted_depth = curr_data["depth"] * mask
                weighted_render_candy = (edge * mask)
                weighted_candy = (gt_edge * mask)
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
                    torch.abs(presence_sil_mask.float() - curr_data["mask"])
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
                ax[1, 1].imshow(
                    weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6
                )
                ax[1, 1].set_title("Weighted Rendered Depth")
                ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
                ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
                ax_im = ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
                cbar = fig.colorbar(ax_im, ax=ax[1, 2])
                ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
                ax_im = ax[0, 3].imshow(silhouette.detach().squeeze().cpu())
                cbar = fig.colorbar(ax_im, ax=ax[0, 3])
                ax[0, 3].set_title("Silhouette Mask")
                ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
                ax[1, 3].set_title("Loss Mask")
                ax[2, 0].imshow(curr_data["mask"].detach().squeeze().cpu())
                ax[2, 0].set_title("gt Mask")
                ax_im = ax[2, 1].imshow(depth_sq.detach().squeeze().cpu())
                # Add colorbar
                cbar = fig.colorbar(ax_im, ax=ax[2, 1])
                ax[2, 1].set_title("uncertainty mask")
                ax[2, 2].imshow(
                    (im.permute(1, 2, 0).detach().squeeze().cpu().numpy() * 255).astype(
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
                ax[3, 2].set_title(f"Diff Edge: {torch.round(losses['edge'])}")
                ax[3, 3].imshow(diff_sil, cmap="jet")
                ax[3, 3].set_title(f"Silhouette Loss: {float(losses['silhouette'])}")
                # Turn off axis
                for i in range(2):
                    for j in range(4):
                        ax[i, j].axis("off")
                # Set Title
                suptitle = f"frame_id: {curr_data['id']}, Training Iteration: {training_iter}"
                fig.suptitle(suptitle, fontsize=16)
                # Figure Tight Layout
                fig.tight_layout()
                plot_dir = "/home/yjin/repos/BundleSDF/gs_debug_imgs"
                os.makedirs(plot_dir, exist_ok=True)
                current_time = datetime.datetime.now()
                filename = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                #plt.show()
                plt.savefig(
                    os.path.join(plot_dir, f"{filename}.png"), bbox_inches="tight", dpi=180
                )
                plt.close()
                
                seen = radius > 0
                self.variables["max_2D_radius"][seen] = torch.max(
                    radius[seen], self.variables["max_2D_radius"][seen]
                )
                self.variables["seen"] = seen

        loss_weights = self.cfg_gs["train"]["loss_weights"]
        weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}

        loss = sum(weighted_losses.values())/len(batch_data)

        weighted_losses['loss'] = loss

        # convert to numpy
        for k, v in weighted_losses.items():
            weighted_losses[k] = v.item()
    
        return loss, weighted_losses

    def train_together(self):
        # TODO pop the earliest data
        config = self.cfg_gs
        params = self.params
        variables = self.variables

        #for curr_data in data_for_train:
        while self.queued_data_for_train:
            curr_data = self.queued_data_for_train.popleft()
            time_idx = curr_data["id"]
            curr_gt_w2c = curr_data["iter_gt_w2c_list"]

            color = curr_data['im']
            depth = curr_data['depth']
            mask = curr_data['mask']
            intrinsics = curr_data['intrinsics']
            print(f"INFO: Training, processing frame_id : {time_idx}")

            if time_idx > 0:
                with torch.no_grad():
                    # Get the ground truth pose relative to frame 0
                    rel_w2c = curr_gt_w2c[-1]
                    rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                    rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                    rel_w2c_tran = rel_w2c[:3, 3].detach()
                    # Update the camera parameters
                    params["cam_unnorm_rots"][..., time_idx] = rel_w2c_rot_quat
                    params["cam_trans"][..., time_idx] = rel_w2c_tran

                # tracking
                tracking_start_time = time.time()

                # reset optimizer & learning rates for tracking
                optimizer = initialize_optimizer_sep(
                    params, config["tracking"]["lrs"], tracking=True
                )
                # keep track of best candidate rotation & translation
                candidate_cam_unnorm_rot = (
                    params["cam_unnorm_rots"][..., time_idx].detach().clone()
                )
                candidate_cam_tran = params["cam_trans"][..., time_idx].detach().clone()
                current_min_loss = float(1e20)

                # tracking optimization
                tracking_iter = 0
                do_continue_slam = False
                num_iters_tracking = config["tracking"]["num_iters"]
                progress_bar = tqdm(
                    range(num_iters_tracking), desc=f"tracking time step: {time_idx}"
                )
                while True:
                    # loss for current frame
                    loss, variables, losses = self.get_loss(
                        params,
                        curr_data,
                        variables,
                        time_idx,
                        config["tracking"]["loss_weights"],
                        config["tracking"]["use_sil_for_loss"],
                        config["tracking"]["sil_thres"],
                        config["tracking"]["use_l1"],
                        config["tracking"]["ignore_outlier_depth_loss"],
                        tracking=True,
                        training_iter=tracking_iter,
                    )
                    if config["use_wandb"]:
                        # report loss
                        wandb_tracking_step = report_loss(
                            losses, wandb_run, wandb_tracking_step, tracking=True
                        )
                    # backprop
                    loss.backward()
                    # optimizer update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        # save the best candidate rotation & translation
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_unnorm_rot = (
                                params["cam_unnorm_rots"][..., time_idx].detach().clone()
                            )
                            candidate_cam_tran = (
                                params["cam_trans"][..., time_idx].detach().clone()
                            )
                        # report progress
                        if config["report_iter_progress"]:
                            if config["use_wandb"]:
                                report_progress(
                                    params,
                                    curr_data,
                                    tracking_iter + 1,
                                    progress_bar,
                                    time_idx,
                                    sil_thres=config["tracking"]["sil_thres"],
                                    tracking=True,
                                    wandb_run=wandb_run,
                                    wandb_step=wandb_tracking_step,
                                    wandb_save_qual=config["wandb"]["save_qual"],
                                )
                            else:
                                report_progress(
                                    params,
                                    curr_data,
                                    tracking_iter + 1,
                                    progress_bar,
                                    time_idx,
                                    sil_thres=config["tracking"]["sil_thres"],
                                    tracking=True,
                                )
                        else:
                            progress_bar.update(1)
                    # update the runtime numbers
                    # check if we should stop tracking
                    tracking_iter += 1
                    if tracking_iter == num_iters_tracking:
                        if (
                            losses["depth"] < config["tracking"]["depth_loss_thres"]
                            and config["tracking"]["use_depth_for_loss"]
                        ):
                            break
                        elif (
                            config["tracking"]["use_depth_for_loss"]
                            and not do_continue_slam
                        ):
                            do_continue_slam = True
                            progress_bar = tqdm(
                                range(num_iters_tracking),
                                desc=f"tracking time step: {time_idx}",
                            )
                            num_iters_tracking = 2 * num_iters_tracking
                            if config["use_wandb"]:
                                wandb_run.log(
                                    {
                                        "tracking/extra tracking iters frames": time_idx,
                                        "tracking/step": wandb_time_step,
                                    }
                                )
                        else:
                            break

                progress_bar.close()
                # Copy over the best candidate rotation & translation
                if (losses["depth"] < config["tracking"]["depth_loss_thres"] and losses["edge"] < config["tracking"]["edge_loss_thres"]):
                    print(f"INFO: Camera Pose of frame {time_idx} Updated, new cam unnorm rots: {candidate_cam_unnorm_rot.detach().cpu()}, new cam trans: {candidate_cam_tran.detach().cpu()}")
                    with torch.no_grad():
                        params["cam_unnorm_rots"][..., time_idx] = candidate_cam_unnorm_rot
                        params["cam_trans"][..., time_idx] = candidate_cam_tran

                # Update the runtime numbers
                tracking_end_time = time.time()
                tracking_iter_time =  tracking_end_time - tracking_start_time
               
                print(f"INFO: Frame id: {time_idx} Tracking Ends, it takes {tracking_iter_time: .6f}s for {num_iters_tracking} iterations. \n \
                        \tTracking loss: ")

            if (
                time_idx == 0
                or (time_idx + 1) % config["report_global_progress_every"] == 0
            ):
                try:
                    # Report Final Tracking Progress
                    progress_bar = tqdm(
                        range(1), desc=f"Tracking Result Time Step: {time_idx}"
                    )
                    with torch.no_grad():
                        if config["use_wandb"]:
                            report_progress(
                                params,
                                tracking_curr_data,
                                1,
                                progress_bar,
                                iter_time_idx,
                                sil_thres=config["tracking"]["sil_thres"],
                                tracking=True,
                                wandb_run=wandb_run,
                                wandb_step=wandb_time_step,
                                wandb_save_qual=config["wandb"]["save_qual"],
                                global_logging=True,
                            )
                        else:
                            report_progress(
                                params,
                                tracking_curr_data,
                                1,
                                progress_bar,
                                iter_time_idx,
                                sil_thres=config["tracking"]["sil_thres"],
                                tracking=True,
                            )
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(
                        config["workdir"], config["run_name"]
                    )
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print("Failed to evaluate trajectory.")

            # Densification & KeyFrame-based Mapping
            print(f"INFO: Number of Gaussians before adding new gaussians: {params['means3D'].shape[0]}")
            print(f"INFO: lof_opacities: {torch.mean(params['logit_opacities'])}, max of opacity: {torch.max(params['logit_opacities'])}, min of opacity: {torch.min(params['logit_opacities'])}")
            if time_idx == 0 or (time_idx + 1) % config["map_every"] == 0:
                # Densification
                if config["mapping"]["add_new_gaussians"] and time_idx > 0:
                    # Add new Gaussians to the scene based on the Silhouette
                    # TODO add masked gaussian
                    self.add_new_gaussians(
                        params,
                        variables,
                        curr_data,
                        config["mapping"]["sil_thres"],
                        config["mapping"]["depth_thres"],
                        time_idx,
                        config["mean_sq_dist_method"],
                        config["gaussian_distribution"],
                    )
                    post_num_pts = params["means3D"].shape[0]
                    if config["use_wandb"]:
                        wandb_run.log(
                            {
                                "Mapping/Number of Gaussians": post_num_pts,
                                "Mapping/step": wandb_time_step,
                            }
                        )

                print(f"INFO: Number of Gaussians after adding new gaussians: {params['means3D'].shape[0]}")
                print(f"INFO: lof_opacities: {torch.mean(params['logit_opacities'])}, max of opacity: {torch.max(params['logit_opacities'])}, min of opacity: {torch.min(params['logit_opacities'])}")
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(
                        params["cam_unnorm_rots"][..., time_idx].detach()
                    )
                    curr_cam_tran = params["cam_trans"][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Select Keyframes for Mapping
                    num_keyframes = config["mapping_window_size"] - 2

                    selected_keyframes = keyframe_selection_overlap(
                        depth, curr_w2c, intrinsics, self.keyframe_list[:-1], num_keyframes
                    )
                    selected_time_idx = [
                        self.keyframe_list[frame_idx]["id"]
                        for frame_idx in selected_keyframes
                    ]
                    if len(self.keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(self.keyframe_list[-1]["id"])
                        selected_keyframes.append(len(self.keyframe_list) - 1)
                    # Add current frame to the selected keyframes
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # Print the selected keyframes
                    print(
                        f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}"
                    )

                # Reset Optimizer & Learning Rates for Full Map Optimization
                optimizer = initialize_optimizer_sep(
                    params, config["mapping"]["lrs"], tracking=False
                )

                # Mapping
                mapping_start_time = time.time()
                if config['mapping']['num_iters'] > 0:
                    progress_bar = tqdm(
                        range(config['mapping']['num_iters']), desc=f"Mapping Time Step: {time_idx}"
                    )
                for mapping_iter in range(config['mapping']['num_iters']):
                    iter_start_time = time.time()
                    # Randomly select a frame until current time step amongst keyframes
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_mask = mask
                    else:
                        # Use Keyframe Data
                        iter_time_idx = self.keyframe_list[selected_rand_keyframe_idx]["id"]
                        iter_color = self.keyframe_list[selected_rand_keyframe_idx]["color"]
                        iter_depth = self.keyframe_list[selected_rand_keyframe_idx]["depth"]
                        iter_mask = self.keyframe_list[selected_rand_keyframe_idx]["mask"]
                    #TODO should updated after tracking
                    iter_gt_w2c = self.gt_w2c_all_frames[: iter_time_idx + 1]
                    iter_data = {
                        "cam": self.cam,
                        "im": iter_color,
                        "depth": iter_depth,
                        "mask": iter_mask,
                        "id": iter_time_idx,
                        "intrinsics": intrinsics,
                        "w2c": self.first_frame_w2c,
                        "iter_gt_w2c_list": iter_gt_w2c,
                    }
                    # Loss for current frame
                    loss, variables, losses = self.get_loss(
                        params,
                        iter_data,
                        variables,
                        iter_time_idx,
                        config["mapping"]["loss_weights"],
                        config["mapping"]["use_sil_for_loss"],
                        config["mapping"]["sil_thres"],
                        config["mapping"]["use_l1"],
                        config["mapping"]["ignore_outlier_depth_loss"],
                        tracking=False,
                        do_ba=False,
                        training_iter=mapping_iter
                    )
                    if config["use_wandb"]:
                        # Report Loss
                        wandb_mapping_step = report_loss(
                            losses, wandb_run, wandb_mapping_step, mapping=True
                        )
                    # Backprop
                    loss.backward()
                    print(f"INFO: Number of Gaussians after mapping: {params['means3D'].shape[0]} at mapping iteration {mapping_iter}")
                    print(f"INFO: lof_opacities: {torch.mean(params['logit_opacities'])}, max of opacity: {torch.max(params['logit_opacities'])}, min of opacity: {torch.min(params['logit_opacities'])}")

                    with torch.no_grad():
                        # Prune Gaussians
                        pcNum1 = pcNum2 = pcNum3 = None
                        if config["mapping"]["prune_gaussians"]:
                            pcNum1 = int(params["means3D"].shape[0])
                            # visualize_param_info(params)
                            params, variables = prune_gaussians(
                                params,
                                variables,
                                optimizer,
                                mapping_iter,
                                config["mapping"]["pruning_dict"],
                            )
                            pcNum2 = int(params["means3D"].shape[0])
                            # visualize_param_info(params)
                            if config["use_wandb"]:
                                wandb_run.log(
                                    {
                                        "Mapping/Number of Gaussians - Pruning": params[
                                            "means3D"
                                        ].shape[0],
                                        "Mapping/step": wandb_mapping_step,
                                    }
                                )
                        # Gaussian-Splatting's Gradient-based Densification
                        if config["mapping"]["use_gaussian_splatting_densification"]:
                            params, variables = densify(
                                params,
                                variables,
                                optimizer,
                                mapping_iter,
                                config["mapping"]["densify_dict"],
                            )
                            pcNum3 = int(params["means3D"].shape[0])
                            # visualize_param_info(params)
                            if config["use_wandb"]:
                                wandb_run.log(
                                    {
                                        "Mapping/Number of Gaussians - Densification": params[
                                            "means3D"
                                        ].shape[
                                            0
                                        ],
                                        "Mapping/step": wandb_mapping_step,
                                    }
                                )
                        # print(f"/////////////PRUNE AND DENSIFY POINTCLOUD is DONE. origial numbers of points: {pcNum1}, pruned pcl: {pcNum2}, densiftied pcl: {pcNum3} " )
                        # Optimizer Update
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        # Report Progress
                        if config["report_iter_progress"]:
                            if config["use_wandb"]:
                                report_progress(
                                    params,
                                    iter_data,
                                    mapping_iter + 1,
                                    progress_bar,
                                    iter_time_idx,
                                    sil_thres=config["mapping"]["sil_thres"],
                                    wandb_run=wandb_run,
                                    wandb_step=wandb_mapping_step,
                                    wandb_save_qual=config["wandb"]["save_qual"],
                                    mapping=True,
                                    online_time_idx=time_idx,
                                )
                            else:
                                report_progress(
                                    params,
                                    iter_data,
                                    mapping_iter + 1,
                                    progress_bar,
                                    iter_time_idx,
                                    sil_thres=config["mapping"]["sil_thres"],
                                    mapping=True,
                                    online_time_idx=time_idx,
                                )
                        else:
                            progress_bar.update(1)
                    # Update the runtime numbers
                    iter_end_time = time.time()
                    self.mapping_iter_time_sum += iter_end_time - iter_start_time
                    self.mapping_iter_time_count += 1

                # open3d vis pointcloud
                # if time_idx % 5 == 0:
                #     visualize_param_info(params)

                if config['mapping']['num_iters'] > 0:
                    progress_bar.close()
                # Update the runtime numbers
                mapping_end_time = time.time()
                self.mapping_frame_time_sum += mapping_end_time - mapping_start_time
                self.mapping_frame_time_count += 1

                if (
                    time_idx == 0
                    or (time_idx + 1) % config["report_global_progress_every"] == 0
                ):
                    try:
                        # Report Mapping Progress
                        progress_bar = tqdm(
                            range(1), desc=f"Mapping Result Time Step: {time_idx}"
                        )
                        with torch.no_grad():
                            if config["use_wandb"]:
                                report_progress(
                                    params,
                                    curr_data,
                                    1,
                                    progress_bar,
                                    time_idx,
                                    sil_thres=config["mapping"]["sil_thres"],
                                    wandb_run=wandb_run,
                                    wandb_step=wandb_time_step,
                                    wandb_save_qual=config["wandb"]["save_qual"],
                                    mapping=True,
                                    online_time_idx=time_idx,
                                    global_logging=True,
                                )
                            else:
                                report_progress(
                                    params,
                                    curr_data,
                                    1,
                                    progress_bar,
                                    time_idx,
                                    sil_thres=config["mapping"]["sil_thres"],
                                    mapping=True,
                                    online_time_idx=time_idx,
                                )
                        progress_bar.close()
                    except:
                        ckpt_output_dir = os.path.join(
                            config["workdir"], config["run_name"]
                        )
                        save_params_ckpt(params, ckpt_output_dir, time_idx)
                        print("Failed to evaluate trajectory.")

            # Add frame to keyframe list
            if (
                (
                    (time_idx == 0)
                    or ((time_idx + 1) % config["keyframe_every"] == 0)
                    or (time_idx == self._total_num_frames - 2)
                )
                and (not torch.isinf(curr_gt_w2c[-1]).any())
                and (not torch.isnan(curr_gt_w2c[-1]).any())
            ):
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(
                        params["cam_unnorm_rots"][..., time_idx].detach()
                    )
                    curr_cam_tran = params["cam_trans"][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {
                        "id": time_idx,
                        "est_w2c": curr_w2c,
                        "color": color,
                        "depth": depth,
                        "mask": mask,
                    }
                    # Add to keyframe list
                    self.keyframe_list.append(curr_keyframe)
                    self.keyframe_time_indices.append(time_idx)

            # Checkpoint every iteration
            if (
                time_idx % config["checkpoint_interval"] == 0
                and config["save_checkpoints"]
            ):
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                np.save(
                    os.path.join(
                        ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"
                    ),
                    np.array(keyframe_time_indices),
                )

            # Increment WandB Time Step
            if config["use_wandb"]:
                wandb_time_step += 1

            torch.cuda.empty_cache()

            #TODO logging computation time
