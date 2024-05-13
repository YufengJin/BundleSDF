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
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_ply, params2cpu
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

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

# TODO 1. instead of updating camera pose and mapping iteratively, update camera pose and mapping at the same time
# TODO 2. add keyframe selection
# TODO 3. evaluate the error of camera pose optimization on gt_pose
VIS_LOSS_IMAGE = True 

def run_gui_thread(gui_lock, gui_dict):
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

    vis.destroy_window()

def evaluate_batch_pose_error(poses_gt, poses_est):
    num_poses = poses_est.shape[0]
    
    translation_errors = np.zeros(num_poses)
    rotation_errors = np.zeros(num_poses)
    
    for i in range(num_poses):
        pose_gt = poses_gt[i]
        pose_est = poses_est[i]
        
        # Extract translation vectors
        translation_gt = pose_gt[:3, 3]
        translation_est = pose_est[:3, 3]
        
        # Extract rotation matrices
        rotation_gt = pose_gt[:3, :3]
        rotation_est = pose_est[:3, :3]
        
        # Calculate translation error
        translation_error = np.linalg.norm(translation_gt - translation_est)
        
        # Calculate rotation error
        rotation_error_cos = 0.5 * (np.trace(np.dot(rotation_gt.T, rotation_est)) - 1.0)
        rotation_error_cos = min(1.0, max(-1.0, rotation_error_cos))  # Ensure value is in valid range for arccos
        rotation_error_rad = np.arccos(rotation_error_cos)
        rotation_error_deg = np.degrees(rotation_error_rad)
        
        translation_errors[i] = translation_error
        rotation_errors[i] = rotation_error_deg
    

    if True:
        print(f"DEBUG: Pose Estimation Evlaution: \nTranslation Error:\n \
              \tFull Trans Error: {translation_errors} \n \
              \tMean: {np.mean(translation_errors):.4f}, Variance: {np.var(translation_errors):.4f}  \n \
              Rotation Error:\n \
              \tFull Rot Error: {rotation_errors} \n \
              \tMean: {np.mean(rotation_errors):.4f}, Variance: {np.var(rotation_errors):.4f}") 

    return translation_errors, rotation_errors

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
        # new_timestep = (
        #     time_idx * torch.ones(new_pt_cld.shape[0], device="cuda").float()
        # )
        # variables["timestep"] = torch.cat(
        #     (variables["timestep"], new_timestep), dim=0
        # )

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
    elif gaussian_distribution == "anisotropic_2d":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        # add zeros for 3rd dimension
        log_scales = torch.cat((log_scales, log_scales, -1e5 * torch.ones_like(log_scales)), dim=1)
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
        self, cfg_gs, rgbs, depths, masks, K, poses, total_num_frames, dtype=torch.float, offset=None, scale=None, pointcloud=None, poses_gt=None, wandb_run=None, run_gui=False):
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

        self.run_gui = run_gui

        if wandb_run is not None:
            self.wandb_run = wandb_run
            self.use_wandb = True

        if self.run_gui:
            import threading
            self.gui_lock = threading.Lock()
            self.gui_dict = {"join": False}
            gui_runner = threading.Thread(target=run_gui_thread, args=(self.gui_lock, self.gui_dict))
            gui_runner.start()

        #from opengl to opencv
        #TODO  remove pose preprocess outside
        poses[:, :3, 1:3] = -poses[:, :3, 1:3]
        self._fisrt_c2w = poses[0].copy()
        poses = self._preprocess_poses(poses)

        # convert camera pose from opengl to opencv
        poses_gt[:, :3, 1:3] = -poses_gt[:, :3, 1:3]
        self._poses_gt = poses_gt

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
        #self.gt_w2c_all_frames = []
        self.curr_frame_id = 0

        # only +1 when optimizer is called
        self.gaussians_iter = 0
            
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
            #self.gt_w2c_all_frames.append(w2c)
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
                    #"iter_gt_w2c_list": self.gt_w2c_all_frames.copy(),
                    "seen": False,
                    "optimized": False
                }
            )
            self.curr_frame_id += 1

    def evaluate_poses(self, visualize=False):
        if self._poses_gt is None:
            return None
        poses_gt = self._poses_gt.copy()

        poses = self.get_optimized_cam_poses()

        assert isinstance(poses_gt, np.ndarray) and poses_gt.shape[1:] == (4, 4)
        assert isinstance(poses, np.ndarray) and poses.shape[1:] == (4, 4)

        # calculate the relative pose error
        translation_errors, rotation_errors = evaluate_batch_pose_error(poses_gt, poses)

        if visualize:
            num_poses = poses.shape[0]
            poses_gt = poses_gt[:num_poses,...]
            assert poses_gt.shape == poses.shape

            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
            
            poses_gt_frames = o3d.geometry.TriangleMesh()
            poses_est_frames = o3d.geometry.TriangleMesh()
            for pose_gt, pose_est in zip(poses_gt, poses):
                cam_frame_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                cam_frame_gt.transform(pose_gt)
                poses_gt_frames += cam_frame_gt

                cam_frame_est = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                cam_frame_est.transform(pose_est)
                poses_est_frames += cam_frame_est

            est_color = np.asarray(poses_est_frames.vertex_colors)
            poses_est_frames.vertex_colors = o3d.utility.Vector3dVector(est_color*0.5)

            # convert to PointCloud
            pcd = o3d.geometry.PointCloud()

            pcd += poses_gt_frames.sample_points_uniformly(number_of_points=10000)
            pcd += poses_est_frames.sample_points_uniformly(number_of_points=10000)
            pcd += world_frame.sample_points_uniformly(number_of_points=10000)

            # pointcloud to numpy
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            pcl = np.concatenate([points, colors], axis=1)

            return translation_errors, rotation_errors, pcl

        return translation_errors, rotation_errors
    
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
        # from opengl to opencv 
        gl_poses = poses.copy()
        #gl_poses[:, :3, 1:3] = -gl_poses[:, :3, 1:3]
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
        mask = (depth > 0) & mask.bool()  # Mask out invalid depth values
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

        center_pcl = torch.mean(init_pt_cld[:, :3], dim=0)
        radius = torch.norm(init_pt_cld[:, :3] - center_pcl, dim=1).max()
        # Initialize an estimate of scene radius for Gaussian-Splatting Densification, TODO understanding variables scene_radius
        variables["scene_radius"] = 1.5 * radius 

        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = w2c  # relative w2c np.eye(4)
        self.cam = cam

    def _initialize_params(
        self, init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution
    ):
        # TODO mean3_sq_dist can be caluclated from octree, reference: https://github.com/JonathonLuiten/Dynamic3DGaussians/blob/main/train.py:41
        num_pts = init_pt_cld.shape[0]
        means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]
        unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 4]
        logit_opacities = np.zeros((num_pts, 1))

        if isinstance(mean3_sq_dist, torch.Tensor):
            mean3_sq_dist = mean3_sq_dist.cpu().numpy()

        if gaussian_distribution == "isotropic":
            log_scales = np.tile(
                np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 1)
            )
        elif gaussian_distribution == "anisotropic":
            log_scales = np.tile(
                np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)
            )
        elif gaussian_distribution == "anisotropic_2d":
            log_scales = np.tile(
                np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 1)
            )
            # add zeros for 3rd dimension
            log_scales = np.concatenate((log_scales, log_scales, -1e5 * np.ones_like(log_scales)), axis=1)
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

        params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items()}

        variables = {
            "max_2D_radius": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "means2D_gradient_accum": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            "denom": torch.zeros(params["means3D"].shape[0]).cuda().float(),
            #"timestep": torch.zeros(params["means3D"].shape[0]).cuda().float(),
        }

        return params, variables

    def add_new_frames(self, rgbs, depths, masks, poses):
        # TODO need to redesign the add function
        # new_pcd=pcd_normalized,)

        colors, depths, masks = self._preprocess_images_data(rgbs, depths, masks)

        #ForkedPdb().set_trace()
        # get latest poss
        poses[:, :3, 1:3] = -poses[:, :3, 1:3]

        # from opengl to opencv
        poses = self._preprocess_poses(poses)[-1, ...].unsqueeze(0)

        # prepare data from training
        for color, depth, mask, pose in zip(colors, depths, masks, poses):
            w2c = torch.linalg.inv(pose)
            color = color.permute(2, 0, 1)  # /255.
            depth = depth.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)
            #self.gt_w2c_all_frames.append(w2c)
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
                    #"iter_gt_w2c_list": self.gt_w2c_all_frames.copy(),      # only for trajectory prediction
                    "seen": False,
                    "optimized": False
                }
            )
            self.curr_frame_id += 1

    def initialize_optimizer(self, lr_dict, epoch):
        lrs = lr_dict.copy()
        if epoch > 0:
            lrs['means3D'] = 0.0
            lrs['rgb_colors'] = 0.0
            lrs['unnorm_rotations'] = 0.0
            lrs['logit_opacities'] = 0.0
            lrs['log_scales'] = 0.0

        param_groups = [
            {"params": [v], "name": k, "lr": lrs[k]} for k, v in self.params.items()
        ]
        return torch.optim.Adam(param_groups)

    def get_xyz_rgb_params(self):
        points = self.params["means3D"].detach().cpu().numpy()
        colors = self.params["rgb_colors"].detach().cpu().numpy()

        # alleviate the first frame error
        # with torch.no_grad():
        #     cam_rot = F.normalize(
        #         self.params["cam_unnorm_rots"][..., 0].detach()
        #         )
            
        #     cam_tran = self.params["cam_trans"][..., 0].detach()
        #     first_w2c_err = torch.eye(4).cuda().float()
        #     first_w2c_err[:3, :3] = build_rotation(cam_rot)
        #     first_w2c_err[:3, 3] = cam_tran
        #     system_err = torch.linalg.inv(first_w2c_err)

        # # Transform the points to the world frame
        # transform = system_err.cpu().numpy()
        # points = points @ transform[:3, :3].T + transform[:3, 3]
        #center = points.mean(axis=0)
        #points = points - center
        # Create a point cloud object
        return np.concatenate((points, colors), axis=1)

    def get_optimized_cam_poses(self):
        opt_cam_poses = []
        with torch.no_grad():
            # get the first frame pose
            cam_rot0 = F.normalize(
                    self.params["cam_unnorm_rots"][..., 0].detach()
                )
            cam_tran0 = self.params["cam_trans"][..., 0].detach()
            w2c0 = torch.eye(4).cuda().float()
            w2c0[:3, :3] = build_rotation(cam_rot0)
            w2c0[:3, 3] = cam_tran0
            w2c0 = w2c0.cpu().numpy()

            # Get the current estimated rotation & translation
            for time_idx in range(self.curr_frame_id):
                cam_rot = F.normalize(
                    self.params["cam_unnorm_rots"][..., time_idx].detach()
                )
                cam_tran = self.params["cam_trans"][..., time_idx].detach()
                w2c = torch.eye(4).cuda().float()
                w2c[:3, :3] = build_rotation(cam_rot)
                w2c[:3, 3] = cam_tran

                #w2c = torch.linalg.inv(w2c0) @ w2c
                w2c = torch.linalg.inv(w2c)

                opt_cam_poses.append(self._fisrt_c2w @ w2c.cpu().numpy())
                
        return np.asarray(opt_cam_poses)

    def train(self):
        # TODO transform means3D regarding first frame error
        batch_size = self.cfg_gs['train']["batch_size"]
        lr_dict = self.cfg_gs['train']["lrs"]
        batch_iters = self.cfg_gs['train']["batch_iters"]
        progress_bar = tqdm(total=self.cfg_gs['train']["num_epochs"])

        # initialize camera poses
        for curr_data in self.queued_data_for_train:
            with torch.no_grad():
                time_idx = curr_data["id"]
                rel_w2c = curr_data["w2c"]
                if time_idx > 0:
                    # update initial pose relative to frame 0
                    rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                    rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                    rel_w2c_tran = rel_w2c[:3, 3].detach()
                    # Update the camera parameters
                    self.params["cam_unnorm_rots"][..., time_idx] = rel_w2c_rot_quat

        # intialize optimizer
        for epoch in range(self.cfg_gs['train']["num_epochs"]):
            # TODO select keyframes + latest frames + first frame
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
                    # print("//////////////////////DEBUG//////////////////////")
                    # print(f"w2c of frame {time_idx}: {rel_w2c}")
                    if time_idx > 0:
                        # update initial pose relative to frame 0
                        rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                        rel_w2c_tran = rel_w2c[:3, 3].detach()
                        # Update the camera parameters
                        self.params["cam_unnorm_rots"][..., time_idx] = rel_w2c_rot_quat
                        self.params["cam_trans"][..., time_idx] = rel_w2c_tran

                        if self.cfg_gs['add_new_gaussians']:
                        #if not curr_data["seen"] and self.cfg_gs['add_new_gaussians']:
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

            optimizer = self.initialize_optimizer(lr_dict, epoch)
            for _ in range(batch_iters):
                # TODO not working add new gaussians must before optimizer
                # # add new gaussians every 100 iterations
                # with torch.no_grad():
                #     if self.cfg_gs['add_new_gaussians'] and self.gaussians_iter % self.cfg_gs['add_gaussian_dict']['every_iter'] == 0:
                #         for curr_data in batch_data:
                #             print(f"INFO: Adding new gaussians for frame {time_idx}")
                #             pcl_num = self.params["means3D"].shape[0]
                #             # add new gaussians
                #             add_new_gaussians(
                #                 self.params,
                #                 self.variables,
                #                 curr_data,
                #                 self.cfg_gs['add_gaussian_dict']['sil_thres'],
                #                 self.cfg_gs['add_gaussian_dict']['depth_thres'],
                #                 time_idx,
                #                 self.cfg_gs["mean_sq_dist_method"],
                #                 self.cfg_gs["gaussian_distribution"],
                #             )         
                #             curr_data['seen'] = True
                #             print(f"INFO: Adding new gaussians done, the number of gaussians added: {self.params['means3D'].shape[0] - pcl_num}")

                loss, losses = self.train_once(batch_data, self.gaussians_iter)
                loss /= len(batch_data)

                # minimize systematic error for fisrt frame
                # offset_loss = torch.abs(self.params["cam_unnorm_rots"][...,0] - torch.tensor([1, 0, 0, 0]).to(self.device)).sum()
                # offset_loss += torch.abs(self.params["cam_trans"][...,0] - torch.tensor([0, 0, 0]).to(self.device)).sum()

                # loss += 1e5 * offset_loss
                # losses["offset_loss"] = 1e5 * offset_loss

                loss.backward()

                
                if True:
                    # print the loss and gradients
                    msg = f"DEBUG: Epoch: {epoch}/{self.cfg_gs['train']['num_epochs']}, Iteration: {iter}/{batch_iters}\n"
                    losses_msg = f"\tTotal Loss: {loss.item()}\n"
                    for k, v in losses.items():
                        losses_msg += f"\t\t{k} Loss: {v.item()}\n"

                    msg += losses_msg

                    grad_msg = f"\tGradients:\n"
                    for k, v in self.params.items():
                        grad_msg += f"\t{k} grad(mean, max, min): {v.grad.mean().item()}, {v.grad.max().item()}, {v.grad.min().item()}\n"
                
                    msg += grad_msg
                    print(msg)
                
                # set the gradients to zero, TODO add system error to all frame
                # self.params["cam_unnorm_rots"].grad[:,:,0] = torch.tensor(0.0).to(self.device) 
                # self.params["cam_trans"].grad[:,:,0]= torch.tensor(0.0).to(self.device) 
                  
                # the 3 dimensions of the log_scales are the same
                if self.cfg_gs['gaussian_distribution'] == "aniostropic_2d":
                    self.params["log_scales"].grad[...,2] = torch.tensor(0.0).to(self.device) 

                optimizer.step()
                # TODO save keyframes depending on the losses

                with torch.no_grad():
                    #Prune Gaussians
                    # TODO think prune iteration self.gaussians_iter or batch iter?
                    if self.cfg_gs['train']["prune_gaussians"]:
                        print(f"INFO: number of gaussians before pruning: {self.params['means3D'].shape[0]}")
                        pcl_num = self.params["means3D"].shape[0]
                        self.params, self.variables = prune_gaussians(
                            self.params,
                            self.variables,
                            optimizer,
                            self.gaussians_iter,
                            self.cfg_gs['train']["pruning_dict"],
                        )
                        print(f"INFO: Gaussian Pruning Done. the number of gaussians pruned: {pcl_num - self.params['means3D'].shape[0]}, the remaining number of gaussians: {self.params['means3D'].shape[0]}")
                        msg = f"\tMore Info:\n"
                        msg += f"\t\tLog Scales(mean, max, min): {self.params['log_scales'].mean().item()}, {self.params['log_scales'].max().item()}, {self.params['log_scales'].min().item()}\n"
                        msg += f"\t\tLog Opacities(mean, max, min): {self.params['logit_opacities'].mean().item()}, {self.params['logit_opacities'].max().item()}, {self.params['logit_opacities'].min().item()}\n" 
                        print(msg)

                #     # Gaussian-Splatting's Gradient-based Densification
                #     if self.cfg_gs['train']["use_gaussian_splatting_densification"]: 
                #         pcl_num = self.params["means3D"].shape[0]
                #         self.params, self.variables = densify(
                #             self.params,
                #             self.variables,
                #             optimizer,
                #             self.gaussians_iter,
                #             self.cfg_gs['train']["densify_dict"],
                #         )
                #         print(f"INFO: Training Iteration: {self.gaussians_iter} Gaussian-Splatting Densification Done. the number of gaussians added: {self.params['means3D'].shape[0] - pcl_num}")
                #         #visualize_param_info(self.params)

                optimizer.zero_grad(set_to_none=True)

                self.gaussians_iter += 1

                trans_errs, rot_errs, cam_pcls = self.evaluate_poses(visualize=True)
                #trans_errs, rot_errs = self.evaluate_poses()

                if self.run_gui:
                    with self.gui_lock:
                        obj_pcl = self.get_xyz_rgb_params()
                        try:
                            pcd = np.concatenate((obj_pcl, cam_pcls), axis=0)

                        except NameError as e:
                            pcd = obj_pcl

                        self.gui_dict["pointcloud"] = pcd

                # log all losses
                if self.use_wandb:
                    for idx, (trans_err, rot_err) in enumerate(zip(trans_errs, rot_errs)):
                        wandb.log({f"trans_err_frame_{idx}": trans_err})
                        wandb.log({f"rot_err_frame_{idx}": rot_err})

                    wandb.log({"mean_trans_err":np.mean(trans_errs)})
                    wandb.log({"mean_rot_err":np.mean(rot_errs)})
                    wandb.log(losses)
                    wandb.log({"num_pts": self.params["means3D"].shape[0]})

            # update translation and rotation of the camera
            with torch.no_grad():
                # get transformation of the first frame
                cam_rot0 = F.normalize(
                    self.params["cam_unnorm_rots"][..., 0].detach()
                    )
                cam_trans0 = self.params["cam_trans"][..., 0].detach()
                rel_w2c0 = torch.eye(4).cuda().float()
                rel_w2c0[:3, :3] = build_rotation(cam_rot0)
                rel_w2c0[:3, 3] = cam_trans0

                for curr_data in batch_data:
                    time_idx = curr_data["id"]
                    if time_idx > 0:
                        cam_rot = F.normalize(
                            self.params["cam_unnorm_rots"][..., time_idx].detach()
                            )
                    
                        cam_tran = self.params["cam_trans"][..., time_idx].detach()

                        # update the camera pose, relative to the first frame
                        # TODO remove redundant code
                        self.params["cam_unnorm_rots"][..., time_idx] = cam_rot

                        w2c = torch.eye(4).cuda().float()
                        w2c[:3, :3] = build_rotation(cam_rot)
                        w2c[:3, 3] = cam_tran
                        curr_data["w2c"] = torch.linalg.inv(rel_w2c0) @ w2c

                # # # update gaussian parameters
                if self.params['log_scales'].shape[1] == 1:
                    transform_rots = False # Isotropic Gaussians
                else:
                    transform_rots = True # Anisotropic Gaussians
               
                # Get Centers and Unnorm Rots of Gaussians in World Frame 
                pts = self.params['means3D'].detach()
                unnorm_rots = self.params['unnorm_rotations'].detach()

                transformed_gaussians = {}
                # Transform Centers of Gaussians to Camera Frame
                pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
                pts4 = torch.cat((pts, pts_ones), dim=1)
                transformed_pts = (rel_w2c0 @ pts4.T).T[:, :3]
                self.params["means3D"] = transformed_pts

                if transform_rots:
                    norm_rots = F.normalize(unnorm_rots)
                    transformed_rots = quat_mult(cam_rot0, norm_rots)
                    self.params['unnorm_rotations'] = transformed_rots
                else:
                    self.params['unnorm_rotations'] = unnorm_rots
                
                self.params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in self.params.items()}
                torch.cuda.empty_cache()


            progress_bar.update(1)

        # save ply file
        if False:
            params = params2cpu(self.params)
            means = params['means3D']
            scales = params['log_scales']
            rotations = params['unnorm_rotations']
            rgbs = params['rgb_colors']
            opacities = params['logit_opacities']

            ply_path = os.path.join(work_path, run_name, "splat.ply")


    def train_once(self, batch_data, iter, dssim_weight=0.2, gaussians_grad=True, camera_grad=True):
        # TODO save each loss for frame
        losses = {k: torch.tensor(0.0).to(self.device) for k in ["edge", "depth", "silhouette", "im"]}

        # loss weights
        loss_weights = self.cfg_gs["train"]["loss_weights"]

        for curr_data in batch_data:
            iter_time_idx = curr_data["id"]
            
            # transform the gaussians to the current frame
            transformed_gaussians = transform_to_frame(
                self.params, iter_time_idx, gaussians_grad=gaussians_grad, camera_grad=camera_grad
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

            # TODO if uncertainty is necessary? depth_sq is the square of the depth
            depth_sq = depth_sil[2, :, :].unsqueeze(0)
            uncertainty = depth_sq - depth**2
            uncertainty = uncertainty.detach()
            # M    ask with valid depth values (accounts for outlier depth values)
            nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))

            mask_gt = curr_data["mask"].bool()
            gt_im = curr_data["im"] 

            mask = mask_gt & presence_sil_mask & nan_mask

            # canny edge detection
            gt_gray = ki.color.rgb_to_grayscale(curr_data["im"].unsqueeze(0))

            # sobel edges
            gt_edge = ki.filters.sobel(gt_gray).squeeze(0)
            # canny edges
            # gt_edge= ki.filters.canny(gt_gray)[0]

            gray = ki.color.rgb_to_grayscale(im.unsqueeze(0))

            # sobel edges
            edge = ki.filters.sobel(gray).squeeze(0)
            
            # edge loss
            edge_loss = torch.abs(gt_edge - edge)[mask].sum()
            losses["edge"] += edge_loss

            # Depth loss
            depth_loss = torch.abs(curr_data["depth"] - depth)[mask].sum()
            losses["depth"] += depth_loss

            # silhouette loss
            losses["silhouette"] += torch.abs(silhouette.float() - curr_data["mask"]).sum()

            # color loss
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()

            rgbl1 = torch.abs(gt_im - im)[color_mask].sum()
            im_loss = (1-dssim_weight) * rgbl1 + dssim_weight * (1.0 - calc_ssim(im, gt_im))
            losses["im"] += im_loss

            # TODO track pose loss
            if curr_data.get("track_losses") is None:
                curr_data['track_losses'] = [(loss_weights['im'] * im_loss + loss_weights['depth'] * depth_loss + loss_weights['edge'] * edge_loss).item()]
            else:
                curr_data['track_losses'].append((loss_weights['im'] * im_loss + loss_weights['depth'] * depth_loss + loss_weights['edge'] * edge_loss).item())

            if self.use_wandb:
                frame_id = curr_data["id"]
                curr_losses = {f'im_{frame_id}': losses['im'], f'depth_{frame_id}': losses['depth'], f'edge_{frame_id}': losses['edge'], f'silhouette_{frame_id}': losses['silhouette']}
                curr_losses[f'total_{frame_id}'] = curr_data['track_losses'][-1]
                wandb.log(curr_losses)
            # visualize debugging images

            if VIS_LOSS_IMAGE and (iter % 100 == 0):
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
                ax_im = ax[1, 1].imshow(
                    weighted_render_depth[0].detach().cpu())
                cbar = fig.colorbar(ax_im, ax=ax[1, 1])
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
                suptitle = f"frame_id: {curr_data['id']} Training Iteration: {iter}"
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
                
            # TODO not update every time, because optimizer is updated once per batch, ?????????
            seen = radius > 0
            self.variables["max_2D_radius"][seen] = torch.max(
                radius[seen], self.variables["max_2D_radius"][seen]
            )
            self.variables["seen"] = seen

        weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}

        loss = sum(weighted_losses.values())

        # regularization of first frame 0
        # cam_rot0 = F.normalize(
        #             self.params["cam_unnorm_rots"][..., 0]
        #             )
        # cam_trans0 = self.params["cam_trans"][..., 0]
        # loss += 1e10 * torch.abs(cam_rot0 - torch.tensor([1, 0, 0, 0]).to(self.device)).sum() 
        # loss += 1e10 * torch.abs(cam_trans0 - torch.tensor([0, 0, 0]).to(self.device)).sum()

        weighted_losses['loss'] = loss
        return loss, weighted_losses
