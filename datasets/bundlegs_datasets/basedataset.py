"""
PyTorch dataset classes for Object 6D Pose Tracking v1.0.

Author: Yufeng Jin 
Date: April 19, 2024

A few parts of this code are adapted from NICE-SLAM
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py
"""

import abc
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import yaml
from natsort import natsorted

from .geometryutils import relative_transformation
from . import datautils

def convert_ho3d_rgb2depth(depth_img):
    depth_scale = 0.00012498664727900177
    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale
    return dpt.astype(np.float32)

def to_scalar(inp: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """
    Convert the input to a scalar
    """
    if isinstance(inp, float):
        return inp

    if isinstance(inp, np.ndarray):
        assert inp.size == 1
        return inp.item()

    if isinstance(inp, torch.Tensor):
        assert inp.numel() == 1
        return inp.item()


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def from_intrinsics_matrix(K):
    """
    Get fx, fy, cx, cy from the intrinsics matrix

    return 4 scalars
    """
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


class GradSLAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        relative_pose: bool = True,  # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width

        self.relative_pose = relative_pose

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError("end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start))

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"] if "crop_size" in config_dict["camera_params"] else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]

        if self.name in ['bop']:
            try:
                self.poses = self.load_poses()
                posesNotFound = False
            except FileNotFoundError as e:
                raise FileNotFoundError("ERROR: scene_gt.json does not exist in BOP dataset") from e

        paths = self.get_filepaths()
        if len(paths) == 2:
            self.color_paths, self.depth_paths = paths
        elif len(paths) == 3:
            self.color_paths, self.depth_paths, self.mask_paths = paths

        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")

        if hasattr(self, 'mask_paths'):
            if len(self.color_paths) != len(self.mask_paths):
                raise ValueError("Number of color and mask images must be the same.")
        
        self.num_imgs = len(self.color_paths)

        if not self.name in ['bop']:
            try:
                self.poses = self.load_poses()
                posesNotFound = False
            except FileNotFoundError as e:
                posesNotFound = True

        if posesNotFound:
            poses = []
            for i in range(self.num_imgs):
                c2w = np.eye(4) 
                c2w = torch.from_numpy(c2w).float()
                poses.append(c2w)
            self.poses = poses
        
        if self.end == -1:
            self.end = self.num_imgs

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.depth_paths = self.depth_paths[self.start : self.end : stride]

        self.poses = self.poses[self.start : self.end : stride]
        # Tensor of retained indices (indices of frames and poses that were retained)
        self.retained_inds = torch.arange(self.num_imgs)[self.start : self.end : stride]
        # Update self.num_images after subsampling the dataset
        self.num_imgs = len(self.color_paths)

        self.poses = np.asarray(self.poses)
        if self.relative_pose:
            # TODO relative transformation in numpy
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

    def __len__(self):
        return self.num_imgs

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_mask(self, mask:np.ndarray):
        mask = cv2.resize(
            mask,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = np.where(mask.astype(bool), 1, 0).astype(np.uint8)
        return mask

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if color.dtype == 'uint8':
            return color
        else:
            return (color * 255.).astype('uint8')

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return depth / self.png_depth_scale

    def _preprocess_poses(self, poses: torch.Tensor):
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
        # TODO to numpy
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    def get_cam_K(self):
        """
        Return camera intrinsics matrix K

        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        """
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path))
        color = self._preprocess_color(color)
        if ".png" in depth_path:
            depth = np.asarray(imageio.imread(depth_path))
            
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        if self.name == 'ho3d_v3' and depth.shape[-1] == 3:
            depth = convert_ho3d_rgb2depth(depth)

        depth = self._preprocess_depth(depth)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)

        pose = self.transformed_poses[index]

        if hasattr(self,'mask_paths'):
            mask_path = self.mask_paths[index]
            mask = np.asarray(imageio.imread(mask_path), dtype=float)
            mask = self._preprocess_mask(mask)

            return (
                color,
                depth,
                mask,
                K,
                pose,
            )

        return (
            color,
            depth,
            K,
            pose,
        )
