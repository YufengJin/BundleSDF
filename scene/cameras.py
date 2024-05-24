#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2_torch, getProjectionMatrix, SE3_exp

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        mask,
        fovx,
        fovy,
        R,
        t,
        image_height,
        image_width,
        trans=np.array([0.0, 0.0, 0.0]), 
        scale=1.0,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        self.R = R.to(device)
        self.T = t.to(device)

        self.color = color.clamp(0.0, 1.0).to(device)
        self.depth = depth.to(device) if depth is not None else None
        self.mask = mask.to(device) if mask is not None else None

        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = torch.tensor(trans, device=device)
        self.scale = torch.tensor(scale, device=device)

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

    @property
    def world_view_transform(self):
        return getWorld2View2_torch(self.R, self.T, self.trans, self.scale).transpose(0, 1).cuda()

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    @torch.no_grad()
    def update(self, converged_threshold=1e-5):
        tau = torch.cat([self.cam_trans_delta, self.cam_rot_delta], axis=0)
        
        T_w2c = torch.eye(4, device=self.device)
        T_w2c[0:3, 0:3] = self.R
        T_w2c[0:3, 3] = self.T

        new_w2c = SE3_exp(tau) @ T_w2c

        self.R = new_w2c[0:3, 0:3]
        self.T = new_w2c[0:3, 3]

        self.cam_rot_delta.data.fill_(0)
        self.cam_trans_delta.data.fill_(0)

        converged = tau.norm() < converged_threshold
        return converged

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None