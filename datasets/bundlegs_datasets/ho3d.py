import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class HO3D_v3Dataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, "evaluation", sequence)
        self.mask_folder  = os.path.join(basedir, "masks_XMem", sequence)

        self.pose_path = os.path.join(self.input_folder, "ob_in_cams.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

        self._update_camera_params()

    def _update_camera_params(self):
        import pickle
        cam_paths = natsorted(glob.glob(f"{self.input_folder}/meta/*.pkl"))
        K = pickle.load(open(cam_paths[0], 'rb'))['camMat']
        self.fx = K[0][0] 
        self.fy = K[1][1] 
        self.cx = K[0][2] 
        self.cy = K[1][2] 

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        mask_paths  = natsorted(glob.glob(f"{self.mask_folder}/*.png"))

        return color_paths, depth_paths, mask_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            c2w = []
            for rowId in range(4):
                c2w += list(map(float, lines[4*i+rowId].split()))
            # object 6D Pose
            c2w = np.array(c2w).reshape(4,4)
            #c2w = np.linalg.inv(c2w)
            ##c2w[:3, 1] *= -1
            ##c2w[:3, 2] *= -1
            #c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

