import glob
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class BOPDataset(GradSLAMDataset):
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
        # avoid two same objects in the scene
        self.input_folder = os.path.join(basedir, sequence)

        # get target obj id
        target_object_id = kwargs.get('target_object_id')
        if target_object_id is None:
            print("ERROR: Bopdataset must set a desired object ID")
            raise
        else:
            # set object id
            self.target_object_id = target_object_id

        self.pose_path = os.path.join(self.input_folder, "scene_gt.json")
        self._mask_paths = list() 
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
        cam_paths = os.path.join(self.input_folder, "scene_camera.json")
        cameraInfo = json.load(open(cam_paths,  'r'))
        keys = list(cameraInfo.keys())
        K = np.asarray(cameraInfo[keys[0]]['cam_K'], dtype=np.float64).reshape(3, 3)
        self.fx = K[0][0] 
        self.fy = K[1][1] 
        self.cx = K[0][2] 
        self.cy = K[1][2] 
        depth_scale = np.asarray(cameraInfo[keys[0]]['depth_scale'],np.float64)
        self.png_depth_scale = 1./ depth_scale * 1000.

    def get_filepaths(self):
        color_paths = []
        for ext in ['.jpg', '.png']:
            if len(color_paths) > 0:
                break
            color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/*{ext}"))

        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        if len(self._mask_paths) == 0:
            print("ERROR: BOPDataset has not loaded poses, masks_path not found")
            raise 

        mask_paths  = natsorted(self._mask_paths)
        return color_paths, depth_paths, mask_paths

    def load_poses(self):
        poses = []
        scene_gt = json.load(open(self.pose_path, 'r'))

        # TODO generalize to multi objects
        for imIdx, content in scene_gt.items():
            objFound = False
            
            for i, obj_info in enumerate(content):
                if int(obj_info['obj_id']) == int(self.target_object_id):
                    if objFound:
                        print("ERROR: obj {int(self.target_object_id)} occurs twices")
                        raise 
                    # object 6D Pose
                    c2w = np.eye(4)
                    R = np.array(obj_info['cam_R_m2c']).reshape(3,3)
                    t = np.array(obj_info['cam_t_m2c'])
                    c2w[:3, :3] = R
                    c2w[:3, 3] = t
                    c2w[:3, 3] /= 1000. 

                    #c2w = np.linalg.inv(c2w)
                    ##c2w[:3, 1:3] = -c2w[:3,1:3]
                    #c2w[:3, 3] /= 1000. 
                    #c2w = torch.from_numpy(c2w).float()
                    poses.append(c2w)

                    # update masks
                    self._mask_paths.append(os.path.join(self.input_folder, "mask", f"{int(imIdx):06d}_{i:06d}.png"))
                    objFound = True
            if not objFound:
                print(f"WARNING: Obj {int(self.target_object_id)} not found in image {imIdx:06d}")

        return poses

