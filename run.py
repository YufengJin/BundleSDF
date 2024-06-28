import os, sys
_BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import shutil
import time
import yaml
import cv2
import argparse
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

from datasets.bundlegs_datasets import (load_dataset_config, HO3D_v3Dataset, BOPDataset)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from pose_splats import *

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["ho3d_v3"]:
        return HO3D_v3Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["bop"]:
        return BOPDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def run_once(config: dict):
    cfg_bundletrack_cfg = os.path.join(_BASE_DIR, config["bundletrack_cfg"])
    cfg_bundletrack = yaml.load(open(cfg_bundletrack_cfg, 'r'))#, Loader=yaml.Loader)

    out_folder = os.path.join(
        config["workdir"], config["run_name"]
    )

    # reconfigure 
    cfg_bundletrack['SPDLOG'] = int(config["debug_level"])
    cfg_bundletrack['depth_processing']["zfar"] = 1
    cfg_bundletrack['depth_processing']["percentile"] = 95
    cfg_bundletrack['erode_mask'] = 3
    cfg_bundletrack['debug_dir'] = out_folder+'/'
    cfg_bundletrack['bundle']['max_BA_frames'] = 10
    cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
    cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
    cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
    cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
    cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
    cfg_bundletrack['feature_corres']['map_points'] = True
    cfg_bundletrack['feature_corres']['resize'] = 400
    cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
    cfg_bundletrack['keyframe']['min_rot'] = 5
    cfg_bundletrack['ransac']['inlier_dist'] = 0.01
    cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
    cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
    cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
    cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
    cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
    cfg_bundletrack['p2p']['max_dist'] = 0.02
    cfg_bundletrack['p2p']['max_normal_angle'] = 45
    cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
    yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

    # TODO use Efficient SAM
    use_segmenter = config['use_segmenter']
    if use_segmenter:
        segmenter = Segmenter()

    # dataLoader
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "data_cfg" not in dataset_config:
        data_cfg = {}
        data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        data_cfg = load_dataset_config(dataset_config["data_cfg"])

    dataset = get_dataset(
        config_dict=data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        relative_pose=False,
        target_object_id = dataset_config.get("target_object_id", None),
    )

    # update data frames
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)
        dataset_config['num_frames'] = num_frames

    tracker = PoseSplats(cfg_track_dir=cfg_track_dir, cfg_gs=config)

    if config['load_checkpoint']:
        #TODO check save ckpt before updating checkpoint_time_idx and save into exp
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
    else:
        checkpoint_time_idx = 0

    for time_idx in range(checkpoint_time_idx, num_frames):
        print(f"INFO: Loading data: {time_idx}/{num_frames}(total)")

        # Load RGBD frames incrementally instead of all frames
        data = dataset[time_idx]
        mask = None
        # color: np.uint8(0, 255), depth: float (m); mask: uint8(1, 0)    , in bundlesdf color in bgr
        if len(data) == 5:
            color, depth, mask, K, gt_pose = data
        else:
            color, depth, K, gt_pose = data

        if time_idx == 0:
            if mask is None:
                # get initial mask through labeling 
                if use_segmenter:
                    mask = segmenter()
                else:
                    raise("ERROR: No initial mask")

        else:
            if use_segmenter:
                mask = segmenter.update()
            else:
                mask = mask  
        
        if cfg_bundletrack['erode_mask']>0:
            kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)

        # create a hash
        id_str = f"{time_idx:06d}".encode('utf-8')
        hash_object = hashlib.sha256()
        hash_object.update(id_str)

        # Get the hexadecimal representation of the hash
        id_str = hash_object.hexdigest()[:8]

        id_str += f"{time_idx:06d}"

        # set initial pose identity
        pose_in_model = np.eye(4)

        tracker.run(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)
    tracker.on_finish()

    #TODO pose evaluation


    
    # Object tracking and Gaussian splats generation occur simultaneously
    pass

def run_global_gs():
    # Once object tracking concludes, keyframes are selected for refining Gaussian Splats
    pass

def draw():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to config file")

    args = parser.parse_args()

    # load config
    experiment = SourceFileLoader(
        os.path.basename(args.config), args.config
    ).load_module()

    # set seed
    seed_everything(seed=experiment.config['seed'])

    # create results dir and save config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(results_dir, "config.py"))

    # TODO load checkpoint, only train from scratch currently

    run_once(experiment.config)
    
