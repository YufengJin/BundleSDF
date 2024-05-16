import os
from os.path import join as p_join

# scenes  from 000000 to 000049
scenes = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009", "000010", "000011", "000012", "000013", "000014", "000015", "000016", "000017", "000018", "000019", "000020", "000021", "000022", "000023", "000024", "000025", "000026", "000027", "000028", "000029", "000030", "000031", "000032", "000033", "000034", "000035", "000036", "000037", "000038", "000039", "000040", "000041", "000042", "000043", "000044", "000045", "000046", "000047", "000048", "000049"]

primary_device="cuda:0"
seed = 0
use_gui = False
debug_level= 3 

scene_name = scenes[2]

target_object_id = 1    
map_every = 1 
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 80 
mapping_iters = 80

group_name = "bop"
run_name = f"{scene_name}_{seed}"

config = dict(
    bundletrack_cfg = './BundleTrack/config_ho3d.yml',
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    debug_level=debug_level, 
    seed=seed,
    primary_device=primary_device,
    use_segmenter=False,
    use_gui=use_gui,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=0.8, # TODO Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    sync_max_delay=0, # Max frames delay between bundletrack and gaussian splats
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=False,
    add_new_gaussians=False, # add new gaussians during training         
    add_gaussian_dict=dict( # Needs to be updated based on the number of mapping iterations
        every_iter=100,
        sil_thres=0.8,
        depth_thres=0.01,
    ),
    train=dict(
        num_epochs=1,
        batch_size=10,
        batch_iters=1,
        sil_thres=0.9,
        lrs=dict(
            #means3D=0.0001,
            #rgb_colors=0.0025,
            #unnorm_rotations=0.001,
            #logit_opacities=0.05,
            #log_scales=0.001,
            #cam_unnorm_rots=0.0
            #cam_trans=0.00,
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.00,
            cam_unnorm_rots=0.001,
            cam_trans=0.0001
        ),
        loss_weights=dict(
            mapping=dict(
                im=1.,
                depth=1.,
                edge=1.,
                silhouette=1.
            ),
            tracking=dict(
                im=1., 
                depth=1.,
                edge=1.,
                silhouette=0.
            ),
        ),
        prune_gaussians=False, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=3000,
            stop_after=5000,
            prune_every=1,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.25,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=True, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=100,
            remove_big_after=500,
            stop_after=5000,
            densify_every=50,
            grad_thresh=0.0001,
            num_to_split_into=2,
            removal_opacity_threshold=0.02,
            final_removal_opacity_threshold=0.25,
            reset_opacities=False,
            reset_opacities_every=600, # Doesn't consider iter 0
        ),
    ),
    wandb=dict(
        #entity="theairlab",
        project="SplaTAM",
        #group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/home/datasets/BOP/ycbv/train_pbr", 
        data_cfg="./configs/data/bop.yaml",  #None
        sequence=scene_name,
        target_object_id = target_object_id,
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
    ),
)
