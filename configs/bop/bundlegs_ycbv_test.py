import os
from os.path import join as p_join

scenes = ["000048", "000049", "000050", "000051", "000052", "000053", "000054", "000055", "000056", "000057", "000058", "000059"]

device="cuda:0"
seed = 0
use_gui = False 
debug_level= 2 

scene_name = scenes[2]

"""
Object 01 (002_master_chef_can): [1.3360, -0.5000, 3.5105]
Object 02 (003_cracker_box): [0.5575, 1.7005, 4.8050]
Object 03 (004_sugar_box): [-0.9520, 1.4670, 4.3645]
Object 04 (005_tomato_soup_can): [-0.0240, -1.5270, 8.4035]
Object 05 (006_mustard_bottle): [1.2995, 2.4870, -11.8290]
Object 06 (007_tuna_fish_can): [-0.1565, 0.1150, 4.2625]
Object 07 (008_pudding_box): [1.1645, -4.2015, 3.1190]
Object 08 (009_gelatin_box): [1.4460, -0.5915, 3.6085]
Object 09 (010_potted_meat_can): [2.4195, 0.3075, 8.0715]
Object 10 (011_banana): [-18.6730, 12.1915, -1.4635]
Object 11 (019_pitcher_base): [5.3370, 5.8855, 25.6115]
Object 12 (021_bleach_cleanser): [4.9290, -2.4800, -13.2920]
Object 13 (024_bowl): [-0.2270, 0.7950, -2.9675]
Object 14 (025_mug): [-8.4675, -0.6995, -1.6145]
Object 15 (035_power_drill): [9.0710, 20.9360, -2.1190]
Object 16 (036_wood_block): [1.4265, -2.5305, 17.1890]
Object 17 (037_scissors): [7.0535, -28.1320, 0.0420]
Object 18 (040_large_marker): [0.0460, -2.1040, 0.3500]
Object 19 (051_large_clamp): [10.5180, -1.9640, -0.4745]
Object 20 (052_extra_large_clamp): [-0.3950, -10.4130, 0.1620]
Object 21 (061_foam_brick): [-0.0805, 0.0805, -8.2435]

"""

target_object_id = 5    
start_gs_keyframes = 5 # start keyframes for gaussian splats
map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 80 
mapping_iters = 80

group_name = "bop_ycbv_test"
run_name = f"{scene_name}_{seed}"

config = dict(
    bundletrack_cfg = './BundleTrack/config_ho3d.yml',
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    start_gs_keyframes=start_gs_keyframes,
    debug_level=debug_level, 
    seed=seed,
    device=device,
    use_segmenter=False,
    continual=True,    # Continual Learning scale factor and translation
    use_gui=use_gui,
    
    # octree
    use_octree=True, # Use Octree
    octree_smallest_voxel_size=0.02, # Smallest Voxel Size for Octree
    octree_raytracing_voxel_size=0.02,
    octree_dilate_size=0.02, 

    down_scale_ratio=2, # Downscale Ratio for RGB and Depth
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
    vox_res=0.01, # Voxel Resolution for DBSCAN
    gaussians=dict(
        rgb2sh=False, # Convert RGB to SH
        init_pts_noise=0.02, # Initial Gaussian Noise
        distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)

    ),
    dbscan=dict(
        eps=0.06,
        eps_min_samples=1,
    ),
    optimizer=dict(
        lrs=dict(
                means3D=0.0001,
                rgb_colors=0.0025,
                unnorm_rotations=0.001,
                logit_opacities=0.05,
                log_scales=0.001,
                cam_unnorm_rots=0.001,
                cam_trans=0.0001,
                sdf=0.0001,
            ),
    ),
    train=dict(
        num_epochs=500,
        batch_size=10,
        sil_thres=0.9,
        use_edge_loss=True,
        use_depth_loss=True,
        use_silhouette_loss=True,
        use_im_loss=True,
        loss_weights=dict(
            im=1.,
            depth=1.,
            edge=0.,
            silhouette=1.
        ),
        densification=dict( # Needs to be updated based on the number of mapping iterations
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
    data=dict(
        basedir="/home/datasets/BOP/ycbv/test", 
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
