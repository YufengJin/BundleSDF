import os
from os.path import join as p_join

# scenes  from 000000 to 000049
scenes = ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009", "000010", "000011", "000012", "000013", "000014", "000015", "000016", "000017", "000018", "000019", "000020", "000021", "000022", "000023", "000024", "000025", "000026", "000027", "000028", "000029", "000030", "000031", "000032", "000033", "000034", "000035", "000036", "000037", "000038", "000039", "000040", "000041", "000042", "000043", "000044", "000045", "000046", "000047", "000048", "000049"]

primary_device="cuda:0"
seed = 0
use_gui = False
debug_level= 2 

scene_name = scenes[0]

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

target_object_id = 4    
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
    use_octree=True,
    save_octree_clouds = True, # save octree clouds for debugging
    octree_smallest_voxel_size = 0.005,
    octree_raytracing_voxel_size = 0.02,
    octree_dilate_size = 0.01,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=0.8, # TODO Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    sync_max_delay=0, # Max frames delay between bundletrack and gaussian splats
    #mean_sq_dist_method="projective", # only use knn gaussian be centered
    max_sh_degree = 2, # Max SH Degree
    rgb2sh=False, # Use RGB2SH for RGB-D Integration
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    save_params=True, # Save Parameters
    save_params_interval=500, # Save Parameters Interval
    use_wandb=False,
    add_new_gaussians=False, # add new gaussians during training         
    add_gaussian_dict=dict( # Needs to be updated based on the number of mapping iterations
        every_iter=100,
        sil_thres=0.8,
        depth_thres=0.01,
    ),
    pipe=dict(
        convert_SHs_python = False,
        compute_cov3D_python = False
    ),
    gaussians_model = dict(
        sh_degree=3,
        position_lr_init=0.00016,              
        position_lr_final=0.0000016,            
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=100,
        densify_until_iter=15_000,
        densify_grad_threshold=0.0002
    ),
    train=dict(
        num_epochs=1,
        batch_size=10,
        batch_iters=1,
        mapping_iters=mapping_iters, # Mapping Iterations
        tracking_iters=tracking_iters, # Tracking Iterations
        sil_thres=0.9,
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.001,
            cam_trans=0.0001,
        ),
        loss_weights=dict(
            mapping=dict(
                im=1.,
                depth=1.,
                edge=0.,
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
