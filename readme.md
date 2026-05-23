# BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects

This is an implementation of our paper published in CVPR 2023

[[Arxiv](https://arxiv.org/abs/2303.14158)] [[Project page](https://bundlesdf.github.io/)] [[Supplemental video](https://www.youtube.com/watch?v=5PymzKbKv8w/)]

# Abstract
We present a near real-time method for 6-DoF tracking of an unknown object from a monocular RGBD video sequence, while simultaneously performing neural 3D reconstruction of the object. Our method works for arbitrary rigid objects, even when visual texture is largely absent. The object is assumed to be segmented in the first frame only. No additional information is required, and no assumption is made about the interaction agent. Key to our method is a Neural Object Field that is learned concurrently with a pose graph optimization process in order to robustly accumulate information into a consistent 3D representation capturing both geometry and appearance. A dynamic pool of posed memory frames is automatically maintained to facilitate communication between these threads. Our approach handles challenging sequences with large pose changes, partial and full occlusion, untextured surfaces, and specular highlights. We show results on HO3D, YCBInEOAT, and BEHAVE datasets, demonstrating that our method significantly outperforms existing approaches.

<img src="./media/problem_setup_c.gif" width="80%">

<img src="./media/preview_results_c.gif" width="80%">

<img src="./media/driller.gif" width="80%">

# Bibtex
```bibtex
@InProceedings{bundlesdfwen2023,
author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
booktitle     = {CVPR},
year          = {2023},
}
```

# Data download
- Download pretrained [weights of segmentation network](https://drive.google.com/file/d/1MEZvjbBdNAOF7pXcq6XPQduHeXB50VTc/view?usp=share_link), and put it under
`./BundleTrack/XMem/saves/XMem-s012.pth`

- Download pretrained [weights of LoFTR outdoor_ds.ckpt](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp), and put it under
`./BundleTrack/LoFTR/weights/outdoor_ds.ckpt`

- Download HO3D data. We provide the augmented data that you can download [here](https://drive.google.com/drive/folders/1Wk-HZDvUExyUrRn7us4WWEbHnnFHgOAX?usp=share_link). Then download YCB-Video object models from [here](https://drive.google.com/file/d/1-1m7qMMyUHYLhaRiQBbsSRMt5dMRX4jD/view?usp=share_link). Finally, make sure the structure is like below, and update your root path of `HO3D_ROOT` at the top of `BundleTrack/scripts/data_reader.py`
  ```
  HO3D_v3
    ├── evaluation
    ├── models
    └── masks_XMem
  ```


# Docker/Environment setup

The environment has been upgraded to a modern CUDA 12.x / PyTorch 2.6 stack
(NVIDIA DeepStream 7.1 base image, kaolin 0.17, pytorch3d, SAM2). It is built and
run via **docker compose** — the repo is live-mounted into the container, so source
edits take effect without rebuilding.

- (Optional, only if you use SAM2 mask generation) download the SAM2 checkpoint once
  on the host. It is **not** baked into the image and **not** committed — it is
  bind-mounted from `.docker_assets/` (see `docker/docker-compose.yaml`):
```
mkdir -p .docker_assets/sam2_checkpoints
curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
     -o .docker_assets/sam2_checkpoints/sam2.1_hiera_small.pt
```

- Build the image and start the container (build only needs to happen once and can
  take a while). The entrypoint compiles the machine-dependent C++/CUDA modules
  (`mycuda`, `BundleTrack`) on first start automatically.
```
cd docker
docker compose up -d --build      # build image + start container (named "bundlesdf")
docker compose exec bundlesdf bash # open a shell inside the running container
```

- Subsequent sessions (image already built):
```
cd docker && docker compose up -d
docker compose exec bundlesdf bash
```

> GUI tools (the SAM2 labeler, `--use_gui`) need X11. The compose file forwards
> `$DISPLAY`; if windows are blocked run `xhost +local:root` on the host first.

# Run on your custom data
- Prepare your RGBD video folder as below (also refer to the example milk data). You can find an [example milk data here](https://drive.google.com/file/d/1akutk_Vay5zJRMr3hVzZ7s69GT4gxuWN/view?usp=share_link) for testing.
```
root
  ├──rgb/    (PNG files)
  ├──depth/  (PNG files, stored in mm, uint16 format. Filename same as rgb)
  ├──masks/       (PNG files. Filename same as rgb. 0 is background. Else is foreground)
  └──cam_K.txt   (3x3 intrinsic matrix, use space and enter to delimit)
```

Due to license issues, we are not able to include [XMem](https://github.com/hkchengrex/XMem) in this codebase for running segmentation online. If you are interested in doing so, please download the code separately and add a wrapper in `segmentation_utils.py`.

### Helper scripts (this fork)

Two helper scripts are provided to produce the layout above (run them **inside the
container**):

- **Record from a ZED camera** → writes `rgb/`, `depth/` (mm uint16) and `cam_K.txt`:
```
python scripts/data_record_bundlesdf.py --obj milk --res HD720 --depth_mode NEURAL
# Ctrl-C to stop. Output goes to a timestamped folder under demo_data/.
```

- **Generate `masks/` with SAM2** (interactive labeler or headless box). Requires the
  SAM2 checkpoint downloaded above:
```
# GUI: scroll frames, drag a box / click points on any frame, Space to propagate, 'w' to save
python scripts/make_masks_sam2.py --video_dir demo_data/<your_clip>

# Headless: box on frame 0, propagate, save
python scripts/make_masks_sam2.py --video_dir demo_data/<your_clip> --bbox 440 180 840 560
```
  SAM2 paths/thresholds live in `configs/sam2.yaml`.

- Run your RGBD video (specify the video_dir and your desired output path). There are 3 steps. Note we assume the max relevant depth in the demo data <1. If this is not the case for you, change it [here](https://github.com/NVlabs/BundleSDF/blob/master/BundleTrack/config_ho3d.yml#L16)
```
# 1) Run joint tracking and reconstruction. 
python run_custom.py --mode run_video --video_dir /home/bowen/debug/2022-11-18-15-10-24_milk --out_folder /home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk --use_segmenter 1 --use_gui 1 --debug_level 2

# 2) Run global refinement post-processing to refine the mesh
python run_custom.py --mode global_refine --video_dir /home/bowen/debug/2022-11-18-15-10-24_milk --out_folder /home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk   # Change the path to your video_directory

# 3) (Optional) If you want to draw the oriented bounding box to visualize the pose, similar to our demo
python run_custom.py --mode draw_pose --out_folder /home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk
```

- Finally the results will be dumped in the `out_folder`, including the tracked poses stored in `ob_in_cam/` and reconstructed mesh with texture `textured_mesh.obj`.

### Viewing the textured mesh

`textured_mesh.obj` is a valid OBJ, but the **MeshLab 2020.09** shipped by Ubuntu apt
has a buggy OBJ importer that can abort with
`import_obj.h:640: Assertion ... numVerticesPlusFaces ... failed` (core dump). This is
a MeshLab importer bug, not a problem with the mesh (it loads fine in trimesh, Blender,
f3d, etc.). Options:

- Use a viewer with a different importer — recommended `f3d` (lightweight, opens the
  textured OBJ directly): `sudo apt install f3d && f3d out_folder/textured_mesh.obj`.
  Blender and CloudCompare also work.
- Or use the latest MeshLab AppImage (importer rewritten, bug fixed) from the
  [MeshLab releases](https://github.com/cnr-isti-vclab/meshlab/releases).
- Or convert the mesh (e.g. with trimesh) to `.glb`/`.ply`, which MeshLab 2020.09 reads
  without issue.

<img src="./media/milk_jug.gif" height="400">


# Run on HO3D dataset
```
# Run BundleSDF to get the pose and reconstruction results
python run_ho3d.py --video_dirs /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/SM1 --out_dir /home/bowen/debug/ho3d_ours

# Benchmark the output results
python benchmark_ho3d.py --video_dirs /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/SM1 --out_dir /home/bowen/debug/ho3d_ours
```


# Acknowledgement

We would like to thank Jeff Smith for helping with the code release. Marco Foco and his team for providing the test data on the static scene.


# Contact
For questions, please contact Bowen Wen (bowenw@nvidia.com)
