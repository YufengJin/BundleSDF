#!/usr/bin/env python3
"""Record ZED RGB + depth into BundleSDF demo_data layout, via the ZED SDK.

Uses the ZED Python SDK (pyzed.sl) directly — no ROS. Runs inside the BundleSDF
docker (where the ZED SDK is installed) with the ZED camera passed through.

Writes:

    <out>/<YYYY-MM-DD-HH-MM-SS>_<obj>/
        cam_K.txt          # 3x3 left-rectified intrinsics for the chosen resolution
        rgb/<ts_ns>.png     # 8-bit BGR PNG (rectified LEFT view)
        depth/<ts_ns>.png   # 16-bit PNG, millimeters (registered to LEFT)

Layout matches demo_data/2022-11-18-15-10-24_milk/. masks/tf/annotated_poses/
ply are BundleSDF downstream products and are not produced here.

Depth is requested in MILLIMETER units, so it maps straight to uint16 PNG with
no scaling — consistent with BundleTrack/scripts/data_reader.py:83 (`/1e3`).

Usage (inside the BundleSDF container):
    python scripts/data_record_bundlesdf.py --obj milk --res HD720 --depth_mode NEURAL
    # A live RGB preview window opens by default (q/ESC in the window, or Ctrl-C, to stop).
    # Add --no-gui when running headless / without X11 forwarding.
"""

import argparse
import os
import signal
import sys
from datetime import datetime

import cv2
import numpy as np
import pyzed.sl as sl


def build_init_params(res_name, fps, depth_mode_name, min_depth, max_depth):
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, res_name)
    init.camera_fps = fps
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode_name)
    init.coordinate_units = sl.UNIT.MILLIMETER  # depth measure -> mm directly
    init.depth_minimum_distance = float(min_depth)
    init.depth_maximum_distance = float(max_depth)
    return init


def get_left_K(zed):
    """3x3 intrinsics of the rectified LEFT camera at the open resolution."""
    info = zed.get_camera_information()
    # SDK 4.x/5.x: calibration lives under camera_configuration
    conf = getattr(info, "camera_configuration", info)
    cam = conf.calibration_parameters.left_cam
    return np.array(
        [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj", default="object", help="object name suffix for the output folder")
    p.add_argument("--out", default="demo_data", help="output root directory")
    p.add_argument("--res", default="HD720", help="HD2K | HD1080 | HD720 | VGA")
    p.add_argument("--fps", type=int, default=30, help="camera fps (must be valid for --res)")
    p.add_argument("--depth_mode", default="NEURAL",
                   help="NONE | PERFORMANCE | QUALITY | ULTRA | NEURAL | NEURAL_PLUS")
    p.add_argument("--min_depth", type=float, default=500.0, help="min depth in mm")
    p.add_argument("--max_depth", type=float, default=1000.0, help="max depth in mm")
    p.add_argument("--svo", default=None, help="optional .svo(2) file to replay instead of a live camera")
    p.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True,
                   help="show a live RGB preview window (q/ESC to stop). Use --no-gui when headless.")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join(args.out, f"{stamp}_{args.obj}")
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    init = build_init_params(args.res, args.fps, args.depth_mode, args.min_depth, args.max_depth)
    if args.svo:
        init.set_from_svo_file(args.svo)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] ZED open failed: {status}", file=sys.stderr)
        sys.exit(1)

    K = get_left_K(zed)
    np.savetxt(os.path.join(out_dir, "cam_K.txt"), K)
    print(f"Recording into: {out_dir}")
    print(f"resolution={args.res} fps={args.fps} depth_mode={args.depth_mode} "
          f"depth_range=[{args.min_depth},{args.max_depth}]mm")
    print(f"cam_K:\n{K}")
    if args.gui:
        print("Live RGB preview on. Press q or ESC in the window (or Ctrl-C) to stop.")
    else:
        print("Press Ctrl-C to stop.")

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    WIN = "BundleSDF recorder - RGB (q/ESC to stop)"
    gui = args.gui  # may be disabled at runtime if no display is available

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    count = 0
    try:
        while not stop["flag"]:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                # SVO end-of-file or transient grab failure
                if args.svo:
                    break
                continue
            zed.retrieve_image(image, sl.VIEW.LEFT)          # BGRA8
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)    # float32, mm
            ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            name = f"{ts}.png"

            bgr = image.get_data()[:, :, :3]                 # drop alpha
            cv2.imwrite(os.path.join(rgb_dir, name), bgr)

            d = depth.get_data().astype(np.float32)          # mm
            d[~np.isfinite(d)] = 0.0
            d[d < 0] = 0.0
            d16 = np.clip(d, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(depth_dir, name), d16)

            count += 1
            if count % 30 == 0:
                print(f"saved {count} frames")

            if gui:
                # Overlay frame counter on a copy so the saved PNG stays clean.
                disp = np.ascontiguousarray(bgr)
                cv2.putText(disp, f"REC  frames={count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                try:
                    cv2.imshow(WIN, disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # q or ESC
                        stop["flag"] = True
                except cv2.error as e:
                    # No display (headless / X11 not forwarded): keep recording.
                    print(f"[WARN] GUI disabled, no display available: {e}", file=sys.stderr)
                    gui = False
    finally:
        if args.gui:
            cv2.destroyAllWindows()
        zed.close()
        print(f"Stopped. Saved {count} frames into {out_dir}")


if __name__ == "__main__":
    main()
