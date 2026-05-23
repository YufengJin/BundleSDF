#!/usr/bin/env python3
"""Generate per-frame single-object masks for a recorded demo_data folder, via SAM2.

Offline tool — no ROS. Runs inside the BundleSDF docker (SAM2 fork at /opt/sam2).
Uses the SAM2 **video predictor** so prompts can be placed on ANY frame and refined
iteratively, then propagated (bidirectionally) across the whole clip.

Two modes:
  GUI (default when a display is available, or --gui):
      Tk previewer. Scroll frames with Left/Right, drag a box / click points to
      prompt on any frame, Space to propagate, scroll to inspect the overlay, add
      correction prompts where wrong, re-propagate, then 'w' to save.
  Headless (--bbox x0 y0 x1 y1):
      box on frame 0 -> propagate -> save. No window.

For <video_dir>/rgb/<name>.png it writes:
    <video_dir>/masks/<name>.png      uint8, object=255 / bg=0   (per frame)
    <video_dir>/masks_vis/<name>.png  RGB overlay                (unless --no_vis)
    <video_dir>/mask.png              copy of the first-frame mask
Mask filenames match rgb (BundleSDF reads masks/<same_name>.png; nonzero=object).

Usage:
    python scripts/make_masks_sam2.py --video_dir demo_data/records_1242_1905/mug
    python scripts/make_masks_sam2.py --video_dir <dir> --bbox 440 180 840 560
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
import yaml

from sam2.build_sam import build_sam2_video_predictor


# --------------------------------------------------------------------------- cfg
def load_cfg(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def logits_to_mask(obj_ids, mask_logits, obj_id=1):
    """Pull obj_id's mask out of a SAM2 (obj_ids, logits) return -> bool HxW."""
    for oid, logit in zip(obj_ids, mask_logits):
        if int(oid) != obj_id:
            continue
        m = (logit > 0.0).cpu().numpy()
        while m.ndim > 2:
            m = m[0]
        return m.astype(bool)
    return None


# ----------------------------------------------------------------- engine wrapper
class VideoLabeler:
    """Wraps the SAM2 video predictor: prompts per frame + propagated masks."""

    OBJ_ID = 1

    def __init__(self, video_dir, model_cfg, ckpt, no_vis=False, min_px=100):
        self.video_dir = video_dir
        self.no_vis = no_vis
        self.min_px = min_px

        self.rgb_files = sorted(glob.glob(os.path.join(video_dir, "rgb", "*.png")))
        if not self.rgb_files:
            raise RuntimeError(f"no rgb/*.png under {video_dir}")
        self.names = [os.path.basename(f) for f in self.rgb_files]
        self.n = len(self.rgb_files)
        h, w = cv2.imread(self.rgb_files[0]).shape[:2]
        self.H, self.W = h, w

        # Stage frames as <i>.jpg for the video predictor (it only takes a JPEG dir).
        self.tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        for i, f in enumerate(self.rgb_files):
            cv2.imwrite(os.path.join(self.tmp_dir, f"{i}.jpg"), cv2.imread(f))

        print(f"loading SAM2 video predictor (cfg={model_cfg}, ckpt={ckpt})")
        self.predictor = build_sam2_video_predictor(model_cfg, ckpt)
        # Make every frame that receives a click a CONDITIONING frame. Default is
        # False, which stores corrections as non-cond outputs -> re-propagation
        # overwrites them from the original prompts and the correction is lost.
        # True = corrections anchor propagation and stick (SAM2's finetune setting).
        self.predictor.add_all_frames_to_correct_as_cond = True
        self.state = self.predictor.init_state(
            self.tmp_dir, offload_video_to_cpu=True, offload_state_to_cpu=True
        )

        # frame_idx -> {"points": [(x,y,label)...], "box": (x0,y0,x1,y1)|None}
        self.prompts = {}
        self.masks = {}  # frame_idx -> bool HxW

    # -- prompt editing -------------------------------------------------------
    def _ensure(self, fidx):
        return self.prompts.setdefault(fidx, {"points": [], "box": None})

    def _forget_frame(self, fidx):
        """Drop SAM2's tracked/history state for one frame so the next add segments
        it FRESH from the current prompts, instead of "correcting" the already-
        propagated mask (which biases the live preview toward the old, possibly
        wrong, mask). Safe: propagate() fully resets + replays from self.prompts."""
        st = self.state
        st["frames_already_tracked"].pop(fidx, None)
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            st["output_dict"][key].pop(fidx, None)
            for od in st["output_dict_per_obj"].values():
                od[key].pop(fidx, None)
            for td in st["temp_output_dict_per_obj"].values():
                td[key].pop(fidx, None)
            st["consolidated_frame_inds"][key].discard(fidx)

    def _send(self, fidx):
        """Push this frame's full prompt set to SAM2; return its immediate mask.
        The frame is segmented fresh from its own prompts (see _forget_frame), so
        the returned mask is a faithful preview of what your prompts produce here."""
        p = self.prompts.get(fidx)
        if not p or (not p["points"] and p["box"] is None):
            return None
        self._forget_frame(fidx)
        points = np.array([[x, y] for x, y, _ in p["points"]], dtype=np.float32) \
            if p["points"] else None
        labels = np.array([lab for _, _, lab in p["points"]], dtype=np.int32) \
            if p["points"] else None
        box = np.array(p["box"], dtype=np.float32) if p["box"] is not None else None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, oids, logits = self.predictor.add_new_points_or_box(
                self.state, frame_idx=fidx, obj_id=self.OBJ_ID,
                points=points, labels=labels, box=box, clear_old_points=True,
            )
        m = logits_to_mask(oids, logits, self.OBJ_ID)
        if m is not None:
            self.masks[fidx] = m
        return m

    def add_point(self, fidx, x, y, label):
        self._ensure(fidx)["points"].append((float(x), float(y), int(label)))
        return self._send(fidx)

    def set_box(self, fidx, box):
        self._ensure(fidx)["box"] = tuple(float(v) for v in box)
        return self._send(fidx)

    def clear_frame(self, fidx):
        """Drop one frame's prompts; SAM2 has no per-frame remove -> reset + replay."""
        if fidx not in self.prompts:
            return
        del self.prompts[fidx]
        remaining = dict(self.prompts)
        self.predictor.reset_state(self.state)
        self.masks = {}
        self.prompts = {}
        for f in sorted(remaining):
            self.prompts[f] = remaining[f]
            self._send(f)

    def reset_all(self):
        self.predictor.reset_state(self.state)
        self.prompts = {}
        self.masks = {}

    def prompted_frames(self):
        return sorted(self.prompts.keys())

    # -- propagation ----------------------------------------------------------
    def propagate(self):
        if not self.prompts:
            return False
        # Re-assert every prompt into a fresh SAM2 state so the engine can never
        # drift from self.prompts (what the frame list shows). A swallowed error
        # during interactive add could otherwise leave SAM2 with zero conditioning
        # frames while the list still shows them -> propagate_in_video raises
        # "No points are provided". After reset, every prompted frame is an initial
        # conditioning frame and anchors propagation.
        self.predictor.reset_state(self.state)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for f in sorted(self.prompts):
                self._send(f)
        self.masks = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for fidx, oids, logits in self.predictor.propagate_in_video(self.state):
                self.masks[fidx] = logits_to_mask(oids, logits, self.OBJ_ID)
            for fidx, oids, logits in self.predictor.propagate_in_video(
                self.state, reverse=True
            ):
                self.masks[fidx] = logits_to_mask(oids, logits, self.OBJ_ID)
        return True

    # -- output ---------------------------------------------------------------
    def save(self):
        masks_dir = os.path.join(self.video_dir, "masks")
        vis_dir = os.path.join(self.video_dir, "masks_vis")
        os.makedirs(masks_dir, exist_ok=True)
        if not self.no_vis:
            os.makedirs(vis_dir, exist_ok=True)
        n_lost = 0
        for i, name in enumerate(self.names):
            m = self.masks.get(i)
            out = np.zeros((self.H, self.W), dtype=np.uint8)
            area = int(m.sum()) if m is not None else 0
            if m is not None and area >= self.min_px:
                out[m] = 255
            else:
                n_lost += 1
            cv2.imwrite(os.path.join(masks_dir, name), out)
            if not self.no_vis:
                bgr = cv2.imread(self.rgb_files[i])
                if out.any():
                    ov = bgr.copy()
                    ov[out > 0] = (0, 0, 255)
                    bgr = cv2.addWeighted(bgr, 0.6, ov, 0.4, 0)
                cv2.imwrite(os.path.join(vis_dir, name), bgr)
            if i == 0:
                cv2.imwrite(os.path.join(self.video_dir, "mask.png"), out)
        print(f"Saved {self.n} masks -> {masks_dir}  "
              f"({n_lost} empty/below {self.min_px}px). mask.png written.")
        return n_lost

    def close(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# ------------------------------------------------------------------------- Tk GUI
def run_gui(labeler: "VideoLabeler"):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image as PILImage, ImageTk

    MAX_W, MAX_H = 1280, 800

    class Gui:
        def __init__(self, lab):
            self.lab = lab
            self.idx = 0
            self.scale = 1.0
            self._tk_img = None
            self.press = None          # (x,y) canvas, for box-vs-click
            self.rect_id = None

            self.root = tk.Tk()
            self.root.title("SAM2 multi-frame labeler")
            self.root.protocol("WM_DELETE_WINDOW", self._quit)
            self._build()
            for k in ("<Left>", "<Right>", "<Shift-Left>", "<Shift-Right>",
                      "<space>", "w", "c", "r", "q"):
                self.root.bind(k, self._on_key)
            self._refresh_list()
            self._show()

        def _build(self):
            outer = ttk.Frame(self.root); outer.pack(fill=tk.BOTH, expand=True)
            left = ttk.Frame(outer); left.pack(side=tk.LEFT, fill=tk.Y)
            self.info = tk.StringVar()
            ttk.Label(left, textvariable=self.info, anchor=tk.W,
                      justify=tk.LEFT).pack(fill=tk.X, padx=4, pady=4)
            ttk.Label(left, text="frames with prompts:").pack(fill=tk.X, padx=4)
            self.listbox = tk.Listbox(left, height=24, width=22)
            self.listbox.pack(side=tk.LEFT, fill=tk.Y, padx=4)
            self.listbox.bind("<<ListboxSelect>>", self._on_list)

            right = ttk.Frame(outer); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.canvas = tk.Canvas(right, bg="#101010", highlightthickness=0,
                                    width=960, height=540, cursor="crosshair")
            self.canvas.pack(fill=tk.BOTH, expand=True)
            self.canvas.bind("<ButtonPress-1>", self._press)
            self.canvas.bind("<B1-Motion>", self._drag)
            self.canvas.bind("<ButtonRelease-1>", self._release)
            self.canvas.bind("<Button-3>", self._rclick)

            bottom = ttk.Frame(self.root); bottom.pack(side=tk.BOTTOM, fill=tk.X)
            for txt, cmd in (("Propagate (space)", self._propagate),
                             ("Save (w)", self._save),
                             ("Clear frame (c)", self._clear),
                             ("Reset all (r)", self._reset),
                             ("Quit (q)", self._quit)):
                ttk.Button(bottom, text=txt, command=cmd).pack(side=tk.LEFT, padx=3, pady=3)
            self.status = tk.StringVar()
            ttk.Label(bottom, textvariable=self.status, anchor=tk.W
                      ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            self.status.set("LMB drag=box | LMB click=+point | RMB click=-point | "
                            "←/→ scroll | space=propagate | w=save")

        # -- coordinate mapping --
        def _c2i(self, cx, cy):
            return cx / self.scale, cy / self.scale

        # -- rendering --
        def _show(self):
            bgr = cv2.imread(self.lab.rgb_files[self.idx])
            m = self.lab.masks.get(self.idx)
            if m is not None and m.shape[:2] == bgr.shape[:2] and m.any():
                ov = bgr.copy(); ov[m] = (0, 0, 255)
                bgr = cv2.addWeighted(bgr, 0.55, ov, 0.45, 0)
            p = self.lab.prompts.get(self.idx)
            if p:
                if p["box"] is not None:
                    x0, y0, x1, y1 = [int(v) for v in p["box"]]
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                for x, y, lab in p["points"]:
                    col = (0, 255, 0) if lab == 1 else (0, 0, 255)
                    cv2.circle(bgr, (int(x), int(y)), 5, col, -1)
                    cv2.circle(bgr, (int(x), int(y)), 6, (255, 255, 255), 1)
            cw = min(max(self.canvas.winfo_width(), 320), MAX_W)
            ch = min(max(self.canvas.winfo_height(), 240), MAX_H)
            self.scale = min(cw / self.lab.W, ch / self.lab.H)
            disp = cv2.resize(bgr, (max(1, int(self.lab.W * self.scale)),
                                    max(1, int(self.lab.H * self.scale))),
                              interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self._tk_img = ImageTk.PhotoImage(PILImage.fromarray(rgb))
            self.canvas.delete("frame")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img, tags="frame")
            self.canvas.tag_lower("frame")
            np_ = len(p["points"]) if p else 0
            hasb = "box" if (p and p["box"] is not None) else "no-box"
            self.info.set(f"frame {self.idx+1}/{self.lab.n}\n"
                          f"name {self.lab.names[self.idx]}\n"
                          f"this frame: {np_} pts, {hasb}\n"
                          f"prompted frames: {len(self.lab.prompted_frames())}")

        def _refresh_list(self):
            self.listbox.delete(0, tk.END)
            for f in self.lab.prompted_frames():
                self.listbox.insert(tk.END, f"frame {f}  ({self.lab.names[f]})")

        # -- events --
        def _goto(self, idx):
            self.idx = max(0, min(self.lab.n - 1, idx)); self._show()

        def _on_key(self, e):
            k = e.keysym
            shift = (e.state & 0x1) != 0
            if k == "Left": self._goto(self.idx - (10 if shift else 1))
            elif k == "Right": self._goto(self.idx + (10 if shift else 1))
            elif k == "space": self._propagate()
            elif k == "w": self._save()
            elif k == "c": self._clear()
            elif k == "r": self._reset()
            elif k == "q": self._quit()

        def _on_list(self, _e):
            sel = self.listbox.curselection()
            if sel:
                self._goto(self.lab.prompted_frames()[sel[0]])

        def _press(self, e):
            self.press = (e.x, e.y)
            if self.rect_id: self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                        outline="#00ff00", width=2)

        def _drag(self, e):
            if self.press and self.rect_id:
                self.canvas.coords(self.rect_id, self.press[0], self.press[1], e.x, e.y)

        def _release(self, e):
            if not self.press:
                return
            x0, y0 = self.press; x1, y1 = e.x, e.y
            self.press = None
            if self.rect_id:
                self.canvas.delete(self.rect_id); self.rect_id = None
            try:
                if abs(x1 - x0) < 5 and abs(y1 - y0) < 5:      # click -> +point
                    ix, iy = self._c2i(x1, y1)
                    self.lab.add_point(self.idx, ix, iy, 1)
                    self.status.set(f"+point on frame {self.idx}")
                else:                                          # drag -> box
                    ix0, iy0 = self._c2i(x0, y0); ix1, iy1 = self._c2i(x1, y1)
                    box = (min(ix0, ix1), min(iy0, iy1), max(ix0, ix1), max(iy0, iy1))
                    self.lab.set_box(self.idx, box)
                    self.status.set(f"box on frame {self.idx}")
            except Exception as ex:
                self.status.set(f"SAM2 error: {ex}")
            self._refresh_list(); self._show()

        def _rclick(self, e):
            try:
                ix, iy = self._c2i(e.x, e.y)
                self.lab.add_point(self.idx, ix, iy, 0)
                self.status.set(f"-point on frame {self.idx}")
            except Exception as ex:
                self.status.set(f"SAM2 error: {ex}")
            self._refresh_list(); self._show()

        def _propagate(self):
            self.status.set("propagating...")
            self.root.update_idletasks()
            try:
                ok = self.lab.propagate()
                self.status.set("propagated across all frames" if ok
                                else "no prompts yet — add a box/point first")
            except Exception as ex:
                self.status.set(f"propagate error: {ex}")
            self._show()

        def _save(self):
            self.status.set("saving...")
            self.root.update_idletasks()
            try:
                n_lost = self.lab.save()
                self.status.set(f"saved. {n_lost} empty frames. "
                                f"masks/ + masks_vis/ + mask.png written.")
            except Exception as ex:
                self.status.set(f"save error: {ex}")

        def _clear(self):
            self.lab.clear_frame(self.idx); self._refresh_list(); self._show()
            self.status.set(f"cleared prompts on frame {self.idx}")

        def _reset(self):
            self.lab.reset_all(); self._refresh_list(); self._show()
            self.status.set("reset all prompts")

        def _quit(self):
            try: self.root.quit()
            finally: self.root.destroy()

        def run(self):
            self.root.mainloop()

    Gui(labeler).run()


# ---------------------------------------------------------------------- headless
def run_headless(labeler: "VideoLabeler", bbox):
    x0, y0, x1, y1 = bbox
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    print(f"frame 0 box={box}; propagating...")
    labeler.set_box(0, box)
    labeler.propagate()
    labeler.save()


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video_dir", required=True, help="folder containing rgb/")
    p.add_argument("--config", default="configs/sam2.yaml", help="sam2/tracking config yaml")
    p.add_argument("--sam2_ckpt", default=None, help="override checkpoint path")
    p.add_argument("--sam2_cfg", default=None, help="override model_cfg path")
    p.add_argument("--gui", action="store_true", help="force the Tk labeler")
    p.add_argument("--bbox", type=int, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"), help="headless: box on frame 0")
    p.add_argument("--min_mask_pixels", type=int, default=None)
    p.add_argument("--no_vis", action="store_true", help="do not write masks_vis/")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    sam2_cfg = cfg.get("sam2", {})
    ckpt = args.sam2_ckpt or sam2_cfg.get("checkpoint")
    model_cfg = args.sam2_cfg or sam2_cfg.get("model_cfg")
    min_px = (args.min_mask_pixels if args.min_mask_pixels is not None
              else int(cfg.get("tracking", {}).get("sam2_min_mask_pixels", 100)))
    if not ckpt or not model_cfg:
        print("[ERROR] checkpoint/model_cfg missing (config or --sam2_ckpt/--sam2_cfg).",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(ckpt):
        print(f"[ERROR] checkpoint not found: {ckpt}\n"
              f"        download it into .docker_assets/sam2_checkpoints/.",
              file=sys.stderr)
        sys.exit(1)

    labeler = VideoLabeler(args.video_dir, model_cfg, ckpt,
                           no_vis=args.no_vis, min_px=min_px)
    try:
        if args.bbox is not None:
            run_headless(labeler, args.bbox)
        else:
            if not args.gui and not os.environ.get("DISPLAY"):
                print("[ERROR] no DISPLAY and no --bbox. Set up X11 for the GUI, or "
                      "pass --bbox x0 y0 x1 y1 for headless.", file=sys.stderr)
                sys.exit(1)
            run_gui(labeler)
    finally:
        labeler.close()


if __name__ == "__main__":
    main()
