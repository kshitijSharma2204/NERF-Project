import argparse
import os
import subprocess
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from scripts.data_loader import NeRFDataset
from scripts.occupancy import OccupancyGrid
from scripts.render_rays import render_image
from scripts.utils import (
    build_models,
    configure_fp32,
    get_device,
    load_config,
    model_config_mismatch,
    rays_for_pose,
)


def load_models_from_ckpt(cfg: dict, ckpt_path: str, device: torch.device):
    coarse, fine, peXYZ, peDir = build_models(cfg, device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    bad = model_config_mismatch(cfg, ck.get("config"))
    if bad:
        raise SystemExit(f"config differs from the checkpoint's in {bad}; render with the "
                         "config the model was trained with")
    coarse.load_state_dict(ck["coarse"])
    fine.load_state_dict(ck["fine"])
    coarse.eval()
    fine.eval()
    return coarse, fine, peXYZ, peDir, ck


def load_occupancy_from_ckpt(cfg: dict, ck: dict, device: torch.device) -> OccupancyGrid | None:
    """The trained grid is part of the model: use it when the checkpoint has one."""
    if "occupancy" not in ck:
        return None
    m = cfg["model"]
    delta = (m.get("far", 6.0) - m.get("near", 2.0)) / m["coarse"]["depth_samples"]
    occ = OccupancyGrid(cfg.get("occupancy", {}), delta, device)
    occ.load_state_dict(ck["occupancy"])
    return occ


def to_uint8(rgb: torch.Tensor, H: int, W: int) -> np.ndarray:
    """(H*W,3) float in [0,1] -> (H,W,3) uint8, rounding to nearest."""
    img = rgb.clamp(0.0, 1.0).reshape(H, W, 3).cpu().numpy()
    return np.round(img * 255.0).astype(np.uint8)


def render_poses(cfg, models, poses, H, W, focal, device, outdir=None, desc="Rendering",
                 occ=None, names=None):
    """Render each pose; returns [(uint8 image, seconds)]. Saves NNN.png if outdir is set."""
    coarse, fine, peXYZ, peDir = models
    chunk = cfg.get("render", {}).get("chunk_size", 4096)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    frames = []
    for k, pose in enumerate(tqdm(poses, desc=desc, unit="img")):
        t0 = time.perf_counter()
        c2w = torch.as_tensor(pose, dtype=torch.float32, device=device)
        rays_o, rays_d = rays_for_pose(H, W, focal, c2w)
        rgb = render_image(coarse, fine, peXYZ, peDir, rays_o, rays_d, cfg, chunk=chunk, occ=occ)
        img = to_uint8(rgb, H, W)  # .cpu() inside synchronises, so the timing is complete
        secs = time.perf_counter() - t0
        if outdir:
            name = names[k] if names else f"{k:03d}"
            Image.fromarray(img).save(os.path.join(outdir, f"{name}.png"))
        frames.append((img, secs))
    return frames


def pose_spherical(theta_deg: float, phi_deg: float, radius: float) -> np.ndarray:
    """Camera-to-world matrix on a sphere around the origin (as in the original NeRF code)."""
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    trans = np.eye(4)
    trans[2, 3] = radius
    rot_phi = np.array([[1, 0, 0, 0], [0, np.cos(ph), -np.sin(ph), 0],
                        [0, np.sin(ph), np.cos(ph), 0], [0, 0, 0, 1]])
    rot_theta = np.array([[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0],
                          [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]])
    flip = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return (flip @ rot_theta @ rot_phi @ trans).astype(np.float32)


def spherical_path(n_frames: int, phi_deg: float = -30.0, radius: float = 4.0) -> np.ndarray:
    thetas = np.linspace(-180.0, 180.0, n_frames + 1)[:-1]
    return np.stack([pose_spherical(t, phi_deg, radius) for t in thetas])


def encode_video(frame_dir: str, mp4_path: str, gif_path: str | None, fps: int = 30,
                 gif_width: int = 320) -> None:
    """MP4 (H.264) and an optional small GIF (palette, downscaled) with ffmpeg."""
    pattern = os.path.join(frame_dir, "%03d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4_path], check=True)
    if gif_path:
        vf = (f"fps={fps // 2},scale={gif_width}:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                        pattern, "-vf", vf, gif_path], check=True)


def _check_outdir(outdir: str) -> None:
    """Never write renders into outputs/ itself: that is where the v1 results live."""
    if os.path.abspath(outdir) == os.path.abspath("outputs"):
        raise SystemExit("refusing to write into outputs/ (v1 artifacts); use outputs/v2/...")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--expname", default=None, help="unused; kept for run_lego.sh compatibility")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--path", choices=["split", "spherical"], default="split",
                   help="render the split's cameras, or a 360-degree fly-around")
    p.add_argument("--n_frames", type=int, default=120)
    p.add_argument("--video", default=None, help="with --path spherical: write this .mp4")
    p.add_argument("--gif", default=None, help="with --video: also write this small .gif")
    p.add_argument("--outdir", default="outputs/v2/renders")
    args = p.parse_args()
    _check_outdir(args.outdir)

    configure_fp32()
    cfg = load_config(args.config)
    device = get_device()
    ds = NeRFDataset(cfg, split=args.split)
    coarse, fine, peXYZ, peDir, ck = load_models_from_ckpt(cfg, args.ckpt, device)
    occ = load_occupancy_from_ckpt(cfg, ck, device)
    poses = spherical_path(args.n_frames) if args.path == "spherical" else ds.poses
    if args.video:  # ffmpeg reads every NNN.png: drop frames left by an earlier, longer render
        for name in os.listdir(args.outdir) if os.path.isdir(args.outdir) else []:
            if len(name) == 7 and name.endswith(".png") and name[:3].isdigit():
                os.remove(os.path.join(args.outdir, name))
    render_poses(cfg, (coarse, fine, peXYZ, peDir), poses, ds.H, ds.W, ds.focal, device,
                 args.outdir, occ=occ)
    if args.video:
        encode_video(args.outdir, args.video, args.gif)


if __name__ == "__main__":
    main()
