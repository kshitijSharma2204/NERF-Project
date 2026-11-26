import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from scripts.data_loader import NeRFDataset
from scripts.occupancy import OccupancyGrid
from scripts.render_rays import render_image
from scripts.utils import build_models, configure_fp32, get_device, load_config, rays_for_pose


def load_models_from_ckpt(cfg: dict, ckpt_path: str, device: torch.device):
    coarse, fine, peXYZ, peDir = build_models(cfg, device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--expname", default=None, help="unused; kept for run_lego.sh compatibility")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()

    configure_fp32()
    cfg = load_config(args.config)
    device = get_device()
    ds = NeRFDataset(cfg, split=args.split)
    coarse, fine, peXYZ, peDir, ck = load_models_from_ckpt(cfg, args.ckpt, device)
    occ = load_occupancy_from_ckpt(cfg, ck, device)
    render_poses(cfg, (coarse, fine, peXYZ, peDir), ds.poses, ds.H, ds.W, ds.focal, device,
                 args.outdir, occ=occ)


if __name__ == "__main__":
    main()
