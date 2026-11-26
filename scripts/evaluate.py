"""Evaluate a checkpoint on every view of a split (default: all 200 test views).

Renders deterministically (perturb=False, with the checkpoint's occupancy grid if it
has one), saves 8-bit PNGs to <outdir>/test/NNN.png and computes PSNR and SSIM of
those 8-bit renders against white-composited ground truth at the configured
resolution. Writes <outdir>/eval/metrics.json and <outdir>/eval/per_view.csv.

    python -m scripts.evaluate --config configs/synthetic_lego.yaml \
        --ckpt logs/lego_v2/checkpoint_last.pth --outdir outputs/v2
"""
import argparse
import csv
import json
import os
import platform

import numpy as np
import torch

from scripts.data_loader import NeRFDataset
from scripts.metrics import psnr, ssim
from scripts.render import (
    _check_outdir,
    load_models_from_ckpt,
    load_occupancy_from_ckpt,
    render_poses,
)
from scripts.utils import configure_fp32, get_device, git_sha, load_config

PROTOCOL = (
    "Deterministic rendering (no jitter, deterministic fine sampling). Metrics on the "
    "8-bit PNG renders vs ground truth loaded as RGBA, LANCZOS-resized to the configured "
    "resolution and composited onto white. PSNR = -10 log10(MSE) on [0,1] RGB. SSIM: "
    "11x11 Gaussian window, sigma 1.5, C1 = 0.01^2, C2 = 0.03^2, valid convolution, "
    "averaged over RGB channels and pixels. Std is the population std over views."
)


def _summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()),
            "max": float(arr.max())}


def evaluate(cfg: dict, ckpt_path: str, outdir: str, split: str = "test",
             limit: int | None = None) -> dict:
    configure_fp32()
    device = get_device()
    if not cfg["data"].get("white_bkgd", True):
        raise ValueError("v2 evaluation uses white-composited ground truth (data.white_bkgd)")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    indices = list(range(limit)) if limit is not None else None
    ds = NeRFDataset(cfg, split=split, indices=indices)
    coarse, fine, peXYZ, peDir, ck = load_models_from_ckpt(cfg, ckpt_path, device)
    occ = load_occupancy_from_ckpt(cfg, ck, device)

    render_dir = os.path.join(outdir, split)
    eval_dir = os.path.join(outdir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    names = [f"{k:03d}" for k in ds.indices]
    frames = render_poses(cfg, (coarse, fine, peXYZ, peDir), ds.poses, ds.H, ds.W, ds.focal,
                          device, render_dir, desc=f"Evaluating {split}", occ=occ, names=names)

    rows = []
    for k, (img, secs) in enumerate(frames):
        pred = img.astype(np.float32) / 255.0
        gt = ds.images[k]
        rows.append({"index": ds.indices[k], "name": ds.names[k], "psnr": psnr(pred, gt),
                     "ssim": ssim(pred, gt), "render_seconds": secs})

    with open(os.path.join(eval_dir, "per_view.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "name", "psnr", "ssim", "render_seconds"])
        w.writeheader()
        w.writerows(rows)

    times = [r["render_seconds"] for r in rows]
    metrics = {
        "split": split,
        "n_views": len(rows),
        "psnr": {**_summary([r["psnr"] for r in rows]),
                 "per_view": {r["name"]: r["psnr"] for r in rows}},
        "ssim": {**_summary([r["ssim"] for r in rows]),
                 "per_view": {r["name"]: r["ssim"] for r in rows}},
        "resolution": [ds.H, ds.W],
        "background": "white",
        "protocol": PROTOCOL,
        "checkpoint": os.path.abspath(ckpt_path),
        "checkpoint_step": int(ck.get("step", -1)),
        "checkpoint_git_sha": ck.get("git_sha", "unknown"),
        "eval_git_sha": git_sha(),
        "occupancy_grid": occ is not None,
        "render_seconds_per_frame": {"mean": float(np.mean(times)),
                                     "median": float(np.median(times))},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "config": cfg,
    }
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--outdir", default="outputs/v2")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None, help="only the first N views (smoke test)")
    args = p.parse_args()
    _check_outdir(args.outdir)
    m = evaluate(load_config(args.config), args.ckpt, args.outdir, args.split, args.limit)
    print(f"{m['n_views']} {m['split']} views at {m['resolution'][0]}x{m['resolution'][1]}, "
          f"step {m['checkpoint_step']}: PSNR {m['psnr']['mean']:.2f} +/- {m['psnr']['std']:.2f} "
          f"dB, SSIM {m['ssim']['mean']:.4f} +/- {m['ssim']['std']:.4f}")


if __name__ == "__main__":
    main()
