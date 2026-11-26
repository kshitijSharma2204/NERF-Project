"""Reproduce the v1 numbers from the four saved v1 renders (outputs/150.png, ...).

v1 ground-truth protocol (what v1's loader did): the 800x800 RGBA test frame is
converted with convert("RGB"), which drops alpha and leaves a black background,
then LANCZOS-resized to 400x400. Metrics as in scripts/metrics.py. These numbers
are NOT comparable to the NeRF paper (800x800, white background, all 200 views).

    python -m scripts.evaluate_saved_renders
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from scripts.data_loader import load_image
from scripts.metrics import psnr, ssim

V1_VIEWS = (150, 162, 178, 193)


def evaluate_saved(render_dir: str = "outputs", data_root: str = "data/synthetic/lego",
                   views: tuple[int, ...] = V1_VIEWS) -> dict:
    per_view = {}
    for v in views:
        pred_img = Image.open(os.path.join(render_dir, f"{v}.png")).convert("RGB")
        W, H = pred_img.size
        pred = np.asarray(pred_img, dtype=np.float32) / 255.0
        gt = load_image(os.path.join(data_root, "test", f"r_{v}.png"), H, W, white_bkgd=False)
        per_view[f"r_{v}"] = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
    return {
        "per_view": per_view,
        "mean_psnr": float(np.mean([m["psnr"] for m in per_view.values()])),
        "mean_ssim": float(np.mean([m["ssim"] for m in per_view.values()])),
        "protocol": "v1: 4 saved renders, 400x400, GT convert('RGB') -> black background, "
                    "LANCZOS resize",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--render_dir", default="outputs")
    p.add_argument("--data_root", default="data/synthetic/lego")
    p.add_argument("--json", default=None, help="optional path to write the results as JSON")
    args = p.parse_args()
    res = evaluate_saved(args.render_dir, args.data_root)
    print(f"{'view':<6} {'PSNR':>9} {'SSIM':>7}")
    for name, m in res["per_view"].items():
        print(f"{name:<6} {m['psnr']:6.2f} dB {m['ssim']:.4f}")
    print(f"{'mean':<6} {res['mean_psnr']:6.2f} dB {res['mean_ssim']:.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
