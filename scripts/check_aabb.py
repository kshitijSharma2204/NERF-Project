"""Check that an occupancy AABB contains the whole object (visual-hull test).

Points on a dense grid over a box larger than the AABB are kept only if every
training camera that sees them projects them onto a foreground pixel (alpha > 0,
dilated by a few pixels to be conservative). What survives is a superset of the
object (its visual hull). The AABB is safe if the hull's bounds lie inside it.
Points seen by fewer than --min_views cameras are ignored: so few views cannot carve
them, so they survive whether or not they are part of the object.

    python -m scripts.check_aabb --root data/synthetic/lego --aabb -1.5 1.5
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def visual_hull_bounds(root: str, search: float = 3.0, res: int = 160, dilate: int = 3,
                       min_views: int = 1, device: str = "cpu"
                       ) -> tuple[np.ndarray, np.ndarray, int]:
    """Bounds of points that >= min_views cameras see and every camera sees as foreground."""
    with open(os.path.join(root, "transforms_train.json")) as f:
        meta = json.load(f)
    lin = torch.linspace(-search, search, res, device=device)
    pts = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    keep = torch.ones(pts.shape[0], dtype=torch.bool, device=device)
    n_seen = torch.zeros(pts.shape[0], dtype=torch.int32, device=device)
    for fr in meta["frames"]:
        img = Image.open(os.path.join(root, fr["file_path"] + ".png"))
        alpha = torch.from_numpy(np.asarray(img)[..., 3] > 0).float().to(device)
        if dilate:
            alpha = F.max_pool2d(alpha[None, None], 2 * dilate + 1, 1, dilate)[0, 0]
        H, W = alpha.shape
        focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32, device=device)
        cam = (pts - c2w[:3, 3]) @ c2w[:3, :3]  # world -> camera (camera looks down -z)
        z = -cam[:, 2]
        u = focal * cam[:, 0] / z + 0.5 * W
        v = -focal * cam[:, 1] / z + 0.5 * H
        seen = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = u.clamp(0, W - 1).long()
        vi = v.clamp(0, H - 1).long()
        fg = alpha[vi, ui] > 0
        keep &= ~seen | fg  # carve points a camera sees as background
        n_seen += seen.int()
    hull = pts[keep & (n_seen >= min_views)]
    return hull.min(0).values.cpu().numpy(), hull.max(0).values.cpu().numpy(), len(hull)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/synthetic/lego")
    p.add_argument("--aabb", nargs=2, type=float, default=[-1.5, 1.5])
    p.add_argument("--res", type=int, default=160)
    p.add_argument("--min_views", type=int, default=10,
                   help="ignore points seen by fewer cameras: too few views to carve them")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lo, hi, n = visual_hull_bounds(args.root, res=args.res, min_views=args.min_views,
                                   device=device)
    step = 6.0 / (args.res - 1)
    print(f"visual hull (seen by >= {args.min_views} views): {n} points, "
          f"bounds min {np.round(lo, 3)} max {np.round(hi, 3)} (grid spacing {step:.3f})")
    margin = min((lo - args.aabb[0]).min(), (args.aabb[1] - hi).min())
    ok = margin > step
    print(f"AABB [{args.aabb[0]}, {args.aabb[1]}]^3: smallest margin {margin:.3f} -> "
          f"{'CONTAINS the object' if ok else 'TOO SMALL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
