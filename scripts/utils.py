"""Shared helpers for training, rendering and evaluation."""
import os
import random
import subprocess

import numpy as np
import torch
import yaml

from models.nerf import NeRFMLP, PosEncoding


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """CUDA if available; NERF_DEVICE=cpu forces CPU (used by exact-reproducibility tests)."""
    forced = os.environ.get("NERF_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_fp32() -> None:
    """Full fp32 math: disable TF32 for matmuls and cuDNN."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def git_sha() -> str:
    """Commit SHA of the working tree, with a '-dirty' suffix if there are uncommitted edits."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "scripts", "models", "configs"], cwd=root
        ).returncode
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_models(cfg: dict, device: torch.device):
    """Return (coarse, fine, peXYZ, peDir) built from the config, on device."""
    enc = cfg["model"].get("encoding", {})
    pos_freqs = enc.get("pos_freqs", 10)
    dir_freqs = enc.get("dir_freqs", 4)
    peXYZ = PosEncoding(pos_freqs)
    peDir = PosEncoding(dir_freqs)
    input_ch = 3 + 2 * 3 * pos_freqs
    input_ch_dir = 3 + 2 * 3 * dir_freqs
    c_cfg = cfg["model"]["coarse"]
    f_cfg = cfg["model"]["fine"]
    coarse = NeRFMLP(c_cfg["n_layers"], c_cfg["hidden_dim"], input_ch, input_ch_dir).to(device)
    fine = NeRFMLP(f_cfg["n_layers"], f_cfg["hidden_dim"], input_ch, input_ch_dir).to(device)
    return coarse, fine, peXYZ, peDir


def rays_for_pose(
    H: int, W: int, focal: float, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """All rays of one camera on c2w's device, flattened to (H*W, 3) each."""
    j, i = torch.meshgrid(
        torch.arange(H, device=c2w.device, dtype=torch.float32),
        torch.arange(W, device=c2w.device, dtype=torch.float32),
        indexing="ij",
    )
    return rays_at_pixels(i.reshape(-1), j.reshape(-1), H, W, focal, c2w)


def rays_at_pixels(
    i: torch.Tensor, j: torch.Tensor, H: int, W: int, focal: float, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rays through pixel columns i and rows j (same convention as NeRFDataset.get_rays)."""
    dirs = torch.stack([(i - 0.5 * W) / focal, -(j - 0.5 * H) / focal, -torch.ones_like(i)], -1)
    rays_d = dirs @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d
