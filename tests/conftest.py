import json
import math

import numpy as np
import pytest
from PIL import Image


def _look_at_origin(theta: float, radius: float = 4.0) -> list[list[float]]:
    """Camera-to-world matrix on a circle at z=1, looking at the origin (OpenGL convention)."""
    pos = np.array([radius * math.cos(theta), radius * math.sin(theta), 1.0])
    back = pos / np.linalg.norm(pos)  # camera +z points away from the target
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, back, pos
    return c2w.tolist()


@pytest.fixture
def tiny_scene(tmp_path):
    """A 3-view, 16x16 Blender-style scene: a red disc on a transparent background."""
    root = tmp_path / "scene"
    yy, xx = np.mgrid[0:16, 0:16]
    disc = (xx - 7.5) ** 2 + (yy - 7.5) ** 2 < 25
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        frames = []
        for k in range(3):
            arr = np.zeros((16, 16, 4), np.uint8)
            arr[disc] = [200, 40, 40, 255]
            Image.fromarray(arr, "RGBA").save(root / split / f"r_{k}.png")
            frames.append({"file_path": f"./{split}/r_{k}",
                           "transform_matrix": _look_at_origin(2 * math.pi * k / 3)})
        with open(root / f"transforms_{split}.json", "w") as f:
            json.dump({"camera_angle_x": 0.69, "frames": frames}, f)
    return str(root)


@pytest.fixture
def tiny_cfg(tiny_scene):
    return {
        "seed": 0,
        "data": {"root": tiny_scene, "height": 16, "width": 16, "focal": "AUTO",
                 "white_bkgd": True},
        "model": {
            "near": 2.0, "far": 6.0,
            "coarse": {"n_layers": 2, "hidden_dim": 32, "depth_samples": 8},
            "fine": {"n_layers": 2, "hidden_dim": 32, "depth_samples": 8},
        },
        "training": {"batch_rays": 64, "lr": 5e-4, "max_steps": 20, "log_interval": 1,
                     "val_interval": 1000, "val_indices": [0, 1], "save_interval": 5,
                     "keep_ckpts": 3},
    }
