import json
import os

import numpy as np
from PIL import Image


def load_image(path: str, height: int, width: int, white_bkgd: bool) -> np.ndarray:
    """Load one Blender-synthetic frame as float32 RGB in [0, 1], shape (H, W, 3).

    white_bkgd=True (v2): load RGBA, resize RGBA with LANCZOS, then composite onto
    white with ``rgb * a + (1 - a)``. Pillow premultiplies alpha while resampling
    RGBA, so object colours do not bleed into the transparent border.

    white_bkgd=False (v1 behaviour, kept for reproducing v1 numbers):
    ``convert("RGB")`` drops alpha. Transparent pixels store RGB 0, so the
    background becomes black. The image is then resized with LANCZOS.
    """
    img = Image.open(path)
    if white_bkgd:
        rgba = img.convert("RGBA").resize((width, height), Image.LANCZOS)
        arr = np.asarray(rgba, dtype=np.float32) / 255.0
        rgb, alpha = arr[..., :3], arr[..., 3:]
        return rgb * alpha + (1.0 - alpha)
    rgb = img.convert("RGB").resize((width, height), Image.LANCZOS)
    return np.asarray(rgb, dtype=np.float32) / 255.0


class NeRFDataset:
    def __init__(self, cfg: dict, split: str = "train", indices: list[int] | None = None):
        data_cfg = cfg["data"]
        root = data_cfg["root"]  # e.g. "./data/synthetic/lego"
        H = data_cfg["height"]
        W = data_cfg["width"]
        self.white_bkgd = bool(data_cfg.get("white_bkgd", True))
        with open(os.path.join(root, f"transforms_{split}.json")) as f:
            meta = json.load(f)

        # Focal length (AUTO uses camera_angle_x from the JSON)
        if data_cfg["focal"] == "AUTO":
            self.focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])
        else:
            self.focal = float(data_cfg["focal"])
        self.H, self.W = H, W

        frames = meta["frames"]
        if indices is not None:
            frames = [frames[i] for i in indices]
        self.indices = list(indices) if indices is not None else list(range(len(frames)))
        self.names = [os.path.basename(fr["file_path"]) for fr in frames]

        images, poses = [], []
        for frame in frames:
            img_path = os.path.normpath(os.path.join(root, frame["file_path"] + ".png"))
            images.append(load_image(img_path, H, W, self.white_bkgd))
            poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

        self.images = np.stack(images)  # (N, H, W, 3)
        self.poses = np.stack(poses)  # (N, 4, 4)

    def get_rays(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          rays_o: (H, W, 3) origins
          rays_d: (H, W, 3) directions
          rgb:    (H, W, 3) ground-truth colors
        """
        c2w = self.poses[idx]
        i, j = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing="xy")
        dirs = np.stack(
            [
                (i - 0.5 * self.W) / self.focal,
                -(j - 0.5 * self.H) / self.focal,
                -np.ones_like(i),
            ],
            axis=-1,
        )  # (H, W, 3)

        # world-space rays
        rays_d = (dirs[..., None, :] @ c2w[:3, :3].T).squeeze(-2)
        rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
        rgb = self.images[idx]
        return rays_o, rays_d, rgb
