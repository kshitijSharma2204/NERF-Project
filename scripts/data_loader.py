import os
import json
import numpy as np
from PIL import Image

class NeRFDataset:
    def __init__(self, cfg, split="train"):
        # 1) Config & paths
        root = cfg["data"]["root"]               # e.g. "./data/synthetic/lego"
        H    = cfg["data"]["height"]             # from synthetic_lego.yaml
        W    = cfg["data"]["width"]
        meta_path = os.path.join(root, f"transforms_{split}.json")
        meta = json.load(open(meta_path, "r"))

        # 2) Focal length (AUTO uses camera_angle_x from JSON)
        if cfg["data"]["focal"] == "AUTO":
            self.focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])
        else:
            self.focal = cfg["data"]["focal"]

        # 3) Image dimensions
        self.H, self.W = H, W

        # 4) Load all frames
        self.images = []
        self.poses  = []
        for frame in meta["frames"]:
          img_rel = frame["file_path"] + ".png"
          img_path = os.path.normpath(os.path.join(root, img_rel))
          # — load, force RGB, resize to (W,H)
          img_pil = Image.open(img_path).convert("RGB")
          img_pil = img_pil.resize((W, H), Image.LANCZOS)
          img = np.array(img_pil, dtype=np.float32) / 255.0  
          self.images.append(img)

          pose = np.array(frame["transform_matrix"], dtype=np.float32)
          self.poses.append(pose)

        # 5) Stack into arrays
        self.images = np.stack(self.images)  # (N, H, W, 3)
        self.poses  = np.stack(self.poses)   # (N, 4, 4)

    def get_rays(self, idx):
        """
        Returns:
          rays_o: (H, W, 3) origins
          rays_d: (H, W, 3) directions
          rgb:    (H, W, 3) ground-truth colors
        """
        c2w = self.poses[idx]
        i, j = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing="xy")
        dirs = np.stack([
            (i - 0.5 * self.W) / self.focal,
            -(j - 0.5 * self.H) / self.focal,
            -np.ones_like(i)
        ], axis=-1)  # (H, W, 3)

        # world-space rays
        rays_d = (dirs[..., None, :] @ c2w[:3, :3].T).squeeze(-2)
        rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
        rgb    = self.images[idx]
        return rays_o, rays_d, rgb