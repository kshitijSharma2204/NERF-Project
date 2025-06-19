#!/usr/bin/env python3
import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from scripts.data_loader import NeRFDataset
from scripts.render_rays import render_rays
from models.nerf import PosEncoding, NeRFMLP

def train(cfg, expname, resume):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up log dir
    logdir = os.path.join("logs", expname)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # 1) dataset
    ds = NeRFDataset(cfg, split="train")

    # 2) positional encoders (use defaults if missing)
    enc = cfg["model"].get("encoding", {})
    pos_freqs = enc.get("pos_freqs", 10)
    dir_freqs = enc.get("dir_freqs", 4)
    peXYZ = PosEncoding(pos_freqs)
    peDir = PosEncoding(dir_freqs)

    # 3) channel sizes
    input_ch = 3 + 2 * 3 * pos_freqs
    input_ch_dir = 3 + 2 * 3 * dir_freqs

    # 4) build MLPs (architecture only; use positional args to match signature)
    c_cfg = cfg["model"]["coarse"]
    f_cfg = cfg["model"]["fine"]
    coarse = NeRFMLP(c_cfg["n_layers"], c_cfg["hidden_dim"], input_ch, input_ch_dir)
    fine = NeRFMLP(f_cfg["n_layers"], f_cfg["hidden_dim"], input_ch, input_ch_dir)
    coarse = coarse.to(device)  # Move to GPU
    fine = fine.to(device)      # Move to GPU

    # 5) optimizer
    optimizer = optim.Adam(list(coarse.parameters()) + list(fine.parameters()), lr=float(cfg["training"]["lr"]))

    # 6) resume if requested
    if resume:
        last_ckpt = os.path.join(logdir, "checkpoint_last.pth")
        if os.path.isfile(last_ckpt):
            ck = torch.load(last_ckpt, map_location=device)  # Load to correct device
            coarse.load_state_dict(ck["coarse"])
            fine.load_state_dict(ck["fine"])
            print(f"Resumed from {last_ckpt}")

    # 7) training params
    max_steps = cfg["training"]["max_steps"]
    batch_rays = cfg["training"]["batch_rays"]
    log_interval = cfg["training"]["log_interval"]
    val_interval = cfg["training"]["val_interval"]
    save_interval = cfg["training"]["save_interval"]

    # 8) main loop
    for step in tqdm(range(max_steps), desc="Training", leave=True):
        # sample random rays
        img_i = random.randrange(len(ds.images))
        rays_o, rays_d, target = ds.get_rays(img_i)
        # flatten and pick batch
        H, W = ds.H, ds.W
        rays_o = torch.from_numpy(rays_o.reshape(-1, 3).copy()).float()
        rays_d = torch.from_numpy(rays_d.reshape(-1, 3).copy()).float()
        target = torch.from_numpy(target.reshape(-1, 3).copy()).float()
        idxs = torch.randint(0, H * W, (batch_rays,), dtype=torch.long)
        rays_o, rays_d, target = rays_o[idxs], rays_d[idxs], target[idxs]
        rays_o, rays_d, target = rays_o.to(device), rays_d.to(device), target.to(device)  # Move to GPU

        # render
        rgb_coarse, rgb_fine = render_rays(coarse, fine, peXYZ, peDir, rays_o, rays_d, cfg)

        # loss & step
        loss = (torch.nan_to_num((rgb_coarse - target)**2, nan=0.0).mean() + 
                torch.nan_to_num((rgb_fine - target)**2, nan=0.0).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        if step % log_interval == 0:
          writer.add_scalar("loss/train", loss.item(), step)
          tqdm.write(f"Step {step}, Loss: {loss.item():.4f}, rgb_coarse min/max: {rgb_coarse.min().item():.4f}/{rgb_coarse.max().item():.4f}")

        # optional val & checkpoint
        if step and step % save_interval == 0:
            ck = {"coarse": coarse.state_dict(), "fine": fine.state_dict()}
            torch.save(ck, os.path.join(logdir, f"checkpoint_{step}.pth"))
            torch.save(ck, os.path.join(logdir, "checkpoint_last.pth"))

    writer.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--expname", default="run1")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    train(cfg, args.expname, args.resume)