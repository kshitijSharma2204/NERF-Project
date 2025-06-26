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

import re, glob
def find_last_ckpt_and_step(logdir, device):
    # 1) Glob only the numeric checkpoints
    pattern = os.path.join(logdir, "checkpoint_*.pth")
    all_ckpts = glob.glob(pattern)
    # 2) Exclude checkpoint_last.pth
    num_ckpts = [p for p in all_ckpts 
                 if re.match(r".*checkpoint_\d+\.pth$", os.path.basename(p))]
    if not num_ckpts:
        return None, 0, None
    # 3) Build (step, path) pairs
    pairs = []
    for p in num_ckpts:
        m = re.search(r"checkpoint_(\d+)\.pth$", os.path.basename(p))
        if m:
            pairs.append((int(m.group(1)), p))
    # 4) Find the max step
    last_step, last_path = max(pairs, key=lambda x: x[0])
    # 5) Load it
    ck = torch.load(last_path, map_location=device)
    return ck, last_step, last_path

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
    start_step = 0
    if resume:
        ck, last_step, ck_path = find_last_ckpt_and_step(logdir, device)
        if ck is not None:
            coarse.load_state_dict(ck["coarse"])
            fine.load_state_dict(ck["fine"])
            start_step = last_step + 1
            print(f"Resuming from iteration {start_step} (loaded  {os.path.basename(ck_path)})")
        else:
            print("No checkpoint_* found; starting from scratch")

    # 7) training params
    max_steps = cfg["training"]["max_steps"]
    batch_rays = cfg["training"]["batch_rays"]
    log_interval = cfg["training"]["log_interval"]
    val_interval = cfg["training"]["val_interval"]
    save_interval = cfg["training"]["save_interval"]

    # 8) main loop: start from wherever we left off
    for step in tqdm(range(start_step, max_steps), desc="Training", leave=True):
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
            # include the iteration counter in every save
            ck = {"coarse": coarse.state_dict(), "fine": fine.state_dict(), "step": step}
            torch.save(ck, os.path.join(logdir, f"checkpoint_{step}.pth"))
    

    # ensure we save the very last step even if it's not on a save_interval
    final_step = max_steps - 1
    # only if we didn’t already save it
    if final_step % save_interval != 0:
        ck = {"coarse": coarse.state_dict(),
              "fine":   fine.state_dict(),
              "step":   final_step}
        path = os.path.join(logdir, f"checkpoint_{final_step}.pth")
        torch.save(ck, path)
        print(f"[INFO] Saved final checkpoint at step {final_step}: {path}")
            

    writer.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--expname", default="run1")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    train(cfg, args.expname, args.resume)