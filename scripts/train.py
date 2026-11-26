#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import time

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scripts.checkpoint import capture_rng, find_last_ckpt_and_step, restore_rng, save_checkpoint
from scripts.data_loader import NeRFDataset
from scripts.render_rays import render_rays
from scripts.utils import build_models, configure_fp32, get_device, git_sha, set_seed


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Return a copy of cfg with dotted overrides applied, e.g. 'training.max_steps=1000'."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def truncate_metrics(path: str, start_step: int) -> None:
    """Drop metric lines at or after start_step (all of them for a fresh run)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        keep = [ln for ln in f if ln.strip() and json.loads(ln)["step"] < start_step]
    with open(path, "w") as f:
        f.writelines(keep)


def train(cfg: dict, expname: str, resume: bool, logroot: str = "logs") -> dict:
    """Train coarse+fine NeRF. Returns a summary dict (final step, last loss, logdir)."""
    configure_fp32()
    device = get_device()
    tcfg = cfg["training"]
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    logdir = os.path.join(logroot, expname)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    metrics_path = os.path.join(logdir, "metrics.jsonl")

    ds = NeRFDataset(cfg, split="train")
    coarse, fine, peXYZ, peDir = build_models(cfg, device)
    params = list(coarse.parameters()) + list(fine.parameters())
    optimizer = torch.optim.Adam(params, lr=float(tcfg["lr"]))

    start_step = 0
    if resume:
        ck, last_step, ck_path = find_last_ckpt_and_step(logdir, device)
        if ck is not None:
            coarse.load_state_dict(ck["coarse"])
            fine.load_state_dict(ck["fine"])
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            else:
                print("[WARN] checkpoint has no optimizer state (old format); Adam restarts")
            if "rng" in ck:
                restore_rng(ck["rng"])
            if ck.get("config") not in (None, cfg):
                print("[WARN] config differs from the one stored in the checkpoint")
            start_step = last_step + 1
            print(f"Resuming from step {start_step} (loaded {os.path.basename(ck_path)})")
        else:
            print("No checkpoint found; starting from scratch")
    truncate_metrics(metrics_path, start_step)

    max_steps = int(tcfg["max_steps"])
    batch_rays = int(tcfg["batch_rays"])
    log_interval = int(tcfg["log_interval"])
    save_interval = int(tcfg["save_interval"])
    keep = int(tcfg.get("keep_ckpts", 3))

    def checkpoint(step: int) -> str:
        return save_checkpoint(logdir, {
            "coarse": coarse.state_dict(),
            "fine": fine.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_state": {"lr": optimizer.param_groups[0]["lr"]},
            "step": step,
            "rng": capture_rng(),
            "config": cfg,
            "git_sha": git_sha(),
        }, keep=keep)

    H, W = ds.H, ds.W
    loss_val = float("nan")
    step = start_step - 1
    t_last = time.perf_counter()
    for step in tqdm(range(start_step, max_steps), desc="Training", initial=start_step,
                     total=max_steps):
        img_i = random.randrange(len(ds.images))
        rays_o, rays_d, target = ds.get_rays(img_i)
        rays_o = torch.from_numpy(rays_o.reshape(-1, 3).copy()).float()
        rays_d = torch.from_numpy(rays_d.reshape(-1, 3).copy()).float()
        target = torch.from_numpy(target.reshape(-1, 3).copy()).float()
        idxs = torch.randint(0, H * W, (batch_rays,), dtype=torch.long)
        rays_o, rays_d, target = rays_o[idxs], rays_d[idxs], target[idxs]
        rays_o, rays_d, target = rays_o.to(device), rays_d.to(device), target.to(device)

        rgb_coarse, rgb_fine = render_rays(coarse, fine, peXYZ, peDir, rays_o, rays_d, cfg,
                                           perturb=True)
        mse_f = ((rgb_fine - target) ** 2).mean()
        loss = ((rgb_coarse - target) ** 2).mean() + mse_f
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            loss_val = loss.item()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss_val} at step {step}")
            now = time.perf_counter()
            n = log_interval if step > start_step else 1
            ms = 1000.0 * (now - t_last) / n
            t_last = now
            psnr = -10.0 * torch.log10(mse_f).item()
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("loss/train", loss_val, step)
            writer.add_scalar("psnr/train", psnr, step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("time/ms_per_step", ms, step)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": loss_val, "psnr_train": psnr,
                                    "lr": lr, "ms_per_step": ms}) + "\n")
            tqdm.write(f"step {step} loss {loss_val:.5f} psnr {psnr:.2f} lr {lr:.2e} "
                       f"{ms:.1f} ms/step")

        if step and step % save_interval == 0:
            checkpoint(step)

    final = max_steps - 1
    if step == final and final % save_interval != 0:
        path = checkpoint(final)
        print(f"[INFO] Saved final checkpoint at step {final}: {path}")
    writer.close()
    return {"step": step, "loss": loss_val, "logdir": logdir}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--expname", default="run1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set training.max_steps=1000 seed=1")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = apply_overrides(yaml.safe_load(f), args.set)
    train(cfg, args.expname, args.resume)


if __name__ == "__main__":
    main()
