import argparse
import yaml
import os
import torch
from PIL import Image
from scripts.data_loader import NeRFDataset
from models.nerf import PosEncoding, NeRFMLP
from scripts.render_rays import render_rays
from tqdm import tqdm

def render_scene(cfg, expname, ckpt, split, outdir):
    os.makedirs(outdir, exist_ok=True)
    ds = NeRFDataset(cfg, split=split)
    # build models & encoders
    enc = cfg["model"].get("encoding", {})
    pos_freqs = enc.get("pos_freqs", 10)
    dir_freqs = enc.get("dir_freqs", 4)

    peXYZ = PosEncoding(pos_freqs)
    peDir = PosEncoding(dir_freqs)
    
    input_ch     = 3 + 2*3*pos_freqs
    input_ch_dir = 3 + 2*3*dir_freqs

    c_cfg = cfg["model"]["coarse"]
    f_cfg = cfg["model"]["fine"]
    coarse = NeRFMLP(
        n_layers    = c_cfg["n_layers"],
        hidden_dim  = c_cfg["hidden_dim"],
        input_ch    = input_ch,
        input_ch_dir= input_ch_dir,
    )
    fine = NeRFMLP(
        n_layers    = f_cfg["n_layers"],
        hidden_dim  = f_cfg["hidden_dim"],
        input_ch    = input_ch,
        input_ch_dir= input_ch_dir,
    )

    # load checkpoint
    ckpt_dict = torch.load(ckpt)
    coarse.load_state_dict(ckpt_dict["coarse"])
    fine.load_state_dict(ckpt_dict["fine"])
    coarse.cuda(); fine.cuda()

    # render in small chunks to avoid OOM
    chunk_size = cfg.get("rendering", {}).get("chunk_size", 4096)
    with torch.no_grad():
        for i in tqdm(range(len(ds.images)), desc="Rendering", unit="img", leave=True):
            rays_o_np, rays_d_np, _ = ds.get_rays(i)
            H, W = ds.H, ds.W
            rays_o = (
                torch.from_numpy(rays_o_np.reshape(-1, 3).copy())
                .float()
                .cuda()
            )
            rays_d = (
                torch.from_numpy(rays_d_np.reshape(-1, 3).copy())
                .float()
                .cuda()
            )

            rgb_chunks = []
            for start in range(0, rays_o.shape[0], chunk_size):
                end = start + chunk_size
                o_chunk = rays_o[start:end]
                d_chunk = rays_d[start:end]
                _, rgb_chunk = render_rays(coarse, fine, peXYZ, peDir,
                                           o_chunk, d_chunk, cfg)
                rgb_chunks.append(rgb_chunk)

            rgb = torch.cat(rgb_chunks, dim=0)
            img = rgb.cpu().numpy().reshape(H, W, 3)
            Image.fromarray((img * 255).astype('uint8')) \
                 .save(os.path.join(outdir, f"{i:03d}.png"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--expname", required=True)
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--split",   default="test")
    p.add_argument("--outdir",  default="outputs")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    render_scene(cfg, args.expname, args.ckpt, args.split, args.outdir)