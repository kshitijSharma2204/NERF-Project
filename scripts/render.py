import argparse
import yaml
import os
import torch
from PIL import Image
from scripts.data_loader import NeRFDataset
from models.nerf import PosEncoding, NeRFMLP
from scripts.render_rays import render_rays

def render_scene(cfg, expname, ckpt, split, outdir):
    os.makedirs(outdir, exist_ok=True)
    ds = NeRFDataset(cfg, split=split)
    # build models & encoders
    peXYZ = PosEncoding(cfg["model"]["encoding"]["pos_freqs"])
    peDir = PosEncoding(cfg["model"]["encoding"]["dir_freqs"])
    input_ch     = 3 + 2*3*cfg["model"]["encoding"]["pos_freqs"]
    input_ch_dir = 3 + 2*3*cfg["model"]["encoding"]["dir_freqs"]
    coarse = NeRFMLP(**cfg["model"]["coarse"], input_ch=input_ch, input_ch_dir=input_ch_dir)
    fine   = NeRFMLP(**cfg["model"]["fine"],   input_ch=input_ch, input_ch_dir=input_ch_dir)

    # load checkpoint
    ckpt_dict = torch.load(ckpt)
    coarse.load_state_dict(ckpt_dict["coarse"])
    fine.load_state_dict(ckpt_dict["fine"])
    coarse.cuda(); fine.cuda()

    with torch.no_grad():
        for i in range(len(ds.images)):
            rays_o, rays_d, _ = ds.get_rays(i)
            rays_o = torch.from_numpy(rays_o).cuda()
            rays_d = torch.from_numpy(rays_d).cuda()
            _, rgb = render_rays(coarse, fine, peXYZ, peDir, rays_o, rays_d, cfg)
            img = rgb.cpu().numpy().reshape(ds.H, ds.W, 3)
            Image.fromarray((img * 255).astype('uint8')).save(os.path.join(outdir, f"{i:03d}.png"))

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