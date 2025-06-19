import torch
import torch.nn.functional as F

def raw2outputs(rgb, sigma, z_vals, rays_d, white_bkgd=False):
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10)], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    alpha = 1 - torch.exp(-sigma * dists)
    trans = torch.cumprod(torch.cat([
        torch.ones((alpha.shape[0],1), device=alpha.device),
        1-alpha + 1e-10
    ], -1), -1)[...,:-1]
    weights = alpha * trans
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    if white_bkgd:
        rgb_map = rgb_map + (1 - weights.sum(-1, keepdim=True))
    return rgb_map, weights

def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=bins.device)
        u = u.expand(cdf.shape[0], N_samples)
    else:
        u = torch.rand(cdf.shape[0], N_samples, device=bins.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds-1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], -1)
    cdf_g = torch.gather(
        cdf.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g
    )
    bins_g = torch.gather(
        bins.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g
    )
    denom = (cdf_g[...,1] - cdf_g[...,0])
    denom[denom<1e-5] = 1
    t = (u - cdf_g[...,0]) / denom
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])
    return samples

def render_rays(coarse_model, fine_model, peXYZ, peDir, rays_o, rays_d, cfg):
    device = rays_o.device
    B = rays_o.shape[0]
    n_coarse = cfg['model']['coarse']['depth_samples']
    n_fine   = cfg['model']['fine']['depth_samples']
    near = cfg['model'].get('near', 2.0)
    far  = cfg['model'].get('far', 6.0)

    # Coarse stratified sampling
    z_vals = torch.linspace(near, far, n_coarse, device=device).unsqueeze(0).repeat(B,1)
    mids = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    t_rand = torch.rand(z_vals.shape, device=device)
    z_vals = lower + (upper-lower) * t_rand

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    pts_flat = pts.reshape(-1,3)
    emb_pts = peXYZ(pts_flat)

    emb_dirs = peDir(rays_d)
    emb_dirs = emb_dirs.unsqueeze(1).expand(-1, n_coarse, -1).reshape(-1,emb_dirs.shape[-1])

    rgb_sigma = coarse_model(emb_pts, emb_dirs)
    rgb_c   = rgb_sigma[0].reshape(B, n_coarse, 3)
    sigma_c = rgb_sigma[1].reshape(B, n_coarse)

    rgb_coarse, weights = raw2outputs(rgb_c, sigma_c, z_vals, rays_d, white_bkgd=True)

    # Fine sampling
    z_vals_mid = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], n_fine, det=False)
    z_vals_all = torch.cat([z_vals, z_samples], -1)
    z_vals_all, _ = torch.sort(z_vals_all, -1)

    pts_f = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_all.unsqueeze(-1)
    pts_f_flat = pts_f.reshape(-1,3)
    emb_pts_f = peXYZ(pts_f_flat)

    emb_dirs_f = peDir(rays_d).unsqueeze(1).expand(-1, z_vals_all.shape[-1], -1).reshape(-1, emb_dirs.shape[-1])
    
    rgb_sigma_f = fine_model(emb_pts_f, emb_dirs_f)
    rgb_f   = rgb_sigma_f[0].reshape(B, z_vals_all.shape[-1], 3)
    sigma_f = rgb_sigma_f[1].reshape(B, z_vals_all.shape[-1])

    rgb_fine, _ = raw2outputs(rgb_f, sigma_f, z_vals_all, rays_d, white_bkgd=True)

    return rgb_coarse, rgb_fine
