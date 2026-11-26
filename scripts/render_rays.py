import torch


def raw2outputs(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_bkgd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Volume-render per-sample (rgb (B,N,3), sigma (B,N)) at depths z_vals (B,N)."""
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    alpha = 1 - torch.exp(-sigma * dists)
    trans = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1 - alpha + 1e-10], -1),
        -1,
    )[..., :-1]
    weights = alpha * trans
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    if white_bkgd:
        rgb_map = rgb_map + (1 - weights.sum(-1, keepdim=True))
    return rgb_map, weights


def sample_pdf(
    bins: torch.Tensor, weights: torch.Tensor, N_samples: int, det: bool = False
) -> torch.Tensor:
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=bins.device)
        u = u.expand(cdf.shape[0], N_samples).contiguous()
    else:
        u = torch.rand(cdf.shape[0], N_samples, device=bins.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def _query(model, peXYZ, peDir, pts: torch.Tensor, rays_d: torch.Tensor):
    """Run the MLP on points pts (B,N,3) along rays with directions rays_d (B,3)."""
    B, N = pts.shape[:2]
    emb_pts = peXYZ(pts.reshape(-1, 3))
    emb_dirs = peDir(rays_d)
    emb_dirs = emb_dirs.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
    rgb, sigma = model(emb_pts, emb_dirs)
    return rgb.reshape(B, N, 3), sigma.reshape(B, N)


def render_rays(
    coarse_model,
    fine_model,
    peXYZ,
    peDir,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    cfg: dict,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a batch of rays (B,3) with the coarse and fine models.

    perturb=True (training): stratified jitter of coarse depths and random
    inverse-CDF fine samples. perturb=False (render/eval): coarse samples at bin
    edges and deterministic fine samples, so the same view renders identically.
    """
    device = rays_o.device
    B = rays_o.shape[0]
    n_coarse = cfg["model"]["coarse"]["depth_samples"]
    n_fine = cfg["model"]["fine"]["depth_samples"]
    near = cfg["model"].get("near", 2.0)
    far = cfg["model"].get("far", 6.0)

    # Coarse sampling: evenly spaced, stratified jitter when perturbing
    z_vals = torch.linspace(near, far, n_coarse, device=device).unsqueeze(0).expand(B, -1)
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    rgb_c, sigma_c = _query(coarse_model, peXYZ, peDir, pts, rays_d)
    rgb_coarse, weights = raw2outputs(rgb_c, sigma_c, z_vals, rays_d, white_bkgd=True)

    # Fine sampling (importance sampling from coarse weights; no gradient through it)
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1].detach(), n_fine, det=not perturb)
    z_vals_all, _ = torch.sort(torch.cat([z_vals, z_samples.detach()], -1), -1)

    pts_f = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_all.unsqueeze(-1)
    rgb_f, sigma_f = _query(fine_model, peXYZ, peDir, pts_f, rays_d)
    rgb_fine, _ = raw2outputs(rgb_f, sigma_f, z_vals_all, rays_d, white_bkgd=True)

    return rgb_coarse, rgb_fine


@torch.no_grad()
def render_image(
    coarse_model,
    fine_model,
    peXYZ,
    peDir,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    cfg: dict,
    chunk: int = 4096,
) -> torch.Tensor:
    """Deterministically render flattened rays (R,3) in chunks; returns fine RGB (R,3)."""
    out = []
    for start in range(0, rays_o.shape[0], chunk):
        _, rgb = render_rays(
            coarse_model, fine_model, peXYZ, peDir,
            rays_o[start:start + chunk], rays_d[start:start + chunk], cfg, perturb=False,
        )
        out.append(rgb)
    return torch.cat(out, 0)
