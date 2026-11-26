import copy
import json

import torch
import torch.nn as nn

from scripts.occupancy import OccupancyGrid
from scripts.render_rays import render_image
from scripts.train import apply_overrides, train
from scripts.utils import build_models, rays_for_pose, set_seed
from tests.test_render import CFG, POSE

CPU = torch.device("cpu")


class _Ball(nn.Module):
    """Density 100 inside a ball of radius 0.5 at the origin, 0 elsewhere."""

    def density(self, emb: torch.Tensor) -> torch.Tensor:
        xyz = emb[:, :3]  # PosEncoding keeps the raw input first
        return (xyz.norm(dim=-1, keepdim=True) < 0.5).float() * 100.0


def _grid(**kw) -> OccupancyGrid:
    cfg = {"resolution": 16, "aabb": [[-1.5] * 3, [1.5] * 3], "threshold": 0.01,
           "ema_decay": 0.5, "update_every": 1, "warmup_steps": 0, **kw}
    return OccupancyGrid(cfg, delta=0.0625, device=CPU)


def test_starts_fully_occupied_and_outside_aabb_is_empty():
    g = _grid()
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.4, -1.4, 0.2], [1.6, 0.0, 0.0], [0, 0, -3.0]])
    assert g.query(pts).tolist() == [True, True, False, False]
    assert g.occupied_fraction() == 1.0


def test_updates_carve_empty_space_but_keep_the_object():
    g = _grid()
    for _ in range(20):  # 0.5 ** 20 < threshold, so empty cells must switch off
        g.update(_Ball(), lambda x: x)
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.1, -0.1], [1.2, 1.2, 1.2], [-1.0, 0.0, 0.0]])
    assert g.query(pts).tolist() == [True, True, False, False]
    assert 0.0 < g.occupied_fraction() < 0.1


def test_skip_statistics_count_queries():
    g = _grid()
    g.query(torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 0, 0.1]]))
    assert g.skipped_fraction() == 0.5
    g.reset_stats()
    assert g.skipped_fraction() == 0.0


def test_all_occupied_grid_renders_like_no_grid():
    set_seed(0)
    coarse, fine, peXYZ, peDir = build_models(CFG, CPU)
    rays_o, rays_d = rays_for_pose(12, 12, 15.0, POSE)
    ref = render_image(coarse, fine, peXYZ, peDir, rays_o, rays_d, CFG)
    big = _grid(aabb=[[-10.0] * 3, [10.0] * 3])  # covers every sample
    out = render_image(coarse, fine, peXYZ, peDir, rays_o, rays_d, CFG, occ=big)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_empty_grid_renders_white_background():
    set_seed(0)
    coarse, fine, peXYZ, peDir = build_models(CFG, CPU)
    g = _grid()
    g.occupied.zero_()
    rays_o, rays_d = rays_for_pose(6, 6, 8.0, POSE)
    out = render_image(coarse, fine, peXYZ, peDir, rays_o, rays_d, CFG, occ=g)
    assert torch.equal(out, torch.ones_like(out))


def test_training_with_grid_logs_skips_and_resumes_exactly(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("NERF_DEVICE", "cpu")
    cfg = apply_overrides(tiny_cfg, [
        "occupancy.enabled=true", "occupancy.resolution=8", "occupancy.update_every=2",
        "occupancy.warmup_steps=0", "occupancy.ema_decay=0.5",
    ])
    root = str(tmp_path)
    train(copy.deepcopy(cfg), "full", resume=False, logroot=root)
    train(apply_overrides(cfg, ["training.max_steps=10"]), "split", resume=False, logroot=root)
    train(copy.deepcopy(cfg), "split", resume=True, logroot=root)

    def rows(name):
        with open(f"{root}/{name}/metrics.jsonl") as f:
            return [r for r in map(json.loads, f) if "loss" in r]

    full, split = rows("full"), rows("split")
    assert all(0.0 <= r["skipped_frac"] <= 1.0 for r in full)
    assert [(r["step"], r["loss"]) for r in split] == [(r["step"], r["loss"]) for r in full]
    ck = torch.load(f"{root}/full/checkpoint_last.pth", weights_only=True)
    assert ck["occupancy"]["occupied"].shape == (8, 8, 8)
