import numpy as np
import pytest
import torch

from scripts.data_loader import NeRFDataset
from scripts.render import _check_outdir, load_models_from_ckpt, spherical_path
from scripts.render_rays import render_image, render_rays
from scripts.utils import build_models, configure_fp32, get_device, rays_for_pose, set_seed

CFG = {
    "model": {
        "near": 2.0, "far": 6.0,
        "coarse": {"n_layers": 2, "hidden_dim": 32, "depth_samples": 16},
        "fine": {"n_layers": 2, "hidden_dim": 32, "depth_samples": 32},
    },
}
POSE = torch.tensor([[1.0, 0, 0, 0], [0, 0, -1, -4.0], [0, 1, 0, 0], [0, 0, 0, 1]])


def _ckpt(tmp_path, device):
    set_seed(0)
    coarse, fine, _, _ = build_models(CFG, device)
    path = str(tmp_path / "ck.pth")
    torch.save({"coarse": coarse.state_dict(), "fine": fine.state_dict(), "step": 0}, path)
    return path


def _render(cfg, path, device):
    coarse, fine, peXYZ, peDir, _ = load_models_from_ckpt(cfg, path, device)
    rays_o, rays_d = rays_for_pose(24, 24, 30.0, POSE.to(device))
    return render_image(coarse, fine, peXYZ, peDir, rays_o, rays_d, cfg, chunk=200)


def test_deterministic_render_is_bit_identical(tmp_path):
    configure_fp32()
    device = get_device()
    path = _ckpt(tmp_path, device)
    torch.manual_seed(1)
    a = _render(CFG, path, device)
    torch.manual_seed(2)  # different RNG state must not matter
    b = _render(CFG, path, device)
    assert torch.equal(a, b)


def test_perturbed_training_render_is_random(tmp_path):
    device = get_device()
    set_seed(0)
    coarse, fine, peXYZ, peDir = build_models(CFG, device)
    rays_o, rays_d = rays_for_pose(8, 8, 10.0, POSE.to(device))
    with torch.no_grad():
        _, a = render_rays(coarse, fine, peXYZ, peDir, rays_o, rays_d, CFG, perturb=True)
        _, b = render_rays(coarse, fine, peXYZ, peDir, rays_o, rays_d, CFG, perturb=True)
    assert not torch.equal(a, b)


def test_spherical_path_orbits_and_looks_at_origin():
    poses = spherical_path(8, phi_deg=-30.0, radius=4.0)
    assert poses.shape == (8, 4, 4)
    for c2w in poses:
        pos = c2w[:3, 3]
        assert np.linalg.norm(pos) == pytest.approx(4.0, abs=1e-5)
        forward = -c2w[:3, 2]  # camera looks down its -z axis
        np.testing.assert_allclose(forward, -pos / np.linalg.norm(pos), atol=1e-5)
        assert pos[2] > 0  # phi = -30 puts the camera above the object, like the train views


def test_render_refuses_to_write_into_v1_outputs_dir():
    with pytest.raises(SystemExit):
        _check_outdir("outputs")
    _check_outdir("outputs/v2/renders")  # fine


def test_torch_rays_match_numpy_dataset_rays():
    ds = NeRFDataset.__new__(NeRFDataset)  # skip image loading
    ds.H, ds.W, ds.focal = 6, 5, 7.5
    ds.poses = POSE.numpy()[None]
    ds.images = np.zeros((1, 6, 5, 3), np.float32)
    o_np, d_np, _ = ds.get_rays(0)
    o, d = rays_for_pose(6, 5, 7.5, POSE)
    np.testing.assert_allclose(o.numpy(), o_np.reshape(-1, 3), atol=1e-6)
    np.testing.assert_allclose(d.numpy(), d_np.reshape(-1, 3), atol=1e-6)
