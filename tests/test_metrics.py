import math

import numpy as np
import pytest

from scripts.metrics import psnr, ssim


def test_psnr_known_value():
    gt = np.zeros((4, 4, 3), np.float32)
    pred = np.full_like(gt, 0.1)  # MSE 0.01 -> 20 dB
    assert psnr(pred, gt) == pytest.approx(20.0, abs=1e-5)


def test_psnr_identical_is_infinite():
    img = np.random.default_rng(0).random((8, 8, 3))
    assert math.isinf(psnr(img, img))


def test_ssim_identical_is_one():
    img = np.random.default_rng(0).random((32, 32, 3))
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-12)


def test_ssim_drops_with_noise_and_is_symmetric():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3))
    noisy = np.clip(img + 0.2 * rng.standard_normal(img.shape), 0, 1)
    s = ssim(img, noisy)
    assert 0.0 < s < 0.9
    assert ssim(noisy, img) == pytest.approx(s, abs=1e-12)
