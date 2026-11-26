import torch

from scripts.train import sample_pixels


def test_precrop_samples_only_the_central_region():
    torch.manual_seed(0)
    i, j = sample_pixels(10_000, 400, 400, 0.5, torch.device("cpu"))
    assert i.min() >= 100 and i.max() < 300
    assert j.min() >= 100 and j.max() < 300
    # covers the crop, not a corner of it
    assert i.min() == 100 and i.max() == 299


def test_no_crop_covers_the_full_image():
    torch.manual_seed(0)
    i, j = sample_pixels(100_000, 40, 30, None, torch.device("cpu"))
    assert (i.min(), i.max(), j.min(), j.max()) == (0, 29, 0, 39)
