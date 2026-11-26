import os

import numpy as np
import pytest
from PIL import Image

from scripts.data_loader import NeRFDataset, load_image

LEGO = "data/synthetic/lego"


def _write_rgba(path: str) -> None:
    """8x8 frame: left half opaque red, right half fully transparent (RGB 0)."""
    arr = np.zeros((8, 8, 4), np.uint8)
    arr[:, :4] = [255, 0, 0, 255]
    Image.fromarray(arr, "RGBA").save(path)


def test_white_bkgd_composites_transparent_pixels_to_white(tmp_path):
    path = str(tmp_path / "f.png")
    _write_rgba(path)
    img = load_image(path, 8, 8, white_bkgd=True)
    assert img.dtype == np.float32
    np.testing.assert_array_equal(img[0, -1], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(img[0, 0], [1.0, 0.0, 0.0])


def test_black_bkgd_reproduces_v1_loader(tmp_path):
    path = str(tmp_path / "f.png")
    _write_rgba(path)
    img = load_image(path, 8, 8, white_bkgd=False)
    np.testing.assert_array_equal(img[0, -1], [0.0, 0.0, 0.0])


def test_resize_does_not_darken_object_edges(tmp_path):
    # Premultiplied resampling keeps edge pixels red-to-white, never darker than both.
    path = str(tmp_path / "f.png")
    _write_rgba(path)
    img = load_image(path, 4, 4, white_bkgd=True)
    assert np.all(img[..., 0] >= 1.0 - 1e-6)


@pytest.mark.data
@pytest.mark.skipif(not os.path.isdir(LEGO), reason="LEGO data not present")
def test_lego_training_background_is_white():
    cfg = {"data": {"root": LEGO, "height": 400, "width": 400, "focal": "AUTO",
                    "white_bkgd": True}}
    ds = NeRFDataset(cfg, "train", indices=[0])
    np.testing.assert_array_equal(ds.images[0, 0, 0], [1.0, 1.0, 1.0])
    assert ds.images.min() >= 0.0 and ds.images.max() <= 1.0
