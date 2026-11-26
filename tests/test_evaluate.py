import copy
import json
import os

import pytest

from scripts.evaluate import evaluate
from scripts.evaluate_saved_renders import evaluate_saved
from scripts.train import train


def test_evaluate_twice_gives_identical_numbers(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("NERF_DEVICE", "cpu")
    train(copy.deepcopy(tiny_cfg), "ev", resume=False, logroot=str(tmp_path))
    ckpt = str(tmp_path / "ev" / "checkpoint_last.pth")
    a = evaluate(tiny_cfg, ckpt, str(tmp_path / "out_a"))
    b = evaluate(tiny_cfg, ckpt, str(tmp_path / "out_b"))
    for key in ("psnr", "ssim"):
        assert a[key] == b[key]
    assert a["n_views"] == 3 and a["checkpoint_step"] == 19 and a["background"] == "white"
    assert sorted(os.listdir(tmp_path / "out_a" / "test")) == ["000.png", "001.png", "002.png"]
    with open(tmp_path / "out_a" / "eval" / "metrics.json") as f:
        assert json.load(f)["psnr"]["per_view"].keys() == {"r_0", "r_1", "r_2"}
    with open(tmp_path / "out_a" / "eval" / "per_view.csv") as f:
        assert f.readline().strip() == "index,name,psnr,ssim,render_seconds"


@pytest.mark.data
@pytest.mark.skipif(not os.path.exists("data/synthetic/lego/test/r_150.png")
                    or not os.path.exists("outputs/150.png"), reason="needs LEGO + v1 renders")
def test_v1_numbers_reproduce():
    res = evaluate_saved()
    assert res["mean_psnr"] == pytest.approx(29.90, abs=0.01)
    assert res["mean_ssim"] == pytest.approx(0.9448, abs=0.0001)
