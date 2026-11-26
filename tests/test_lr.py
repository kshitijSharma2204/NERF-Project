import json

import pytest

from scripts.train import apply_overrides, lr_at, train


@pytest.mark.parametrize("step,expected", [
    (0, 5e-4),
    (250_000, 5e-4 * 0.1 ** 0.5),
    (300_000, 5e-4 * 0.1 ** 0.6),
    (500_000, 5e-5),
])
def test_lr_formula_spot_steps(step, expected):
    assert lr_at(step, 5e-4, 500_000, 0.1) == pytest.approx(expected, rel=1e-12)


def test_logged_lr_matches_formula(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("NERF_DEVICE", "cpu")
    cfg = apply_overrides(tiny_cfg, ["training.lr_decay_steps=10"])  # fast decay for the test
    train(cfg, "lr", resume=False, logroot=str(tmp_path))
    with open(tmp_path / "lr" / "metrics.jsonl") as f:
        logged = {r["step"]: r["lr"] for r in map(json.loads, f) if "lr" in r}
    for step in (0, 5, 10, 19):
        assert logged[step] == pytest.approx(5e-4 * 0.1 ** (step / 10), rel=1e-12)
