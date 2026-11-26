import copy
import json
import os
import random

import numpy as np
import pytest
import torch

from scripts.checkpoint import (
    capture_rng,
    find_last_ckpt_and_step,
    numbered_checkpoints,
    restore_rng,
    save_checkpoint,
)
from scripts.train import apply_overrides, train


def _losses(logdir):
    with open(os.path.join(logdir, "metrics.jsonl")) as f:
        return [(r["step"], r["loss"]) for r in map(json.loads, f) if "loss" in r]


def test_rng_roundtrip_through_weights_only_load(tmp_path):
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    path = str(tmp_path / "rng.pth")
    torch.save({"rng": capture_rng()}, path)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    restore_rng(torch.load(path, weights_only=True)["rng"])
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rng_restores_after_loading_onto_cuda(tmp_path):
    # Regression: map_location=cuda moved the numpy key tensor to the GPU on resume.
    np.random.seed(5)
    path = str(tmp_path / "rng.pth")
    torch.save({"rng": capture_rng()}, path)
    expected = np.random.rand()
    restore_rng(torch.load(path, map_location="cuda", weights_only=True)["rng"])
    assert np.random.rand() == expected


def test_keeps_newest_three_numbered_plus_last(tmp_path):
    for step in (5, 10, 15, 20, 25):
        save_checkpoint(str(tmp_path), {"step": step}, keep=3)
    assert [s for s, _ in numbered_checkpoints(str(tmp_path))] == [15, 20, 25]
    assert os.path.exists(tmp_path / "checkpoint_last.pth")
    ck, step, path = find_last_ckpt_and_step(str(tmp_path), torch.device("cpu"))
    assert step == 25 and ck["step"] == 25


def test_resume_reproduces_uninterrupted_run_on_cpu(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("NERF_DEVICE", "cpu")
    logroot = str(tmp_path / "logs")
    train(copy.deepcopy(tiny_cfg), "full", resume=False, logroot=logroot)

    half = apply_overrides(tiny_cfg, ["training.max_steps=10"])
    train(half, "split", resume=False, logroot=logroot)
    train(copy.deepcopy(tiny_cfg), "split", resume=True, logroot=logroot)

    full, split = _losses(f"{logroot}/full"), _losses(f"{logroot}/split")
    assert [s for s, _ in split] == list(range(20))
    assert split == full  # bit-identical loss curve on CPU


def test_checkpoint_has_full_state(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("NERF_DEVICE", "cpu")
    train(copy.deepcopy(tiny_cfg), "run", resume=False, logroot=str(tmp_path))
    ck = torch.load(tmp_path / "run" / "checkpoint_last.pth", weights_only=True)
    assert {"coarse", "fine", "optimizer", "lr_state", "step", "rng", "config"} <= ck.keys()
    assert ck["step"] == 19 and ck["optimizer"]["state"]


def test_apply_overrides_parses_yaml_values():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2.5", "a.c=true", "seed=7"])
    assert cfg == {"a": {"b": 2.5, "c": True}, "seed": 7}
