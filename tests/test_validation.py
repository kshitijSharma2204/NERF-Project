import json

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.train import apply_overrides, train


def test_validation_logs_val_psnr_loss_and_lr(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("NERF_DEVICE", "cpu")
    cfg = apply_overrides(tiny_cfg, ["training.val_interval=5", "training.max_steps=11"])
    train(cfg, "val", resume=False, logroot=str(tmp_path))

    with open(tmp_path / "val" / "metrics.jsonl") as f:
        vals = [r for r in map(json.loads, f) if "val_psnr" in r]
    # every val_interval, plus the final step
    assert [r["step"] for r in vals] == [5, 10]
    assert all(set(r["val_psnr_per_view"]) == {"r_0", "r_1"} for r in vals)

    ea = EventAccumulator(str(tmp_path / "val"))
    ea.Reload()
    assert {"loss/train", "val/psnr", "lr"} <= set(ea.Tags()["scalars"])
