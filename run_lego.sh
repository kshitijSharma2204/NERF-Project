#!/usr/bin/env bash
# v2 pipeline: train (resumable), evaluate all 200 test views, render the fly-around.
# Usage: ./run_lego.sh [extra --set overrides for training, e.g. training.max_steps=200000]
set -euo pipefail
export PYTHONUNBUFFERED=1   # step lines appear live when output goes to a log file

PY=${PY:-.venv/bin/python}
CONFIG=${CONFIG:-configs/synthetic_lego.yaml}
EXP=${EXP:-lego_v2}
CKPT=logs/$EXP/checkpoint_last.pth

# Train; --resume continues from checkpoint_last.pth if it exists. Keep the laptop awake.
systemd-inhibit --what=sleep:idle --why="NeRF training" \
  "$PY" -m scripts.train --config "$CONFIG" --expname "$EXP" --resume --set "$@"

# Metrics on all test views -> outputs/v2/eval/metrics.json, renders -> outputs/v2/test/
"$PY" -m scripts.evaluate --config "$CONFIG" --ckpt "$CKPT" --outdir outputs/v2

# 360-degree fly-around -> outputs/v2/video.mp4 (main asset) and a small GIF
"$PY" -m scripts.render --config "$CONFIG" --ckpt "$CKPT" --path spherical --n_frames 120 \
  --outdir outputs/v2/video_frames --video outputs/v2/video.mp4 --gif outputs/v2/video.gif
