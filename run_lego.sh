#!/usr/bin/env bash
set -e

# Train
python3 -m scripts.train \
  --config configs/synthetic_lego.yaml \
  --expname lego_full \
  --resume

# Render
LATEST=$(ls logs/lego_full/checkpoint_*.pth | sort -V | tail -n1)
python3 -m scripts.render \
  --config  configs/synthetic_lego.yaml \
  --expname lego_full \
  --ckpt    "$LATEST" \
  --split   test \
  --outdir  outputs/lego_test

# Make video
ffmpeg -y -framerate 30 -i outputs/lego_test/%03d.png \
  -c:v libx264 -pix_fmt yuv420p outputs/lego_test/video.mp4