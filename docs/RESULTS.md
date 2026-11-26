# NeRF v2 results: Blender LEGO

This is the single source for quoting v2 results. Every number below is copied from
`outputs/v2/eval/metrics.json`, `outputs/v2/eval/per_view.csv` or `logs/lego_v2/metrics.jsonl`,
which `run_lego.sh` produced at commit `5df01d3`.

## Final numbers (all 200 test views)

| Metric | Mean | Std (population) | Min | Max |
|---|---|---|---|---|
| PSNR (dB) | 31.200286236503135 | 1.3463872767822471 | 27.728355379673527 | 33.397222564496914 |
| SSIM | 0.9614983501622538 | 0.005095640977915376 | 0.9469291458192436 | 0.9728876655755726 |

Rounded: **PSNR 31.20 ± 1.35 dB, SSIM 0.9615 ± 0.0051** (mean ± std over 200 views).

- **Resolution:** 400×400. **Background:** white.
- **Checkpoint:** `logs/lego_v2/checkpoint_last.pth` (step 199999, i.e. 200,000 training steps).
- **Commit:** `5df01d3740be4d7462b3ff524c95e0fb214db4ac`. This is both `checkpoint_git_sha` and
  `eval_git_sha` in `metrics.json`, with no uncommitted changes to code or config.
- **Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), torch 2.14.1+cu126, fp32 throughout
  (TF32 off).
- **Training time:** 26,912 s (7 h 28 min) of wall-clock time from the step-0 log to the
  step-199999 log, 2026-11-26 00:03–07:31. This includes 40 validations and the checkpoint saves.
  It was one uninterrupted process with no resume.
- **Training speed:** median 128–131 ms/step in each 20k-step block after the occupancy grid
  warm-up. Peak allocated GPU memory: 2.97 GiB.
- **Rendering speed:** mean 5.92 s per 400×400 frame (`render_seconds_per_frame`).

## Protocol

Test views are rendered deterministically: no stratified jitter, no density noise, and
deterministic fine sampling, so re-running gives bit-identical images. The fine network's output
is saved as 8-bit PNG and scored against the ground truth. The ground truth is loaded as RGBA,
LANCZOS-resized to 400×400 (Pillow premultiplies alpha) and composited onto white.
PSNR = −10·log10(MSE) on [0, 1] RGB. SSIM uses an 11×11 Gaussian window (σ = 1.5,
C1 = 0.01², C2 = 0.03²) with valid convolution, averaged over RGB channels and pixels. The std
is the population std over views. The test split was not used for any training or tuning
decision; only val r_0, r_33 and r_66 were.

## Training curve (val PSNR, mean of val r_0, r_33, r_66)

| Step | 5k | 10k | 25k | 50k | 75k | 100k | 125k | 150k | 175k | 199,999 |
|---|---|---|---|---|---|---|---|---|---|---|
| dB | 23.22 | 24.53 | 26.27 | 28.09 | 28.83 | 29.65 | 30.06 | 30.45 | 30.71 | 31.46 |

All 40 validation points are in `logs/lego_v2/metrics.jsonl` (`val_psnr` records).

## Command

```bash
./run_lego.sh training.max_steps=200000
```

This trains with `configs/synthetic_lego.yaml` (the defaults, plus the step-count override),
then runs `scripts/evaluate.py` on the final checkpoint and renders the fly-around.

## Media

| File | What |
|---|---|
| `outputs/v2/video.mp4` | 360° fly-around, 120 frames, 400×400 (main asset) |
| `outputs/v2/video.gif` | Same fly-around, 320 px GIF (1.5 MB) |
| `outputs/v2/test/160.png`, `161.png`, `159.png` | The three best test views (33.40, 33.33, 33.16 dB) |
| `outputs/v2/test/150.png`, `162.png`, `178.png`, `193.png` | The four views v1 saved, for side-by-side display |
| `outputs/v2/test/081.png` | The worst test view (27.73 dB) |

Per-view scores for the views above (from `per_view.csv`):

| View | PSNR (dB) | SSIM |
|---|---|---|
| r_160 | 33.397222564496914 | 0.9642805232924492 |
| r_161 | 33.33170082534942 | 0.9651965774654213 |
| r_159 | 33.159749114580805 | 0.9616001122932797 |
| r_150 | 32.670256402633356 | 0.9625132647750614 |
| r_162 | 33.1225005546334 | 0.9651230554723278 |
| r_178 | 30.53334193380776 | 0.960699822313193 |
| r_193 | 31.327483650601753 | 0.9602900375682936 |
| r_81 | 27.728355379673527 | 0.9471647794512976 |

## Not comparable to v1

v1 scored 29.90 dB / 0.9448 on 4 hand-picked views with a **black** background (see the README).
The v2 numbers use all 200 views and a **white** background, so the two are not directly
comparable. The per-view v2 scores for r_150, r_162, r_178 and r_193 are shown above for display,
not as a like-for-like comparison.
