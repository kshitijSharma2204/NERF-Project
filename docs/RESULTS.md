# NeRF v2 results on Blender LEGO

This is the file I quote results from. The numbers are copied straight out of
`outputs/v2/eval/metrics.json` and `outputs/v2/eval/per_view.csv` without rounding, so they
can always be checked against the source. Rounded versions are in the README.

## Headline numbers

Over all 200 test views:

| | Mean | Std | Min | Max |
|---|---|---|---|---|
| PSNR (dB) | 31.200286236503135 | 1.3463872767822471 | 27.728355379673527 | 33.397222564496914 |
| SSIM | 0.9614983501622538 | 0.005095640977915376 | 0.9469291458192436 | 0.9728876655755726 |

Rounded, that's 31.20 ± 1.35 dB PSNR and 0.9615 ± 0.0051 SSIM. The std is the population std
across the 200 views.

## How the model was trained

I ran the whole pipeline with one command at commit `5df01d3`
(`5df01d3740be4d7462b3ff524c95e0fb214db4ac`, no uncommitted changes):

```bash
./run_lego.sh training.max_steps=200000
```

That trains with the default `configs/synthetic_lego.yaml` and only overrides the step count.
After training, it evaluates the final checkpoint and renders the fly-around. Both the checkpoint
and the evaluation record that same commit in `metrics.json`.

Training ran on my laptop's RTX 3060 (6 GB) with torch 2.14.1+cu126, all in fp32 with TF32
off. It was one uninterrupted run from 00:03 to 07:31 on 2026-11-26. That's 26,912 seconds
(7 h 28 min) between the step 0 and step 199,999 logs, including 40 validation passes and
checkpoint saves. Once the occupancy grid warmed up, it held steady at 128 to 131 ms per step
(the median in each 20k-step block), and peak GPU memory was 2.97 GiB. The final checkpoint is
`logs/lego_v2/checkpoint_last.pth` (step 199,999). Rendering a 400x400 test image takes about
5.92 s on average.

Here's how validation PSNR improved. Each point is the mean of val views r_0, r_33 and r_66:

| Step | 5k | 10k | 25k | 50k | 75k | 100k | 125k | 150k | 175k | 199,999 |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR (dB) | 23.22 | 24.53 | 26.27 | 28.09 | 28.83 | 29.65 | 30.06 | 30.45 | 30.71 | 31.46 |

It was still going up slowly at the end. All 40 validation points are in
`logs/lego_v2/metrics.jsonl`.

## How it was scored

Each test view is rendered deterministically: no jitter along the rays, no density noise and
no random fine sampling, so running the evaluation twice gives identical images. I save the
fine network's output as an 8-bit PNG and score that file. The ground truth is loaded as RGBA,
resized to 400x400 with LANCZOS (Pillow premultiplies alpha) and composited onto white.

- **PSNR** is -10 log10(MSE) on RGB values in [0, 1].
- **SSIM** uses an 11x11 Gaussian window with sigma 1.5, C1 = 0.01² and C2 = 0.03², and valid
  convolution. It's averaged over the RGB channels and all pixels.

I never used the test split to make any decision during development. Only the three val views
were used for that.

## Images and video

| File | What it is |
|---|---|
| `outputs/v2/video.mp4` | The 360 degree fly-around: 120 frames at 400x400. This is the main one. |
| `outputs/v2/video.gif` | The same fly-around as a 320 px GIF (1.5 MB) |
| `outputs/v2/test/160.png`, `161.png`, `159.png` | The three best test views |
| `outputs/v2/test/150.png`, `162.png`, `178.png`, `193.png` | The four viewpoints v1 saved, for side-by-side pictures |
| `outputs/v2/test/081.png` | The worst test view, which still looks clean |

The scores for those views, from `per_view.csv`:

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

## A note on comparing with v1

v1 got 29.90 dB / 0.9448, but that was on four hand-picked views with a black background,
because the old loader dropped the alpha channel. v2 is scored on all 200 views on white, so
the two numbers don't measure the same thing. I show the v2 scores for r_150, r_162, r_178 and
r_193 above so the pictures can sit next to the v1 ones, not as a fair head-to-head.
