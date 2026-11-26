"""Image metrics. All inputs are float RGB images in [0, 1], shape (H, W, 3)."""
import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred: np.ndarray | torch.Tensor, gt: np.ndarray | torch.Tensor) -> float:
    """PSNR in dB for images in [0, 1]: -10 * log10(MSE)."""
    pred_t = torch.as_tensor(pred, dtype=torch.float64)
    gt_t = torch.as_tensor(gt, dtype=torch.float64, device=pred_t.device)
    mse = torch.mean((pred_t - gt_t) ** 2).item()
    return float("inf") if mse == 0 else -10.0 * float(np.log10(mse))


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(pred: np.ndarray | torch.Tensor, gt: np.ndarray | torch.Tensor) -> float:
    """SSIM with an 11x11 Gaussian window (sigma 1.5), C1 = 0.01^2, C2 = 0.03^2.

    Computed per RGB channel with 'valid' convolution (no padding), then averaged over
    channels and pixels. Done in float64 so results do not depend on the device.
    """
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    x = torch.as_tensor(pred, dtype=torch.float64).cpu().permute(2, 0, 1).unsqueeze(0)
    y = torch.as_tensor(gt, dtype=torch.float64).cpu().permute(2, 0, 1).unsqueeze(0)
    ch = x.shape[1]
    w = _gaussian_window().expand(ch, 1, 11, 11).contiguous()

    def filt(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z, w, groups=ch)

    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x ** 2
    syy = filt(y * y) - mu_y ** 2
    sxy = filt(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float((num / den).mean())
