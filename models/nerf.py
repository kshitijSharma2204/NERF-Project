import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEncoding(nn.Module):
    def __init__(self, freqs: int):
        super().__init__()
        self.freqs = freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = [x]
        for i in range(self.freqs):
            for fn in (torch.sin, torch.cos):
                enc.append(fn((2.0 ** i) * x))
        return torch.cat(enc, -1)


class NeRFMLP(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, input_ch: int, input_ch_dir: int,
                 skips: tuple[int, ...] = (4,), density_activation: str = "relu"):
        super().__init__()
        if density_activation not in ("relu", "softplus"):
            raise ValueError(f"unknown density_activation {density_activation!r}")
        self.density_activation = density_activation
        self.skips = tuple(skips)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_ch = input_ch if i == 0 else hidden_dim
            if i in self.skips:
                in_ch += input_ch
            self.layers.append(nn.Linear(in_ch, hidden_dim))
        self.sigma_out = nn.Linear(hidden_dim, 1)
        self.feature_out = nn.Linear(hidden_dim, hidden_dim)
        self.dir_layer = nn.Linear(hidden_dim + input_ch_dir, hidden_dim // 2)
        self.rgb_out = nn.Linear(hidden_dim // 2, 3)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, x], -1)
            h = torch.relu(layer(h))
        return h

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        if self.density_activation == "softplus":
            return F.softplus(raw - 1.0)  # mip-NeRF style: never zero gradient
        return torch.clamp(raw, min=0.0)  # ReLU, as in the NeRF paper

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """Density only (M, 1), for encoded positions x (M, input_ch)."""
        return self._activate(self.sigma_out(self._trunk(x)))

    def forward(self, x: torch.Tensor, d: torch.Tensor,
                noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """x: encoded positions (M, input_ch), d: encoded directions (M, input_ch_dir).

        noise_std > 0 adds Gaussian noise to the raw density before the activation
        (the NeRF paper's raw-noise regulariser; training only).
        Returns rgb (M, 3) in [0, 1] and sigma (M, 1) >= 0.
        """
        h = self._trunk(x)
        raw = self.sigma_out(h)
        if noise_std > 0.0:
            raw = raw + noise_std * torch.randn_like(raw)
        sigma = self._activate(raw)
        feats = self.feature_out(h)
        h_dir = torch.relu(self.dir_layer(torch.cat([feats, d], -1)))
        rgb = torch.sigmoid(self.rgb_out(h_dir))
        rgb = torch.nan_to_num(rgb, nan=0.0)
        return rgb, sigma
