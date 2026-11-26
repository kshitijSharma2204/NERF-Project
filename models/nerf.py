import torch
import torch.nn as nn


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
                 skips: tuple[int, ...] = (4,)):
        super().__init__()
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

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """Density only (M, 1), for encoded positions x (M, input_ch)."""
        return torch.clamp(self.sigma_out(self._trunk(x)), min=0.0)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: encoded positions (M, input_ch), d: encoded directions (M, input_ch_dir).

        Returns rgb (M, 3) in [0, 1] and sigma (M, 1) >= 0.
        """
        h = self._trunk(x)
        sigma = torch.clamp(self.sigma_out(h), min=0.0)
        feats = self.feature_out(h)
        h_dir = torch.relu(self.dir_layer(torch.cat([feats, d], -1)))
        rgb = torch.sigmoid(self.rgb_out(h_dir))
        rgb = torch.nan_to_num(rgb, nan=0.0)
        return rgb, sigma
