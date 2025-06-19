import torch
import torch.nn as nn

class PosEncoding(nn.Module):
    def __init__(self, freqs):
        super().__init__()
        self.freqs = freqs

    def forward(self, x):
        enc = [x]
        for i in range(self.freqs):
            for fn in (torch.sin, torch.cos):
                enc.append(fn((2.**i) * x))
        return torch.cat(enc, -1)


class NeRFMLP(nn.Module):
    def __init__(self, n_layers, hidden_dim, input_ch, input_ch_dir, skips=[4]):
        super().__init__()
        self.skips = skips
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_ch = input_ch if i == 0 else hidden_dim
            if i in skips:
                in_ch += input_ch
            self.layers.append(nn.Linear(in_ch, hidden_dim))
        self.sigma_out   = nn.Linear(hidden_dim, 1)
        self.feature_out = nn.Linear(hidden_dim, hidden_dim)
        self.dir_layer   = nn.Linear(hidden_dim + input_ch_dir, hidden_dim // 2)
        self.rgb_out     = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x, d):
      h = x
      for i, layer in enumerate(self.layers):
          if i in self.skips:
              h = torch.cat([h, x], -1)
          h = torch.relu(layer(h))
      sigma = self.sigma_out(h)
      feats = self.feature_out(h)
      h_dir = torch.relu(self.dir_layer(torch.cat([feats, d], -1)))
      rgb = torch.sigmoid(self.rgb_out(h_dir))
      # Clamp sigma to avoid negative values and handle nan
      sigma = torch.clamp(sigma, min=0.0)
      rgb = torch.nan_to_num(rgb, nan=0.0)
      return rgb, sigma