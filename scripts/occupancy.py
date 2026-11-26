"""Binary occupancy grid for empty-space skipping (own PyTorch implementation).

The grid covers an axis-aligned box (AABB) with res^3 cells. Each cell keeps an
opacity estimate alpha = 1 - exp(-sigma * delta) of the coarse model, where delta
is the coarse sample spacing. All cells start occupied (alpha = 1). Each update
evaluates the coarse density at one random point per cell and applies a decaying
maximum, ema = max(decay * ema, alpha_new), as in Instant-NGP: a cell only turns
empty after its density stays low for many updates, and one high reading turns it
back on. Cells with ema > threshold are occupied. Points outside the AABB are
treated as empty.
"""
import torch


class OccupancyGrid:
    def __init__(self, occ_cfg: dict, delta: float, device: torch.device):
        self.res = int(occ_cfg.get("resolution", 128))
        aabb = occ_cfg.get("aabb", [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)  # (2, 3)
        self.threshold = float(occ_cfg.get("threshold", 0.01))
        self.decay = float(occ_cfg.get("ema_decay", 0.95))
        self.update_every = int(occ_cfg.get("update_every", 16))
        self.warmup_steps = int(occ_cfg.get("warmup_steps", 500))
        self.chunk = int(occ_cfg.get("update_chunk", 1 << 18))
        self.delta = float(delta)
        self.device = device
        self.ema = torch.ones((self.res,) * 3, device=device)
        self.occupied = torch.ones((self.res,) * 3, dtype=torch.bool, device=device)
        self.reset_stats()

    # --- stats -------------------------------------------------------------------------
    def reset_stats(self) -> None:
        self._total = torch.zeros((), dtype=torch.int64, device=self.device)
        self._skipped = torch.zeros((), dtype=torch.int64, device=self.device)

    def skipped_fraction(self) -> float:
        total = int(self._total)
        return int(self._skipped) / total if total else 0.0

    def occupied_fraction(self) -> float:
        return float(self.occupied.float().mean())

    # --- query -------------------------------------------------------------------------
    def query(self, pts: torch.Tensor) -> torch.Tensor:
        """Boolean occupancy for points (..., 3); points outside the AABB are empty."""
        lo, hi = self.aabb[0], self.aabb[1]
        u = (pts - lo) / (hi - lo)
        inside = ((u >= 0) & (u < 1)).all(-1)
        idx = (u * self.res).long().clamp_(0, self.res - 1)
        occ = self.occupied[idx[..., 0], idx[..., 1], idx[..., 2]]
        mask = occ & inside
        self._total += mask.numel()
        self._skipped += mask.numel() - mask.sum()
        return mask

    # --- update ------------------------------------------------------------------------
    def should_update(self, step: int) -> bool:
        return step >= self.warmup_steps and step % self.update_every == 0

    @torch.no_grad()
    def update(self, coarse_model, peXYZ) -> None:
        """One random point per cell through the coarse density; decaying-max update."""
        lo, hi = self.aabb[0], self.aabb[1]
        n = self.res ** 3
        flat_ema = self.ema.view(-1)
        cell = (hi - lo) / self.res
        r2 = self.res * self.res
        was_training = coarse_model.training
        coarse_model.eval()
        for start in range(0, n, self.chunk):
            ids = torch.arange(start, min(start + self.chunk, n), device=self.device)
            ijk = torch.stack([ids // r2, (ids // self.res) % self.res, ids % self.res], -1)
            pts = lo + (ijk.float() + torch.rand_like(ijk, dtype=torch.float32)) * cell
            sigma = coarse_model.density(peXYZ(pts)).squeeze(-1)
            alpha = 1.0 - torch.exp(-sigma * self.delta)
            flat_ema[ids] = torch.maximum(flat_ema[ids] * self.decay, alpha)
        coarse_model.train(was_training)
        self.occupied = self.ema > self.threshold

    # --- persistence -------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"ema": self.ema, "occupied": self.occupied, "aabb": self.aabb}

    def load_state_dict(self, state: dict) -> None:
        self.ema = state["ema"].to(self.device)
        self.occupied = state["occupied"].to(self.device)
        self.aabb = state["aabb"].to(self.device)
        self.res = self.ema.shape[0]


def build_occupancy(cfg: dict, device: torch.device) -> OccupancyGrid | None:
    occ_cfg = cfg.get("occupancy", {})
    if not occ_cfg.get("enabled", False):
        return None
    m = cfg["model"]
    delta = (m.get("far", 6.0) - m.get("near", 2.0)) / m["coarse"]["depth_samples"]
    return OccupancyGrid(occ_cfg, delta, device)
