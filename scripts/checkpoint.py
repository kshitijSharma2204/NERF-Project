"""Full-state checkpoints: models, optimizer, LR state, step, RNG states and config."""
import glob
import os
import random
import re

import numpy as np
import torch

CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pth$")
LAST = "checkpoint_last.pth"


def capture_rng() -> dict:
    """RNG states in a form torch.load(weights_only=True) accepts (tensors and builtins)."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": name,
            "keys": torch.from_numpy(keys.astype(np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(_to_tuple(state["python"]))
    n = state["numpy"]
    np.random.set_state(
        (n["name"], n["keys"].cpu().numpy().astype(np.uint32), n["pos"], n["has_gauss"],
         n["cached_gaussian"])
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _to_tuple(x):
    return tuple(_to_tuple(v) for v in x) if isinstance(x, list | tuple) else x


def _atomic_save(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def numbered_checkpoints(logdir: str) -> list[tuple[int, str]]:
    """(step, path) of checkpoint_<step>.pth files, sorted by step."""
    out = []
    for p in glob.glob(os.path.join(logdir, "checkpoint_*.pth")):
        m = CKPT_RE.match(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def save_checkpoint(logdir: str, state: dict, keep: int = 3) -> str:
    """Write checkpoint_<step>.pth and checkpoint_last.pth; keep only the newest `keep` numbered."""
    step = state["step"]
    path = os.path.join(logdir, f"checkpoint_{step}.pth")
    _atomic_save(state, path)
    _atomic_save(state, os.path.join(logdir, LAST))
    for _, old in numbered_checkpoints(logdir)[:-keep] if keep > 0 else []:
        os.remove(old)
    return path


def find_last_ckpt_and_step(logdir: str, device: torch.device):
    """Load the most recent checkpoint. Returns (ckpt, step, path) or (None, 0, None)."""
    candidates = []
    last = os.path.join(logdir, LAST)
    if os.path.exists(last):
        candidates.append(last)
    candidates += [p for _, p in reversed(numbered_checkpoints(logdir))]
    if not candidates:
        return None, 0, None
    # checkpoint_last.pth is written after the numbered file, so it is never older;
    # still pick whichever has the highest stored step in case of a partial copy.
    best = None
    for p in candidates[:2]:
        ck = torch.load(p, map_location=device, weights_only=True)
        if best is None or ck["step"] > best[1]:
            best = (ck, ck["step"], p)
    return best
