"""Evaluation helpers: per-video / per-horizon errors, consistency gaps, speeds, timing."""
import os
import platform
import time

import torch

from .data import load_videos, mirror, rotate, to_float, window
from .models import Blank, CopyLast, Symmetrised, build
from .train import CONFIGS
from .utils import CKPT_DIR

BATCH = 32
TRANSFORMS = {"M": (mirror, mirror), "R": (rotate, lambda x: rotate(x, -1))}


def load_model(name, seed):
    """name is a learned model (see train.CONFIGS), sym-<learned model>, Blank or CopyLast."""
    if name.startswith("sym-"):
        return Symmetrised(load_model(name[4:], seed)).eval()
    if name == "Blank":
        return Blank()
    if name == "CopyLast":
        return CopyLast()
    ck = torch.load(os.path.join(CKPT_DIR, f"{name}_s{seed}.pt"), map_location="cpu")
    m = build(CONFIGS[name]["arch"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def is_learned(name):
    return name not in ("Blank", "CopyLast")


@torch.no_grad()
def predict(model, X, batch=BATCH):
    model.eval()
    return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)])


def per_video_horizon_mse(pred, target):
    """(N, h, 1, 64, 64) x2 -> (N, h): MSE over the 64x64 pixels of each horizon."""
    n = target.shape[1]
    return ((pred[:, :n] - target) ** 2).mean(dim=(2, 3, 4))


_cache = {}


def test_window(k=1, t=1):
    """Float test windows X (1000, 5, 1, 64, 64) and the existing targets Y."""
    key = (k, t)
    if key not in _cache:
        if "test" not in _cache:
            _cache["test"] = load_videos("test")
        X, Y = window(_cache["test"], t, k)
        _cache[key] = (to_float(X), to_float(Y))
    return _cache[key]


def evaluate_all(model, speeds=(1, 2, 3)):
    """All metrics used in Tables 2-6 for one forecaster.

    Returns a dict with per-video arrays (kept as lists) and summary numbers.
    """
    out = {}
    X, Y = test_window(1)
    f_X = predict(model, X)
    mse = per_video_horizon_mse(f_X, Y)  # (1000, 5)
    out["MSE_h"] = mse.mean(0).tolist()
    out["E5_1x"] = mse.mean().item()
    out["E2_1x"] = mse[:, :2].mean().item()
    out["per_video_E5_1x"] = mse.mean(1).tolist()
    for T, (fwd, _) in TRANSFORMS.items():
        f_TX = predict(model, fwd(X))
        c = per_video_horizon_mse(f_TX, fwd(f_X)).mean(1)
        e = per_video_horizon_mse(f_TX, fwd(Y)).mean(1)
        out[f"C_{T}"] = c.mean().item()
        out[f"E_{T}"] = e.mean().item()
        out[f"per_video_C_{T}"] = c.tolist()
        out[f"per_video_E_{T}"] = e.tolist()
    for k in speeds:
        if k == 1:
            continue
        Xk, Yk = test_window(k)
        m = per_video_horizon_mse(predict(model, Xk), Yk)
        out[f"E2_{k}x"] = m[:, :2].mean().item()
        if Yk.shape[1] == 5:
            out[f"E5_{k}x"] = m.mean().item()
    return out


@torch.no_grad()
def inference_time(model, repeats=1):
    """Wall-clock time to predict all 1000 original (1x, t = 1) test videos at batch 32,
    under eval() and no_grad(), counting from when each batch is on the device (CPU here)."""
    X, _ = test_window(1)
    batches = [X[i:i + BATCH].contiguous() for i in range(0, len(X), BATCH)]
    model.eval()
    model(batches[0])  # warm-up
    best = float("inf")
    for _ in range(repeats):
        total = 0.0
        for b in batches:
            t0 = time.perf_counter()
            model(b)
            total += time.perf_counter() - t0
        best = min(best, total)
    return best


def hardware():
    return {"platform": platform.platform(), "processor": platform.processor(),
            "torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}
