"""Paths and small I/O helpers shared by the scripts."""
import json
import os
import random

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
GIFS_DIR = os.path.join(ROOT, "gifs")


def ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


def save_json(name, obj):
    path = os.path.join(ensure(RESULTS_DIR), name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sig3(x):
    """Three significant figures; scientific notation below 1e-3."""
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.3g}"
