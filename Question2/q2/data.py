"""Moving MNIST loading, the square symmetries G, and speed windows.

Frames are numbered 1-20 (as in the handout); internally tensors are 0-indexed.
"""
import os

import torch
import torchvision

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

N_IN = 5
N_OUT = 5
N_FRAMES = 20

SPLITS = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}


def load_videos(split):
    """Return the uint8 videos of a split, shape (N, 20, 1, 64, 64).

    The dataset is split by video index, never with torchvision's frame-level
    split="train"/"test" options.
    """
    ds = torchvision.datasets.MovingMNIST(root=DATA_ROOT, split=None, download=True)
    lo, hi = SPLITS[split]
    return ds.data[lo:hi].contiguous()


def to_float(x):
    """uint8 frames -> float32 in [0, 1]."""
    return x.float().div_(255.0)


# ---------------------------------------------------------------------------
# Symmetries of the square
# ---------------------------------------------------------------------------

def mirror(x):
    """(M x)[i, j] = x[i, 63 - j]: flip the column axis."""
    return torch.flip(x, dims=(-1,))


def rotate(x, a=1):
    """R^a: rotate by a * 90 degrees anticlockwise."""
    return torch.rot90(x, k=a, dims=(-2, -1))


# G = {R^a, M R^a : a = 0..3}; element g in 0..7, g < 4 -> R^g, g >= 4 -> M R^(g-4).
G_NAMES = ["I", "R", "R2", "R3", "M", "MR", "MR2", "MR3"]


def apply_g(x, g):
    """Apply group element g to the last two (spatial) axes. (M R^a)(x) = M(R^a(x))."""
    if g < 4:
        return rotate(x, g)
    return mirror(rotate(x, g - 4))


def apply_g_inv(x, g):
    """Apply g^{-1}: R^a -> R^{-a}; every M R^a is its own inverse."""
    if g < 4:
        return rotate(x, -g)
    return apply_g(x, g)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def frame_indices(t, k):
    """1-based frame numbers of the input and of the *existing* target frames."""
    inp = [t + k * i for i in range(N_IN)]
    tgt = [t + k * (N_IN + i) for i in range(N_OUT)]
    return inp, [f for f in tgt if f <= N_FRAMES]


def allowed_starts(k, augmented):
    """Allowed start frames t (1-based).

    Normal-speed training (A, B): k = 1 and all five targets exist -> t in 1..11.
    Speed-augmented training (C, D): five inputs and at least two targets exist.
    """
    if not augmented:
        assert k == 1
        return list(range(1, N_FRAMES - (N_IN + N_OUT - 1) + 1))
    return [t for t in range(1, N_FRAMES + 1) if t + (N_IN + 1) * k <= N_FRAMES]


def window(video, t, k=1):
    """Cut a window from video(s) of shape (..., 20, C, H, W).

    Returns X (..., 5, C, H, W), Y (..., n, C, H, W) with the n <= 5 target
    frames that exist (frame number <= 20).
    """
    inp, tgt = frame_indices(t, k)
    X = video[..., [f - 1 for f in inp], :, :, :]
    Y = video[..., [f - 1 for f in tgt], :, :, :]
    return X, Y


def window_padded(video, t, k=1):
    """Like `window`, but Y is padded to 5 frames with zeros and a (5,) mask marks real targets."""
    X, Y = window(video, t, k)
    n = Y.shape[-4]
    mask = torch.zeros(N_OUT, dtype=torch.bool)
    mask[:n] = True
    if n < N_OUT:
        pad = torch.zeros(*Y.shape[:-4], N_OUT - n, *Y.shape[-3:], dtype=Y.dtype)
        Y = torch.cat([Y, pad], dim=-4)
    return X, Y, mask
