"""Forecasters: ConvLSTM baseline (model A), model D, the symmetriser and the reference predictors.

Every forecaster maps X (B, 5, 1, 64, 64) -> Y_hat (B, 5, 1, 64, 64).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_IN, N_OUT, apply_g, apply_g_inv


class ConvLSTMCell(nn.Module):
    """ConvLSTM without peepholes (handout eqs. 3-4); one conv produces all four gates."""

    def __init__(self, in_ch, hid_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.hid_ch = hid_ch
        pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size,
                              stride=1, padding=pad, dilation=dilation, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Two-layer ConvLSTM on pixel-unshuffled frames with a closed-loop 9-step rollout.

    Model A: patch=4, hidden=64, 3x3 kernels, dilations (1, 1).
    """

    def __init__(self, patch=4, hidden=64, kernel_size=3, dilations=(1, 1)):
        super().__init__()
        self.patch = patch
        cin = patch * patch
        self.cell1 = ConvLSTMCell(cin, hidden, kernel_size, dilations[0])
        self.cell2 = ConvLSTMCell(hidden, hidden, kernel_size, dilations[1])
        self.out = nn.Conv2d(hidden, cin, kernel_size=1, bias=True)

    def step(self, frame, s1, s2):
        u = F.pixel_unshuffle(frame, self.patch)
        s1 = self.cell1(u, s1)
        s2 = self.cell2(s1[0], s2)
        x_next = torch.sigmoid(F.pixel_shuffle(self.out(s2[0]), self.patch))
        return x_next, s1, s2

    def forward(self, X):
        B, T, C, H, W = X.shape
        hs, ws = H // self.patch, W // self.patch
        z = X.new_zeros(B, self.cell1.hid_ch, hs, ws)
        s1, s2 = (z, z), (z, z)
        preds = []
        u = X[:, 0]
        for tau in range(1, N_IN + N_OUT):  # 9 steps
            x_next, s1, s2 = self.step(u, s1, s2)
            if tau >= N_IN:
                preds.append(x_next)
            # Steps 1-5 read the observed frames; steps 6-9 read the previous prediction.
            u = X[:, tau] if tau < N_IN else x_next
        return torch.stack(preds, dim=1)


def model_a():
    return ConvLSTMForecaster(patch=4, hidden=64, kernel_size=3, dilations=(1, 1))


def model_d():
    """Model D: model A on an 8x8 fold (8x8 grid); one 3x3 hop moves 8 px instead of 4."""
    return ConvLSTMForecaster(patch=8, hidden=64, kernel_size=3, dilations=(1, 1))


def build(arch):
    return {"A": model_a, "D": model_d}[arch]()


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 sum_g g^{-1}(f(g(X))) over the 8 elements of G (handout eq. 7)."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, X):
        return sum(apply_g_inv(self.f(apply_g(X, g)), g) for g in range(8)) / 8


class Blank(nn.Module):
    def forward(self, X):
        return torch.zeros_like(X[:, :1]).expand(-1, N_OUT, -1, -1, -1)


class CopyLast(nn.Module):
    def forward(self, X):
        return X[:, -1:].expand(-1, N_OUT, -1, -1, -1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
