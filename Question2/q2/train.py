"""Train one learned model with one seed.

Run: python -m q2.train --model A --seed 0

Model | architecture | training speeds      | symmetry augmentation
A     | baseline     | 1x only              | none
B     | baseline     | 1x only              | g ~ U(G)
C     | baseline     | k ~ U{1, 2, 3}       | none
"""
import argparse
import json
import os
import time

import torch

from .data import N_OUT, allowed_starts, apply_g, load_videos, to_float, window_padded
from .models import build, n_params
from .utils import CKPT_DIR, ROOT, ensure, seed_everything

CONFIGS = {
    "A": dict(arch="A", speed_aug=False, sym_aug=False),
    "B": dict(arch="A", speed_aug=False, sym_aug=True),
    "C": dict(arch="A", speed_aug=True, sym_aug=False),
}

EPOCHS = 30
BATCH = 32
LR = 1e-3


def masked_video_mse(pred, target, mask):
    """Squared error averaged over pixels and available target frames of each video,
    then over the videos in the batch.  mask: (B, 5) bool."""
    per_frame = ((pred - target) ** 2).mean(dim=(2, 3, 4))  # (B, 5)
    m = mask.float()
    return ((per_frame * m).sum(1) / m.sum(1)).mean()


def sample_batch(videos, idx, cfg, gen):
    """One random window per video, drawn under the rules in Setup."""
    Xs, Ys, Ms = [], [], []
    for i in idx.tolist():
        if cfg["speed_aug"]:
            k = int(torch.randint(1, 4, (1,), generator=gen))
            starts = allowed_starts(k, augmented=True)
        else:
            k = 1
            starts = allowed_starts(1, augmented=False)
        t = starts[int(torch.randint(len(starts), (1,), generator=gen))]
        X, Y, m = window_padded(videos[i], t, k)
        if cfg["sym_aug"]:
            g = int(torch.randint(8, (1,), generator=gen))
            X, Y = apply_g(X, g), apply_g(Y, g)
        Xs.append(X)
        Ys.append(Y)
        Ms.append(m)
    return to_float(torch.stack(Xs)), to_float(torch.stack(Ys)), torch.stack(Ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for checkpoint/log names (e.g. design trials)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = CONFIGS[args.model]
    name = f"{args.model}{args.tag}_s{args.seed}"

    videos = load_videos("train")  # loaded before timing starts (download excluded)
    val = load_videos("val")
    seed_everything(args.seed)
    gen = torch.Generator().manual_seed(args.seed)  # shuffling, windows, augmentation
    model = build(cfg["arch"])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"{name}: {cfg}, params={n_params(model)}, threads={torch.get_num_threads()}", flush=True)

    # Fixed validation windows (1x, t = 1) for monitoring only.
    Xv, Yv, Mv = window_padded(val, 1, 1)
    Xv, Yv, Mv = to_float(Xv), to_float(Yv), Mv.expand(len(val), N_OUT)

    log = {"name": name, "config": cfg, "params": n_params(model), "threads": torch.get_num_threads(),
           "epochs": []}
    log_dir = ensure(os.path.join(ROOT, "logs"))
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(videos), generator=gen)
        tot, nb = 0.0, 0
        for b in range(0, len(perm), BATCH):
            X, Y, M = sample_batch(videos, perm[b:b + BATCH], cfg, gen)
            loss = masked_video_mse(model(X), Y, M)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        epoch_time = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            vl = sum(masked_video_mse(model(Xv[i:i + 100]), Yv[i:i + 100], Mv[i:i + 100]).item()
                     for i in range(0, len(Xv), 100)) / (len(Xv) // 100)
        log["epochs"].append({"epoch": epoch, "train_loss": tot / nb, "val_E5_1x": vl, "time_s": epoch_time})
        print(f"{name} epoch {epoch:2d} train {tot / nb:.5f} val {vl:.5f} time {epoch_time:.1f}s", flush=True)
        with open(os.path.join(log_dir, f"{name}.json"), "w") as f:
            json.dump(log, f, indent=2)

    log["mean_epoch_time_s"] = sum(e["time_s"] for e in log["epochs"]) / len(log["epochs"])
    with open(os.path.join(log_dir, f"{name}.json"), "w") as f:
        json.dump(log, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "seed": args.seed},
               os.path.join(ensure(CKPT_DIR), f"{name}.pt"))
    print(f"{name} done; mean epoch time {log['mean_epoch_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
