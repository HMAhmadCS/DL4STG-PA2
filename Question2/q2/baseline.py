"""Section 2 (Baseline): Table 2 and Figure 2.

Run after training A with seeds 0 and 1: python -m q2.baseline
"""
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import load_model, per_video_horizon_mse, predict, test_window
from .models import n_params
from .plots import frame_grid, n_blobs, overlap_score, save_fig
from .utils import save_json, sig3

SEEDS = (0, 1)


def fmt(vals):
    """mean (seed 0, seed 1) or a single value for deterministic predictors."""
    if len(vals) == 1:
        return sig3(vals[0])
    return f"{sig3(float(np.mean(vals)))} ({sig3(vals[0])}, {sig3(vals[1])})"


def table2():
    X, Y = test_window(1)
    rows = {}
    for name, seeds in (("A", SEEDS), ("Blank", (None,)), ("CopyLast", (None,))):
        per_seed = []
        for s in seeds:
            m = load_model(name, s)
            mse = per_video_horizon_mse(predict(m, X), Y)
            per_seed.append({"MSE_h": mse.mean(0).tolist(), "E5": mse.mean().item(), "params": n_params(m)})
        rows[name] = per_seed
    save_json("table2.json", rows)
    print("Table 2")
    print("model | " + " | ".join(f"MSE{h}" for h in range(1, 6)) + " | E(5) | params")
    for name, ps in rows.items():
        cells = [fmt([p["MSE_h"][h] for p in ps]) for h in range(5)] + [fmt([p["E5"] for p in ps])]
        print(name, "|", " | ".join(cells), "|", ps[0]["params"])
    return rows


def pick_figure2_videos(Y, X):
    """Two clean videos plus the one whose true future shows the strongest digit overlap."""
    n = len(Y)
    overlap = np.array([overlap_score(Y[i]) for i in range(n)])
    clean = [i for i in range(n) if all(n_blobs(f) == 2 for f in list(X[i]) + list(Y[i]))]
    ov = int(np.argmin(overlap))
    return [clean[0], clean[1], ov]


def figure2():
    X, Y = test_window(1)
    f = load_model("A", 0)
    idx = pick_figure2_videos(Y, X)
    P = predict(f, X[idx])
    fig, axes = plt.subplots(3 * len(idx), 5, figsize=(5.6, 1.15 * 3 * len(idx)))
    for j, i in enumerate(idx):
        rows = [list(X[i]), list(P[j]), list(Y[i])]
        labels = [f"test {9000 + i}\ninput (1-5)", "prediction\n(6-10)", "truth\n(6-10)"]
        frame_grid(axes[3 * j:3 * j + 3], rows, labels)
    fig.suptitle("Figure 2: model A (seed 0); last video has overlapping digits", fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    print("saved", save_fig(fig, "fig2_model_A_examples.png"), "videos", [9000 + i for i in idx])
    save_json("fig2_videos.json", {"test_indices": idx, "dataset_indices": [9000 + i for i in idx]})


if __name__ == "__main__":
    table2()
    figure2()
