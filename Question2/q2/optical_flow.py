"""Figure 1: direction and speed statistics of the training motion (Farneback flow).

Run: python -m q2.optical_flow
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_videos
from .utils import FIGURES_DIR, ensure, save_json

N_VIDEOS = 50
INT_THRESH = 0.2  # on the [0, 1] scale, i.e. > 51 on uint8
MAG_THRESH = 0.5  # pixels


def pooled_flow(videos):
    dxs, dys = [], []
    for v in videos:
        frames = v[:, 0].numpy()  # (20, 64, 64) uint8 = round(255 * x)
        for s in range(frames.shape[0] - 1):
            prev, nxt = frames[s], frames[s + 1]
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            keep = ((prev / 255.0) > INT_THRESH) & ((nxt / 255.0) > INT_THRESH)
            keep &= np.hypot(dx, dy) > MAG_THRESH
            dxs.append(dx[keep])
            dys.append(dy[keep])
    return np.concatenate(dxs), np.concatenate(dys)


def main():
    videos = load_videos("train")[:N_VIDEOS]
    dx, dy = pooled_flow(videos)
    theta = np.degrees(np.arctan2(-dy, dx)) % 360.0
    speed = np.hypot(dx, dy)
    n = len(speed)
    med = float(np.median(speed))

    # Direction histogram summary: fraction per 20-degree bin.
    dir_frac, edges = np.histogram(theta, bins=18, range=(0, 360))
    dir_frac = dir_frac / n
    # Mirrored version of the same data (theta -> 180 - theta) and rotated (theta + 90).
    mir_frac = np.histogram((180.0 - theta) % 360, bins=18, range=(0, 360))[0] / n
    rot_frac = np.histogram((theta + 90.0) % 360, bins=18, range=(0, 360))[0] / n

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.bar(edges[:-1], dir_frac, width=20, align="edge", edgecolor="k", color="tab:blue")
    ax.axhline(1 / 18, color="tab:red", ls="--", label="uniform (1/18)")
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlim(0, 360)
    ax.set_xlabel(r"flow direction $\theta$ = atan2($-d_y$, $d_x$) [deg]")
    ax.set_ylabel("fraction of kept pixels")
    ax.set_title("(a) direction")
    ax.legend(loc="lower right")

    ax = axes[1]
    bins = np.arange(0, np.ceil(speed.max()) + 0.25, 0.25)
    w = np.full_like(speed, 1.0 / n)
    ax.hist(speed, bins=bins, weights=w, color="tab:green", edgecolor="k")
    ax.axvline(med, color="tab:red", ls="--", label=f"median = {med:.3g} px/frame")
    ax.set_xlabel("flow speed [px / frame]")
    ax.set_ylabel("fraction of kept pixels")
    ax.set_title("(b) speed")
    ax.legend()
    fig.suptitle(f"Farneback flow, first {N_VIDEOS} training videos, 950 frame pairs, {n} pixels kept")
    fig.tight_layout()
    path = os.path.join(ensure(FIGURES_DIR), "fig1_flow.png")
    fig.savefig(path, dpi=150)
    print("saved", path)

    stats = {
        "n_pixels": int(n),
        "median_speed": med,
        "mean_speed": float(speed.mean()),
        "speed_quantiles": {q: float(np.quantile(speed, q)) for q in (0.1, 0.25, 0.75, 0.9, 0.95, 0.99)},
        "dir_bin_edges": edges.tolist(),
        "dir_frac": dir_frac.tolist(),
        "dir_frac_if_mirrored": mir_frac.tolist(),
        "dir_frac_if_rotated": rot_frac.tolist(),
        "dir_frac_min": float(dir_frac.min()),
        "dir_frac_max": float(dir_frac.max()),
        "tv_distance_to_mirrored": float(0.5 * np.abs(dir_frac - mir_frac).sum()),
        "tv_distance_to_rotated": float(0.5 * np.abs(dir_frac - rot_frac).sum()),
        "tv_distance_to_uniform": float(0.5 * np.abs(dir_frac - 1 / 18).sum()),
    }
    save_json("fig1_flow_stats.json", stats)
    print({k: v for k, v in stats.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
