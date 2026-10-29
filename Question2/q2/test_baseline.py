"""Section 3 (Testing the Baseline): Tables 3-4, Figures 3-5 and the Figure 3 GIFs.

Run: python -m q2.test_baseline
"""
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import TRANSFORMS, load_model, predict, test_window
from .plots import four_row_panel, frame_grid, gif_path, n_blobs, rows_gif, save_fig
from .run_eval import get_metrics
from .utils import load_json, save_json, sig3

SEEDS = (0, 1)
ROW_LABELS = {
    "M": ["(1) X | f(X)", "(2) M(X) | M(Y)", "(3) M(X) | f(M(X))", "(4) M(X) | M(f(X))"],
    "R": ["(1) X | f(X)", "(2) R(X) | R(Y)", "(3) R(X) | f(R(X))", "(4) R(X) | R(f(X))"],
}


def fmt(vals):
    if len(vals) == 1:
        return sig3(vals[0])
    return f"{sig3(float(np.mean(vals)))} ({sig3(vals[0])}, {sig3(vals[1])})"


def summary_row(name, keys):
    seeds = SEEDS if name not in ("Blank", "CopyLast") else (None,)
    ms = [get_metrics(name, s) for s in seeds]
    return {k: fmt([m[k] for m in ms]) for k in keys}, [{k: m[k] for k in keys} for m in ms]


def table3():
    keys = ["E5_1x", "C_M", "E_M", "C_R", "E_R"]
    out = {}
    print("Table 3\nmodel | " + " | ".join(keys))
    for name in ("A", "Blank", "CopyLast"):
        row, raw = summary_row(name, keys)
        out[name] = raw
        print(name, "|", " | ".join(row[k] for k in keys))
    save_json("table3.json", out)


def table4():
    keys = ["E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x"]
    out = {}
    print("Table 4\nmodel | " + " | ".join(keys))
    for name in ("A", "CopyLast"):
        row, raw = summary_row(name, keys)
        out[name] = raw
        print(name, "|", " | ".join(row[k] for k in keys))
    save_json("table4.json", out)


def top_videos(name="A", seed=0, n=2):
    m = get_metrics(name, seed)
    return {T: [int(i) for i in np.argsort(m[f"per_video_C_{T}"])[::-1][:n]] for T in TRANSFORMS}


def figure3(model, tag, title_model, videos, fig_name, gif_prefix):
    """One four-row panel (and GIF) per (transformation, video)."""
    X, Y = test_window(1)
    paths, gifs = [], []
    for T, idx in videos.items():
        fwd, _ = TRANSFORMS[T]
        for rank, i in enumerate(idx, 1):
            x, y = X[i:i + 1], Y[i:i + 1]
            fX = predict(model, x)
            fTX = predict(model, fwd(x))
            TfX = fwd(fX)
            title = f"{title_model}, T = {T}, test video {9000 + i} (top-{rank} per-video C_{T} of model A s0)"
            fig, rows = four_row_panel(x[0], fX[0], fwd(x)[0], fwd(y)[0], fTX[0], TfX[0], title, ROW_LABELS[T])
            paths.append(save_fig(fig, f"{fig_name}_{T}_{9000 + i}.png"))
            gifs.append(rows_gif(rows, ROW_LABELS[T], gif_path(f"{gif_prefix}_{T}_video{9000 + i}.gif"), title))
    print("saved", paths, gifs)
    return paths, gifs


def figure4(videos):
    m = get_metrics("A", 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, T in zip(axes, ("M", "R")):
        c = np.array(m[f"per_video_C_{T}"])
        ax.hist(c, bins=40, color="tab:blue" if T == "M" else "tab:orange", edgecolor="k")
        for j, i in enumerate(videos[T]):  # labels staggered so nearby markers stay readable
            ax.axvline(c[i], color="tab:red", ls="--")
            ax.annotate(str(9000 + i), (c[i], ax.get_ylim()[1] * (0.85 - 0.3 * j)), rotation=90, fontsize=7,
                        ha="right", color="tab:red")
        ax.set_xlabel(f"per-video $C_{T}$")
        ax.set_ylabel("number of test videos")
        ax.set_title(f"({'a' if T == 'M' else 'b'}) $C_{T}$, model A seed 0 (mean {sig3(c.mean())})")
    fig.tight_layout()
    print("saved", save_fig(fig, "fig4_consistency_hist_A.png"))


def pick_figure5_video():
    """First test video whose digits do not overlap in any displayed true frame (all speeds)."""
    shown = set()
    for k in (1, 2, 3):
        shown.update([1 + k * i for i in range(5)] + [1 + 5 * k, 1 + 6 * k])
    test = test_window(1)  # make sure the raw test set is cached
    from .evaluate import _cache
    vids = _cache["test"]
    for i in range(len(vids)):
        if all(n_blobs(vids[i, f - 1].float() / 255) == 2 for f in sorted(shown)):
            return i
    raise RuntimeError("no non-overlapping video found")


def figure5(model, title_model, video, fig_name):
    rows, labels = [], []
    for k in (1, 2, 3):
        Xk, Yk = test_window(k)
        P = predict(model, Xk[video:video + 1])[0]
        inp = [1 + k * i for i in range(5)]
        tgt = [1 + k * (5 + i) for i in range(2)]
        rows.append(list(Xk[video]) + [None] + list(P[:2]) + [None] + list(Yk[video, :2]))
        labels.append(f"{k}x\nin {inp}\npred/true {tgt}")
    fig, axes = plt.subplots(3, 11, figsize=(11, 3.6))
    cols = [f"in {i}" for i in range(1, 6)] + [""] + ["pred h=1", "pred h=2"] + [""] + ["true h=1", "true h=2"]
    frame_grid(axes, rows, labels, cols)
    fig.suptitle(f"{title_model}, test video {9000 + video} at 1x, 2x and 3x (t = 1)", fontsize=9)
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    print("saved", save_fig(fig, fig_name))


def main():
    table3()
    table4()
    videos = top_videos()
    save_json("fig3_videos.json", videos)
    figure3(load_model("A", 0), "A", "model A (seed 0)", videos, "fig3_A", "fig3_modelA")
    figure4(videos)
    v = pick_figure5_video()
    save_json("fig5_video.json", {"test_index": v, "dataset_index": 9000 + v})
    figure5(load_model("A", 0), "model A (seed 0)", v, "fig5_speed_A.png")


if __name__ == "__main__":
    main()
