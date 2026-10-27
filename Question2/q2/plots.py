"""Frame-grid figures, animated GIFs and digit-overlap detection."""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .utils import FIGURES_DIR, GIFS_DIR, ensure


def _img(f):
    return np.asarray(f).reshape(64, 64)


def frame_grid(ax_rows, rows, row_labels, col_titles=None):
    """Draw rows of frames on a grid of axes. rows: list of lists of (64, 64) arrays (None = blank)."""
    for r, (axes, frames) in enumerate(zip(ax_rows, rows)):
        for c, ax in enumerate(axes):
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(frames) and frames[c] is not None:
                ax.imshow(_img(frames[c]), cmap="gray", vmin=0, vmax=1)
            else:
                ax.axis("off")
            if r == 0 and col_titles:
                ax.set_title(col_titles[c], fontsize=7)
        axes[0].set_ylabel(row_labels[r], fontsize=7, rotation=0, ha="right", va="center")


def save_fig(fig, name):
    path = os.path.join(ensure(FIGURES_DIR), name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def four_row_panel(X, fX, TX, TY, fTX, TfX, title, labels):
    """Figure 3 / 6(a) panel: four rows of ten frames (inputs in cols 1-5, futures in 6-10)."""
    rows = [
        list(X) + list(fX),
        list(TX) + list(TY),
        list(TX) + list(fTX),
        list(TX) + list(TfX),
    ]
    fig, axes = plt.subplots(4, 10, figsize=(10.5, 4.6))
    cols = [f"in {i}" for i in range(1, 6)] + [f"h={h}" for h in range(1, 6)]
    frame_grid(axes, rows, labels, cols)
    fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    return fig, rows


def rows_gif(rows, labels, path, title):
    """Animated GIF stepping through the five forecast horizons.

    Each GIF frame keeps the same rows; columns 1-5 hold the row's five input frames and
    column 6 its future frame at horizon h = 1..5."""
    frames = []
    for h in range(5):
        fig, axes = plt.subplots(len(rows), 6, figsize=(7.2, 1.3 * len(rows)))
        for r, row in enumerate(rows):
            cells = [(row[i], f"in {i + 1}") for i in range(5)] + [(row[5 + h], f"h = {h + 1}")]
            for c, (img, t) in enumerate(cells):
                ax = axes[r, c]
                ax.imshow(_img(img), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(t, fontsize=8)
            axes[r, 0].set_ylabel(labels[r], fontsize=6, rotation=0, ha="right", va="center")
        fig.suptitle(title, fontsize=7)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    ensure(os.path.dirname(path))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=700, loop=0)
    return path


def gif_path(name):
    return os.path.join(ensure(GIFS_DIR), name)


def n_blobs(frame, thresh=0.2, min_area=8):
    """Number of connected bright regions (digits) in a [0, 1] frame."""
    b = (_img(frame) > thresh).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_area).sum())


def overlap_score(frames):
    """Minimum over frames of (bright-pixel count / max count in the clip): low = overlapping."""
    counts = np.array([(_img(f) > 0.2).sum() for f in frames], dtype=float)
    return counts.min() / max(counts.max(), 1.0)
