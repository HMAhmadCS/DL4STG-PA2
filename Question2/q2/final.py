"""Section 5 (The Final Model): Figure 6 and its GIFs.

Run: python -m q2.final [--final D]
"""
import argparse

from .evaluate import load_model
from .test_baseline import figure3, figure5
from .utils import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="D", help="learned model wrapped by the symmetriser (D or DB)")
    args = ap.parse_args()
    model = load_model(f"sym-{args.final}", 0)
    label = f"final model: symmetrised {args.final} (seed 0)"
    videos = load_json("fig3_videos.json")
    figure3(model, "final", label, videos, "fig6a_final", "fig6a_final")
    figure5(model, label, load_json("fig5_video.json")["test_index"], "fig6b_speed_final.png")


if __name__ == "__main__":
    main()
