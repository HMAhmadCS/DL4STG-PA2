"""Where a model puts each digit in the Figure 5 / 6(b) video: blob centroids of the last input,
the true and the predicted frames at h = 1, 2 for 1x, 2x and 3x (used for Q2.9 and Q2.16).

Run: python -m q2.positions A [sym-D ...]
"""
import argparse

import cv2
import numpy as np

from .evaluate import load_model, predict, test_window
from .utils import load_json, save_json


def blobs(frame, thresh=0.2, min_area=8):
    """(row, col) centroids and areas of the bright connected regions of a [0, 1] frame."""
    b = (np.asarray(frame).reshape(64, 64) > thresh).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(b, connectivity=8)
    return [{"row": round(float(cents[i][1]), 1), "col": round(float(cents[i][0]), 1),
             "area": int(stats[i, cv2.CC_STAT_AREA])} for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def positions(name, video):
    model = load_model(name, 0)
    out = {}
    for k in (1, 2, 3):
        X, Y = test_window(k)
        P = predict(model, X[video:video + 1])[0]
        out[f"{k}x"] = {"last_input": blobs(X[video, 4]),
                        "true": [blobs(Y[video, h]) for h in range(2)],
                        "pred": [blobs(P[h]) for h in range(2)],
                        "pred_peak": [round(float(P[h].max()), 3) for h in range(2)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    video = load_json("fig5_video.json")["test_index"]
    for name in args.names:
        out = positions(name, video)
        save_json(f"positions_{name}_video{9000 + video}.json", out)
        for speed, d in out.items():
            print(name, speed, "last", [(b["row"], b["col"]) for b in d["last_input"]])
            for h in range(2):
                print("   h=%d true" % (h + 1), [(b["row"], b["col"]) for b in d["true"][h]],
                      "pred", [(b["row"], b["col"], b["area"]) for b in d["pred"][h]])


if __name__ == "__main__":
    main()
