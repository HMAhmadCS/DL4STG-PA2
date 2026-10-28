"""Section 4 (What Can Data Fix?): Table 5.  Section 5 (The Final Model): Table 6.

Run: python -m q2.compare table5
     python -m q2.compare table6 [--final DB]
"""
import argparse
import json
import os

import numpy as np

from .run_eval import get_metrics, get_timing
from .utils import ROOT, save_json, sig3

SEEDS = (0, 1)
KEYS = ["E5_1x", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x"]


def train_time(name, seed):
    with open(os.path.join(ROOT, "logs", f"{name}_s{seed}.json")) as f:
        return json.load(f)["mean_epoch_time_s"]


def fmt(vals):
    return f"{sig3(float(np.mean(vals)))} ({sig3(vals[0])}, {sig3(vals[1])})"


def table(names, title, out_name):
    rows = {}
    print(title)
    print("model | " + " | ".join(KEYS) + " | params | train s/epoch | inference s (1000 videos)")
    for name in names:
        base = name[4:] if name.startswith("sym-") else name
        ms = [get_metrics(name, s) for s in SEEDS]
        tr = [train_time(base, s) for s in SEEDS]
        inf = [get_timing(name, s)["inference_s"] for s in SEEDS]
        rows[name] = {"metrics": [{k: m[k] for k in KEYS} for m in ms], "params": ms[0]["params"],
                      "train_s_per_epoch": tr, "inference_s": inf}
        cells = [fmt([m[k] for m in ms]) for k in KEYS]
        print(name, "|", " | ".join(cells), "|", ms[0]["params"], "|", fmt(tr), "|", fmt(inf))
    save_json(out_name, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["table5", "table6"])
    ap.add_argument("--final", default="D", help="model wrapped as the final model (D or DB)")
    args = ap.parse_args()
    if args.which == "table5":
        table(["A", "B", "C", "D"], "Table 5", "table5.json")
    else:
        names = ["sym-B", "sym-D"] + ([f"sym-{args.final}"] if args.final != "D" else [])
        table(names, "Table 6", "table6.json")


if __name__ == "__main__":
    main()
