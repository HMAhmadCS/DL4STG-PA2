"""Compute (and cache) all test-set metrics of the requested forecasters.

Run: python -m q2.run_eval A Blank CopyLast           (learned models: seeds 0 and 1)
     python -m q2.run_eval sym-B sym-D --timing
"""
import argparse
import os

from .evaluate import evaluate_all, hardware, inference_time, is_learned, load_model
from .models import n_params
from .utils import RESULTS_DIR, load_json, save_json

SEEDS = (0, 1)


def metrics_name(name, seed):
    return f"metrics_{name}" + (f"_s{seed}" if seed is not None else "") + ".json"


def get_metrics(name, seed, recompute=False):
    fn = metrics_name(name, seed)
    if not recompute and os.path.exists(os.path.join(RESULTS_DIR, fn)):
        return load_json(fn)
    model = load_model(name, seed)
    out = evaluate_all(model)
    out["params"] = n_params(model)
    save_json(fn, out)
    return out


def get_timing(name, seed, repeats=3):
    fn = f"timing_{name}_s{seed}.json"
    if os.path.exists(os.path.join(RESULTS_DIR, fn)):
        return load_json(fn)
    t = {"inference_s": inference_time(load_model(name, seed), repeats=repeats), "hardware": hardware()}
    save_json(fn, t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    for name in args.names:
        base = name[4:] if name.startswith("sym-") else name
        seeds = SEEDS if is_learned(base) else (None,)
        for s in seeds:
            m = get_metrics(name, s, args.recompute)
            summary = {k: v for k, v in m.items() if not isinstance(v, list)}
            print(name, s, summary, flush=True)
            if args.timing and is_learned(base):
                print("  timing", get_timing(name, s)["inference_s"], flush=True)


if __name__ == "__main__":
    main()
