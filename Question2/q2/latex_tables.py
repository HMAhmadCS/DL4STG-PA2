"""LaTeX versions of Tables 1-6 for the report, built from the cached results.

Run: python -m q2.latex_tables [--out ../report/tables]
Tables 5 and 6 are written only once the metrics of all their models exist.
"""
import argparse
import json
import os

import numpy as np

from .utils import RESULTS_DIR, ROOT, ensure, load_json

SEEDS = (0, 1)
NAMES = {"A": "A", "B": "B", "C": "C", "D": "D", "DB": "D+B", "Blank": "Blank", "CopyLast": "Copy-last",
         "sym-B": r"$\bar f$(B)", "sym-D": r"$\bar f$(D)", "sym-DB": r"$\bar f$(D+B)"}


def num(x, sci=False):
    """Three significant figures; scientific notation below 1e-3 (or when sci is set)."""
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or sci:
        m, e = f"{x:.2e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{x:#.3g}".rstrip(".")


def cell(vals, sci=False):
    if len(vals) == 1:
        return num(vals[0], sci)
    return f"{num(float(np.mean(vals)), sci)} ({num(vals[0], sci)}, {num(vals[1], sci)})"


def exists(fn):
    return os.path.exists(os.path.join(RESULTS_DIR, fn))


def metrics(name):
    if name in ("Blank", "CopyLast"):
        return [load_json(f"metrics_{name}.json")]
    return [load_json(f"metrics_{name}_s{s}.json") for s in SEEDS]


def tabular(cols, header, rows):
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table1():
    rows = [[r["check"].replace("|", r"$|$").replace("^-1", r"$^{-1}$").replace("^4", r"$^4$"),
             r["expected"], r["yours"].replace("[", "").replace("]", "")] for r in load_json("table1.json")]
    return tabular("lll", ["Check", "Expected", "Yours"], rows)


def table2():
    t = load_json("table2.json")
    rows = []
    for name, ps in t.items():
        rows.append([NAMES[name]] + [cell([p["MSE_h"][h] for p in ps]) for h in range(5)]
                    + [cell([p["E5"] for p in ps]), f"{ps[0]['params']:,}"])
    return tabular("l" + "c" * 7, ["Model"] + [f"MSE$_{h}$" for h in range(1, 6)] + ["$E^{(5)}$", "Params"], rows)


def table3():
    keys = ["E5_1x", "C_M", "E_M", "C_R", "E_R"]
    rows = [[NAMES[n]] + [cell([m[k] for m in metrics(n)], sci=False) for k in keys]
            for n in ("A", "Blank", "CopyLast")]
    return tabular("lccccc", ["Model", "$E^{(5)}$", "$C_M$", "$E_M$", "$C_R$", "$E_R$"], rows)


def table4():
    keys = ["E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x"]
    rows = [[NAMES[n]] + [cell([m[k] for m in metrics(n)]) for k in keys] for n in ("A", "CopyLast")]
    return tabular("lccccc", ["Model", r"$E^{(2)}$ 1$\times$", r"$E^{(2)}$ 2$\times$", r"$E^{(2)}$ 3$\times$",
                              r"$E^{(5)}$ 1$\times$", r"$E^{(5)}$ 2$\times$"], rows)


def train_time(base, s):
    with open(os.path.join(ROOT, "logs", f"{base}_s{s}.json")) as f:
        return json.load(f)["mean_epoch_time_s"]


def comparison(names, sym):
    """Models as columns, quantities as rows (Tables 5 and 6)."""
    ms = {n: metrics(n) for n in names}
    rows = []
    for k, label in [("E5_1x", r"$E^{(5)}$ 1$\times$"), ("C_M", "$C_M$"), ("E_M", "$E_M$"), ("C_R", "$C_R$"),
                     ("E_R", "$E_R$"), ("E2_1x", r"$E^{(2)}$ 1$\times$"), ("E2_2x", r"$E^{(2)}$ 2$\times$"),
                     ("E2_3x", r"$E^{(2)}$ 3$\times$")]:
        rows.append([label] + [cell([m[k] for m in ms[n]], sci=sym and k.startswith("C_")) for n in names])
    rows.append(["Trainable parameters"] + [f"{ms[n][0]['params']:,}" for n in names])
    rows.append(["Training time (s/epoch)"]
                + [cell([train_time(n.replace("sym-", ""), s) for s in SEEDS]) for n in names])
    rows.append(["Inference time (s, 1000 videos)"]
                + [cell([load_json(f"timing_{n}_s{s}.json")["inference_s"] for s in SEEDS]) for n in names])
    return tabular("l" + "c" * len(names), [""] + [NAMES[n] for n in names], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "report", "tables"))
    args = ap.parse_args()
    out = ensure(args.out)
    tables = {"table1": table1, "table2": table2, "table3": table3, "table4": table4}
    done = [n for n in ("A", "B", "C", "D", "DB")
            if all(exists(f"metrics_{n}_s{s}.json") and exists(f"timing_{n}_s{s}.json") for s in SEEDS)]
    if done[:4] == ["A", "B", "C", "D"]:
        tables["table5"] = lambda: comparison(done, sym=False)  # D+B included once trained
    sym = [n for n in ("sym-B", "sym-D", "sym-DB") if all(exists(f"timing_{n}_s{s}.json") for s in SEEDS)]
    if len(sym) == 3:
        tables["table6"] = lambda: comparison(sym, sym=True)
    for name, fn in tables.items():
        with open(os.path.join(out, f"{name}.tex"), "w") as f:
            f.write(fn())
        print("wrote", os.path.join(out, f"{name}.tex"))


if __name__ == "__main__":
    main()
