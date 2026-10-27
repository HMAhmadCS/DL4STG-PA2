"""Table 1: sanity checks of the transformation and window functions.

Run: python -m q2.check_transforms
"""
import torch

from .data import mirror, rotate, window
from .utils import save_json


def bright_pixel(v):
    """(frame, row, column) of the single nonzero pixel, indices from 0 as in the handout."""
    f, _, r, c = (v != 0).nonzero()[0].tolist()
    return (f, r, c)


def frames_of(X):
    return [int(v) for v in X[:, 0, 0, 0].tolist()]


def main():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1  # frame index 7 as specified in the handout

    rows = []

    def exact(name, a, b):
        d = (a - b).abs().max().item()
        rows.append((name, "0", f"{d:g} (torch.equal={torch.equal(a, b)})"))

    exact("(a) max|M(M(x)) - x|", mirror(mirror(x)), x)
    exact("(b) max|R^4(x) - x|", rotate(rotate(rotate(rotate(x)))), x)
    exact("(c) max|M R M(x) - R^-1(x)|", mirror(rotate(mirror(x))), rotate(x, -1))

    rows.append(("(d) bright pixel of M(p)", "(7, 10, 43)", str(bright_pixel(mirror(p)))))
    rows.append(("(e) bright pixel of R(p)", "(7, 43, 10)", str(bright_pixel(rotate(p)))))

    # Video whose frame f (1-based) is filled with the value f.
    v = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64)
    expected = {1: "1-5 / 6-10", 2: "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", 3: "1, 4, 7, 10, 13 / 16, 19"}
    for k, tag in zip((1, 2, 3), "fgh"):
        X, Y = window(v, t=1, k=k)
        rows.append((f"({tag}) {k}x input / target frames", expected[k],
                     f"{frames_of(X)} / {frames_of(Y)}"))

    w = max(len(r[0]) for r in rows)
    print(f"{'Check':<{w}} | Expected | Yours")
    for r in rows:
        print(f"{r[0]:<{w}} | {r[1]} | {r[2]}")
    save_json("table1.json", [{"check": a, "expected": b, "yours": c} for a, b, c in rows])


if __name__ == "__main__":
    main()
