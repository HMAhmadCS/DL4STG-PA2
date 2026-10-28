# Question 2: Motion or Memory?

A ConvLSTM forecaster on Moving MNIST (5 frames in, 5 frames out), stress-tested with mirror,
rotation and speed-up.

## Setup

```bash
cd Question2
python -m venv .venv
.venv/Scripts/activate          # Windows; use .venv/bin/activate on Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Moving MNIST is downloaded to `Question2/data/` by `torchvision.datasets.MovingMNIST(split=None)`.
Videos are split by index: 0-7999 train, 8000-8999 validation, 9000-9999 test.
All commands below are run from `Question2/`.

## Code layout

| File | Purpose |
|---|---|
| `q2/data.py` | loading and splitting, the 8 symmetries G, speed windows |
| `q2/models.py` | ConvLSTM (model A), model D (patch 8), the symmetrised forecaster, Blank and Copy-last |
| `q2/train.py` | training protocol for A, B, C, D and D+B (`--resume` continues an interrupted run) |
| `q2/check_transforms.py` | Table 1 |
| `q2/optical_flow.py` | Figure 1 |
| `q2/evaluate.py` | E(5), E(2), MSE_h, consistency gaps C_T, transformed errors E_T, inference timing |
| `q2/run_eval.py` | computes and caches all test metrics of a forecaster |
| `q2/plots.py` | frame grids, animated GIFs, digit-overlap detection |
| `q2/baseline.py` | Table 2, Figure 2 |
| `q2/test_baseline.py` | Tables 3-4, Figures 3-5 and the Figure 3 GIFs |
| `q2/compare.py` | Tables 5-6 |
| `q2/final.py` | Figure 6 and its GIFs |
| `run_all.sh` | every step below in order, restartable |

## Reproducing

```bash
python -m q2.check_transforms                 # Table 1
python -m q2.optical_flow                     # Figure 1
python -m q2.train --model A --seed 0         # and --seed 1
python -m q2.baseline                         # Table 2, Figure 2
python -m q2.run_eval A Blank CopyLast --timing
python -m q2.test_baseline                    # Tables 3-4, Figures 3-5
# commit predictions/Q2_11.md before training B-D
python -m q2.train --model B --seed 0         # likewise C, D, DB; seeds 0 and 1
python -m q2.run_eval B C D DB --timing
python -m q2.compare table5                   # Table 5
python -m q2.run_eval sym-B sym-D sym-DB --timing
python -m q2.compare table6 --final DB        # Table 6
python -m q2.final --final D                  # Figure 6 (or --final DB)
```

`sh run_all.sh` runs all of this in order with 12 threads and waits for the Q2.11 commit before
training B-D.

Generated tables (`results/`), figures (`figures/`), checkpoints and training logs are not tracked;
the GIFs (`gifs/`) are.
