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
| `q2/models.py` | ConvLSTM (model A) and the reference predictors Blank and Copy-last |
| `q2/train.py` | training protocol |
| `q2/check_transforms.py` | Table 1 |
| `q2/optical_flow.py` | Figure 1 |
| `q2/evaluate.py` | E(5), E(2), MSE_h, consistency gaps C_T, transformed errors E_T, inference timing |
| `q2/run_eval.py` | computes and caches all test metrics of a forecaster |
| `q2/plots.py` | frame grids, animated GIFs, digit-overlap detection |
| `q2/baseline.py` | Table 2, Figure 2 |

## Reproducing

```bash
python -m q2.check_transforms                 # Table 1
python -m q2.optical_flow                     # Figure 1
python -m q2.train --model A --seed 0         # and --seed 1
python -m q2.baseline                         # Table 2, Figure 2
```

Generated tables (`results/`), figures (`figures/`), checkpoints and training logs are not tracked.
