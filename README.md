# AI651 Deep Learning for Space, Time and Graphs: Assignment 2

| Part | Location | Content |
|---|---|---|
| Question 1: River Forecasting Proposal | [`Question1/report/`](Question1/report/) | LaTeX design proposal (six required headings) |
| Question 2: Motion or Memory? | [`Question2/`](Question2/) | ConvLSTM forecaster on Moving MNIST, stress tests, models A-D, symmetrised final model |

## Report

The combined report (both questions) is built from [`report/main.tex`](report/main.tex):

```bash
cd report
latexmk -pdf main.tex
```

It needs the Question 2 figures in `Question2/figures/` (`sh Question2/run_all.sh` or the commands in
[`Question2/README.md`](Question2/README.md)); the tables are in `report/tables/`.

## Question 1

Build the proposal with any LaTeX distribution (or Overleaf):

```bash
cd Question1/report
pdflatex main.tex && pdflatex main.tex
```

## Question 2

See [`Question2/README.md`](Question2/README.md) for setup and the commands that reproduce every table and figure.
