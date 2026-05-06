# Rounding Weighted Sums of Positive Semidefinite Matrices

This repository contains implementations of randomized rounding algorithms for the paper "Rounding Weighted Sums of Postive Semidefinite Matrices", including both linear optimal experimental design and graph augmentation problems.

## Overview

Given a set of design points $x_1, \ldots, x_m \in \mathbb{R}^n$ and a budget $k$, the goal is to select a subset $J \subseteq [m]$ with $|J| \leq k$ that optimizes a function of the information matrix $M + \sum_{i \in J} x_i x_i^\top$ given a prior matrix $M$.

### Supported Optimality Criteria

| Criterion | Symbol | Objective |
|-----------|--------|-----------|
| E-optimal | `E` | Maximize minimum eigenvalue |
| D-optimal | `D` | Maximize log-determinant |
| Effective Resistance  | `A` | Minimize trace of inverse |
| Algebraic Connectivity | `Mac` | Maximize 2nd smallest eigenvalue |

### Approximation Methods

| Method | Description |
|--------|-------------|
| `Split` | Split Randomized Rounding|
| `Sparse` | Sparse Randomized Rounding |
| `Weight` | Weighted Randomized Rounding |
| `Exchange` | Fedorov's Exchange Algorithm |
| `Uniform` | Uniform Sampling (baseline) |

---

## Project Structure

```
.
├── core/
│   ├── data_utils.py     # DataGenerator: synthetic data generation
│   ├── data_loaders.py   # Loaders for all datasets
│   ├── solver.py         # RWSSolver: convex relaxation + rounding methods
│   ├── utils.py          # Shared utilities (calc_ratio, grid search, etc.)
│   ├── runner.py         # run_experiment: main pipeline driver
│   └── visualize.py      # Plotting and CSV saving
│
├── experiments/
│   ├── synthetic.py        # Synthetic block-diagonal data
│   ├── housing.py          # Housing dataset (LIBSVM)
│   ├── regensburg.py       # Regensburg Pediatric Appendicitis (UCI)
│   ├── graph_synthetic.py  # Graph augmentation: WS / BA graphs
│   └── graph_real.py       # Graph augmentation: Karate, Dolphins, TrainTerrorists
│
├── results/              # Auto-generated output directory (CSV + PNG)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/haeseong-yang/Rounding-Weighted-Sums-of-PSD-matrices.git
cd Rounding-Weighted-Sums-of-PSD-matrices
pip install -r requirements.txt
```

For best performance, install [MOSEK](https://www.mosek.com/) (free academic license available). The solver automatically falls back to SCS if MOSEK is unavailable.

---

## Usage

Each experiment script is self-contained and prompts interactively for the configuration (e.g., optimality criterion, graph type, dataset). Run them as Python modules from the repository root:

```bash
# Synthetic data experiment (prompts for E or D)
python -m experiments.synthetic

# Housing dataset (prompts for E or D)
python -m experiments.housing

# Regensburg dataset (D-optimal only)
python -m experiments.regensburg

# Synthetic graph (prompts for WS/BA and Mac/A)
python -m experiments.graph_synthetic

# Real graphs (prompts for dataset selection)
python -m experiments.graph_real
```

Pressing Enter at any prompt accepts the default value shown in brackets.

---

## Datasets

| Dataset | Source | Task |
|---------|--------|------|
| Synthetic | Generated | Block-diagonal normal |
| Housing | [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) | Regression |
| Regensburg | [UCI](https://archive.ics.uci.edu/dataset/938) | Clinical |
| Karate | NetworkX built-in | Graph augmentation |
| Dolphins | [Newman](https://websites.umich.edu/~mejn/netdata/) | Graph augmentation |
