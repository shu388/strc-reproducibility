# STRC Reproducibility Code

This repository contains the source code for the STRC node-ranking experiments
in the accompanying manuscript. It includes network dismantling, SIR
spreading, approximation-accuracy, projection-dimension, runtime-complexity,
dataset-statistics, and LFR sensitivity experiments.

## Requirements

Python 3.10 or newer is recommended. Install the dependencies in a virtual
environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Laplacian systems are solved with PyAMG's smoothed-aggregation AMG
backend. This is the solver backend used by the released reproduction code;
the original CMG/PCG implementation is not required.

## Data

Raw datasets are deliberately excluded from Git because of their size and
licensing/distribution conditions. Download the files listed in
[`data/README.md`](data/README.md) and place them under `data/`.

All empirical-network loaders remove self-loops and analyze the largest
connected component. Synthetic ER, WS, and BA networks are generated at run
time.

## Running experiments

Run commands from the repository root. Generated figures and tables are saved
under `results/`.

```bash
python src/dismantling_experiment.py
python src/spreading_experiment.py
python src/complexity_benchmark.py --quick       # smoke test
python src/complexity_benchmark.py                # full runtime benchmark
python src/k_scaling_analysis.py
python src/approximation_accuracy.py
python src/dataset_statistics.py --data-dir data
python src/lfr_bridge_experiment.py
```

The main experiments use a dismantling range of 0--10% removed nodes and a
5% seed fraction for spreading. The default STRC projection dimension is
`K=800` in the main experiments; the runtime benchmark exposes `--k` for
explicit control. Random seeds are defined in the scripts for reproducible
ranking and simulation runs.

## Repository layout

```text
src/       experiment and solver source code
data/      dataset download instructions (raw files omitted)
results/   generated figures/tables (ignored by Git)
```

Generated PDFs, PNGs, SVGs, CSVs, logs, local virtual environments, and the
original large archive are intentionally not part of this release.
