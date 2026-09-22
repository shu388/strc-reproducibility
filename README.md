# STRC reproducibility code

Source code for the STRC experiments in the revised manuscript. The release
contains implementations, experiment drivers, and plotting code only. Raw
networks and generated figures/tables are not committed.

## Setup

Use Python 3.10 or newer from the repository root:

```bash
python -m venv .venv
# Activate .venv (on Windows PowerShell: .venv\Scripts\Activate.ps1)
python -m pip install -r requirements.txt
```

The scalable STRC implementation uses Johnson--Lindenstrauss projection and
PyAMG smoothed aggregation for Laplacian solves. The released Fast-TreeC
reproduction also uses SA-AMG; the original CMG/PCG backend is not bundled.
All networks are treated as simple, undirected, unweighted graphs. Download
the six real-world edge lists following [data/README.md](data/README.md) and
put them in `data/`. Scripts skip missing real-world networks where noted;
the synthetic experiments generate their own graphs.

## Experiments

Run these from the repository root. Outputs go to the ignored `results/`
directory. Full experiments can be expensive, especially exact node-deletion
baselines and the large-network sensitivity analysis.

| Manuscript analysis | Command |
| --- | --- |
| Dismantling, 0--10% removals | `python src/dismantling_experiment.py` |
| SIR spreading, 5% seed fraction | `python src/spreading_experiment.py` |
| Plot the main experiment results | `python src/plot_dismantling.py` and `python src/plot_spreading.py` |
| M1: exact STRC/ER/KI/AER on real-network 80-node subgraphs | `python src/m1_subgraph_consistency.py` |
| M2: STRC/NC/SR correlations and exact-subgraph validation | `python src/m2_spectral_correlations.py` |
| M3: synthetic K-sensitivity and paired task outcomes | `python src/m3_k_sensitivity.py` |
| M3: real-network paired K-sensitivity | `python src/m3_real_k_sensitivity.py` |
| M3: WS rank-agreement threshold | `python src/m3_ws_k_threshold.py` |
| Runtime benchmark | `python src/complexity_benchmark.py` |
| Projection-dimension and accuracy diagnostics | `python src/k_scaling_analysis.py` and `python src/approximation_accuracy.py` |
| Dataset statistics | `python src/dataset_statistics.py` |
| LFR bridge sensitivity | `python src/lfr_bridge_experiment.py` |

The main experiments use `K=400` for approximate STRC. The synthetic M3
study uses 50 independent connected BA/ER/WS graphs with 200 nodes and 400
edges, nested `K={50,200,400}` projections, and graph-level 95% confidence
intervals. The real M3 study uses five paired projection seeds per network;
its intervals reflect projection-seed variability on each fixed network, not
independent graph draws. The extended WS study scans `K=400...4000` against
exact STRC; its grounded sparse-LU solves isolate projection error from AMG
solver error. `K=400` is an operational setting, not a universal guarantee of
Spearman rank correlation at least 0.95.

For a short runtime check, use `python src/complexity_benchmark.py --quick`.
The M1/M2/M3 and benchmark scripts expose `--help` for configurable sample
counts, K values, input locations and output directories. M1 evaluates ER,
KI, and AER exactly only on small induced subgraphs; on full real networks
the main experiment uses the scalable ER approximation and does not claim
full-network KI/AER.

## Layout

```text
src/       experiment, solver, baseline and plotting source code
data/      dataset download instructions (raw files omitted)
results/   generated outputs (ignored by Git)
```

Plots with hard-coded values in `visualize_auc.py` and
`visualize_spreading.py` are legacy presentation scripts. To plot newly
generated main-experiment data, use `plot_dismantling.py` and
`plot_spreading.py` instead.
