"""Render dismantling figures from cached results.

Run ``dismantling_experiment.py`` once to create ``dismantling_plot_data.json``. This script
only reads that file, so changing figure layout does not rerun experiments.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_CONFIG = {
    # Deliberately separated categorical colors.  STRC is the only red
    # series; no other method uses a nearby red/pink hue.
    "STRC": {"color": "#D62728", "marker": "o", "ms": 7.5, "z": 20},
    "DC": {"color": "#1F77B4", "marker": "o", "ms": 6.0, "z": 8},
    "BC": {"color": "#9467BD", "marker": "o", "ms": 6.0, "z": 8},
    "CC": {"color": "#2CA02C", "marker": "o", "ms": 6.0, "z": 8},
    "PR": {"color": "#E6AB02", "marker": "o", "ms": 6.0, "z": 8},
    "EC": {"color": "#FF7F0E", "marker": "o", "ms": 6.0, "z": 8},
    "K-core": {"color": "#8C564B", "marker": "o", "ms": 6.0, "z": 8},
    "NC": {"color": "#17BECF", "marker": "o", "ms": 6.0, "z": 8},
    "CE": {"color": "#BCBD22", "marker": "o", "ms": 6.0, "z": 8},
    "RT": {"color": "#008C95", "marker": "o", "ms": 6.0, "z": 8},
    "CI": {"color": "#7F7F7F", "marker": "o", "ms": 6.0, "z": 8},
    "ER": {"color": "#003F5C", "marker": "o", "ms": 6.0, "z": 8},
    "KI": {"color": "#C23B8E", "marker": "o", "ms": 6.0, "z": 8},
    "AER": {"color": "#222222", "marker": "o", "ms": 6.0, "z": 8},
}


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 17,
        "axes.labelsize": 20,
        "axes.titlesize": 21,
        "legend.fontsize": 16,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.15,
        "xtick.major.width": 1.15,
        "ytick.major.width": 1.15,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def plot_group(group_name, group, top_fraction, output_dir):
    methods = group["methods"]
    datasets = group["datasets"]
    ncols = 3
    nrows = max(1, int(np.ceil(len(datasets) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(25.5, 7.6 * nrows),
                             dpi=300, squeeze=False)
    axes_flat = axes.ravel()
    legend_handles = {}

    for index, (network, values) in enumerate(datasets.items()):
        ax = axes_flat[index]
        x = np.asarray(values["x"], dtype=float)
        minimum = 1.0
        for method in methods:
            cfg = STYLE_CONFIG[method]
            mean = np.asarray(values["mean"][method], dtype=float)
            std = np.asarray(values["std"][method], dtype=float)
            ax.fill_between(x, np.clip(mean - std, 0, 1),
                            np.clip(mean + std, 0, 1), color=cfg["color"],
                            alpha=0.16 if method == "STRC" else 0.07,
                            zorder=cfg["z"] - 1)
            line, = ax.plot(x, mean, color=cfg["color"],
                            linewidth=2.7 if method == "STRC" else 2.1,
                            alpha=1.0 if method == "STRC" else 0.85,
                            marker=cfg["marker"], markersize=cfg["ms"],
                            markevery=max(1, len(x) // 12),
                            markeredgewidth=0.8, markeredgecolor="white",
                            zorder=cfg["z"])
            legend_handles.setdefault(method, line)
            minimum = min(minimum, float(np.min(mean - std)))
        ax.grid(True, linestyle="--", linewidth=0.65, color="#D9DDE3", alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{network}\n($N={values['N']}$, $E={values['E']}$)",
                     fontsize=20, pad=10, fontweight="600")
        ax.set_ylim(max(0.0, minimum - 0.02), 1.01)
        ax.set_xlim(0, top_fraction * 100)
        ax.set_xticks(np.linspace(0, top_fraction * 100, 6))
        ax.set_xticklabels([f"{int(v)}%" for v in np.linspace(0, top_fraction * 100, 6)])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel("Fraction of removed nodes $Q$", fontsize=19, labelpad=7)
        ax.set_ylabel("Relative GCC size $s(Q)$", fontsize=19, labelpad=7)

    for ax in axes_flat[len(datasets):]:
        ax.set_visible(False)
    labels = ["ER (approx.)" if group_name == "real" and m == "ER" else m
              for m in legend_handles]
    # Put the shared legend above the panels.  This keeps it close to the
    # actual plots and avoids the large bottom gap of the previous layout.
    fig.legend([legend_handles[m] for m in legend_handles], labels,
               loc="upper center", bbox_to_anchor=(0.5, 0.995),
               fontsize=16, ncol=min(7, len(labels)), frameon=False,
               handlelength=1.8, handletextpad=0.45,
               columnspacing=0.8, labelspacing=0.5)
    # No group-level suptitle: the manuscript caption supplies the context.
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    fig.subplots_adjust(wspace=0.30, hspace=0.52)
    stem = output_dir / f"Network_Dismantling_{'Synthetic' if group_name == 'synthetic' else 'Real'}"
    fig.savefig(stem.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf/.png/.svg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR.parent / "results" / "dismantling_plot_data.json")
    parser.add_argument("--group", choices=("synthetic", "real", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR.parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    groups = ("synthetic", "real") if args.group == "both" else (args.group,)
    for group_name in groups:
        plot_group(group_name, payload[group_name], float(payload["top_fraction"]), args.output_dir)


if __name__ == "__main__":
    main()
