"""Runtime experiment for STRC and approximate spanning edge centrality.

This script extends the original STRC scalability figure with the approximate
SC algorithm of Mavroforakis et al. Both approximation methods use the same
graph, projection dimension, Rademacher projection seed, AMG solver, and
solver tolerance in every paired trial. Timings include construction of the
projected systems, their solution, and method-specific score evaluation.
"""

import argparse
import csv
import gc
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import scipy.linalg as linalg

from laplacian_solver import solve_laplacian_sdd


warnings.filterwarnings("ignore")

DEFAULT_SIZES = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
METHODS = ("orig", "fast", "strc_approx", "sc_approx")
TOLERANCE = 1e-5
RAW_FIELDS = [
    "N",
    "M",
    "trial",
    "graph_seed",
    "projection_seed",
    "K",
    "tolerance",
    "method",
    "seconds",
]


def _projected_embedding(G, K, projection_seed):
    """Compute the JL embedding used by approximate STRC."""
    nodes = list(G.nodes())
    edges = list(G.edges())
    node_idx = {node: index for index, node in enumerate(nodes)}
    n_edges = len(edges)

    laplacian = nx.laplacian_matrix(G, nodelist=nodes).tocsr()
    incidence = nx.incidence_matrix(
        G,
        nodelist=nodes,
        edgelist=edges,
        oriented=True,
    ).T

    rng = np.random.default_rng(projection_seed)
    projection = rng.integers(0, 2, size=(K, n_edges), dtype=np.int8)
    projection = projection.astype(np.float64)
    projection *= 2.0
    projection -= 1.0
    projection /= np.sqrt(K)

    projected_rhs = incidence.T @ projection.T
    embedding, _ = solve_laplacian_sdd(
        laplacian,
        projected_rhs,
        tolerance=TOLERANCE,
    )
    return nodes, edges, node_idx, embedding


def time_original_exact(G):
    """Time the direct Matrix-Tree-Theorem STRC implementation."""
    start = perf_counter()
    nodes = list(G.nodes())
    laplacian = nx.laplacian_matrix(G).toarray()
    np.linalg.slogdet(laplacian[:-1, :-1])
    for node in nodes:
        subgraph = G.copy()
        subgraph.remove_node(node)
        if nx.is_connected(subgraph):
            sub_laplacian = nx.laplacian_matrix(subgraph).toarray()
            np.linalg.slogdet(sub_laplacian[:-1, :-1])
    return perf_counter() - start


def time_fast_exact(G):
    """Time the exact closed-form STRC* implementation."""
    start = perf_counter()
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_idx = {node: index for index, node in enumerate(nodes)}

    laplacian = nx.laplacian_matrix(G).toarray()
    centering = np.ones((n_nodes, n_nodes)) / n_nodes
    laplacian_pinv = linalg.inv(laplacian + centering) - centering

    for node in nodes:
        index = node_idx[node]
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        if degree <= 1:
            continue
        neighbor_indices = [node_idx[neighbor] for neighbor in neighbors]
        pinv_neighbors = laplacian_pinv[np.ix_(neighbor_indices, neighbor_indices)]
        pinv_node = laplacian_pinv[neighbor_indices, index]
        local_resistance = (
            pinv_neighbors
            - pinv_node[:, None]
            - pinv_node[None, :]
            + laplacian_pinv[index, index]
        )
        local_matrix = (
            np.eye(degree)
            - local_resistance
            + np.ones((degree, degree)) / degree
        )
        np.linalg.det(local_matrix)
    return perf_counter() - start


def time_strc_approx(G, K, projection_seed):
    """Time the proposed approximate STRC algorithm end to end."""
    start = perf_counter()
    nodes, _, node_idx, embedding = _projected_embedding(G, K, projection_seed)

    for node in nodes:
        index = node_idx[node]
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        if degree <= 1:
            continue
        neighbor_indices = [node_idx[neighbor] for neighbor in neighbors]
        differences = embedding[neighbor_indices, :] - embedding[index, :]
        local_resistance = differences @ differences.T
        local_matrix = (
            np.eye(degree)
            - local_resistance
            + np.ones((degree, degree)) / degree
        )
        np.linalg.det(local_matrix)
    return perf_counter() - start


def time_sc_approx(G, K, projection_seed):
    """Time approximate spanning edge centrality (SC) end to end.

    For an unweighted graph, the spanning centrality of edge (u, v) equals
    its effective resistance. The JL embedding estimates all such values as
    ||z_u-z_v||^2.
    """
    start = perf_counter()
    _, edges, node_idx, embedding = _projected_embedding(G, K, projection_seed)
    source = np.fromiter((node_idx[u] for u, _ in edges), dtype=np.int64)
    target = np.fromiter((node_idx[v] for _, v in edges), dtype=np.int64)
    differences = embedding[source, :] - embedding[target, :]
    approximate_sc = np.einsum("ij,ij->i", differences, differences)
    if not np.all(np.isfinite(approximate_sc)):
        raise RuntimeError("Approximate SC produced non-finite scores")
    return perf_counter() - start


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 11.5,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_summary(path):
    """Load an existing summary CSV so figure styling can be revised cheaply."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["N"] = int(row["N"])
        row["mean_seconds"] = float(row["mean_seconds"])
        row["std_seconds"] = float(row["std_seconds"])
        row["n_trials"] = int(row["n_trials"])
    return rows


def _summarize(raw_rows, sizes):
    summary_rows = []
    for n_nodes in sizes:
        rows_n = [row for row in raw_rows if row["N"] == n_nodes]
        for method in METHODS:
            values = [row["seconds"] for row in rows_n if row["method"] == method]
            if values:
                summary_rows.append(
                    {
                        "N": n_nodes,
                        "method": method,
                        "mean_seconds": float(np.mean(values)),
                        "std_seconds": float(np.std(values)),
                        "n_trials": len(values),
                    }
                )

        paired = {}
        for row in rows_n:
            paired.setdefault(row["trial"], {})[row["method"]] = row["seconds"]
        ratios = [
            values["strc_approx"] / values["sc_approx"]
            for values in paired.values()
            if "strc_approx" in values and "sc_approx" in values
        ]
        if ratios:
            summary_rows.append(
                {
                    "N": n_nodes,
                    "method": "strc_to_sc_ratio",
                    "mean_seconds": float(np.mean(ratios)),
                    "std_seconds": float(np.std(ratios)),
                    "n_trials": len(ratios),
                }
            )
    return summary_rows


def _summary_arrays(summary_rows, method, sizes):
    by_n = {
        row["N"]: (row["mean_seconds"], row["std_seconds"])
        for row in summary_rows
        if row["method"] == method
    }
    means = np.array([by_n.get(n, (np.nan, np.nan))[0] for n in sizes])
    stds = np.array([by_n.get(n, (np.nan, np.nan))[1] for n in sizes])
    return means, stds


def _style_axis(ax):
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, which="major", ls="--", color="#D0D0D0", linewidth=0.55, alpha=0.8)
    ax.grid(True, which="minor", ls=":", color="#E5E5E5", linewidth=0.35, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(which="major", length=4.0, width=1.0, pad=3)
    ax.tick_params(which="minor", length=2.2, width=0.8)


def _plot_results(
    summary_rows,
    sizes,
    output_prefix,
    show,
    sc_label="Approx. SC (baseline)",
    ratio_label="SC",
    include_fast=True,
):
    colors = {
        "orig": "#E64B35",
        "fast": "#4DBBD5",
        "strc_approx": "#00A087",
        "sc_approx": "#3C5488",
    }
    markers = {"orig": "o", "fast": "s", "strc_approx": "^", "sc_approx": "D"}
    labels = {
        "orig": r"STRC ($\mathcal{O}(N^4)$)",
        "fast": r"STRC$^*$ ($\mathcal{O}(N^3)$)",
        "strc_approx": "Approx. STRC (proposed)",
        "sc_approx": sc_label,
    }

    n_values = np.asarray(sizes)
    fig, (runtime_ax, ratio_ax) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.05),
        dpi=300,
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    for ax in (runtime_ax, ratio_ax):
        _style_axis(ax)

    for method in METHODS:
        if method == "fast" and not include_fast:
            continue
        means, stds = _summary_arrays(summary_rows, method, sizes)
        mask = np.isfinite(means)
        runtime_ax.plot(
            n_values[mask],
            means[mask],
            color=colors[method],
            marker=markers[method],
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.7,
            label=labels[method],
            zorder=4,
        )
        runtime_ax.fill_between(
            n_values[mask],
            np.maximum(means[mask] - stds[mask], 1e-8),
            means[mask] + stds[mask],
            color=colors[method],
            alpha=0.14,
            zorder=3,
        )

    orig_means, _ = _summary_arrays(summary_rows, "orig", sizes)
    fast_means, _ = _summary_arrays(summary_rows, "fast", sizes)
    base_n = n_values[0]
    if np.isfinite(orig_means[0]):
        reference_n = np.array([base_n, min(400, n_values[-1])])
        runtime_ax.plot(
            reference_n,
            orig_means[0] * (reference_n / base_n) ** 4,
            ":",
            color=colors["orig"],
            alpha=0.65,
            linewidth=1.2,
            label=r"Reference $N^4$",
        )
    if include_fast and np.isfinite(fast_means[0]):
        reference_n = np.array([base_n, min(3200, n_values[-1])])
        runtime_ax.plot(
            reference_n,
            fast_means[0] * (reference_n / base_n) ** 3,
            ":",
            color=colors["fast"],
            alpha=0.65,
            linewidth=1.2,
            label=r"Reference $N^3$",
        )
    runtime_ax.set_xscale("log")
    runtime_ax.set_yscale("log")
    runtime_ax.set_xlabel(r"Network size $N$")
    runtime_ax.set_ylabel("Wall-clock time (s)")
    runtime_ax.set_title("(a) End-to-end runtime", loc="left", fontweight="bold")
    runtime_ax.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.94,
        ncol=1,
        borderpad=0.45,
        labelspacing=0.3,
        handlelength=1.8,
    )

    ratio_means, ratio_stds = _summary_arrays(
        summary_rows, "strc_to_sc_ratio", sizes
    )
    ratio_mask = np.isfinite(ratio_means)
    ratio_ax.axhline(1.0, color="#6B7280", linestyle="--", linewidth=1.0, zorder=2)
    ratio_ax.errorbar(
        n_values[ratio_mask],
        ratio_means[ratio_mask],
        yerr=ratio_stds[ratio_mask],
        color=colors["strc_approx"],
        marker="o",
        markersize=5.0,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=1.7,
        capsize=2.5,
        zorder=4,
    )
    ratio_ax.set_xscale("log")
    ratio_ax.set_xlabel(r"Network size $N$")
    ratio_ax.set_ylabel(
        rf"Runtime ratio $T_{{\mathrm{{STRC}}}}/T_{{\mathrm{{{ratio_label}}}}}$"
    )
    ratio_ax.set_title("(b) Paired approximation comparison", loc="left", fontweight="bold")

    fig.tight_layout(w_pad=2.0)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(
            output_prefix.with_suffix(f".{extension}"),
            format=extension,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_runtime_experiment(
    sizes,
    trials,
    K,
    seed,
    original_max_n,
    exact_max_n,
    output_prefix,
    show=False,
):
    """Run paired runtime trials and save raw data, summary data, and figure."""
    try:
        import pyamg
    except ImportError as exc:
        raise RuntimeError("This experiment requires PyAMG: pip install pyamg") from exc

    output_prefix = Path(output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_prefix.with_name(output_prefix.name + "_raw.csv")

    raw_rows = []
    total_settings = len(sizes) * trials
    completed = 0
    print(
        f"Runtime benchmark: {len(sizes)} sizes x {trials} trials, "
        f"K={K}, tolerance={TOLERANCE:g}"
    )

    for size_index, n_nodes in enumerate(sizes):
        for trial in range(trials):
            graph_seed = seed + 10000 * size_index + trial
            projection_seed = seed + 1000000 + 10000 * size_index + trial
            graph = nx.barabasi_albert_graph(n_nodes, 3, seed=graph_seed)
            timings = {}

            # Alternate the paired approximation order to reduce order bias.
            approximate_order = (
                ("strc_approx", "sc_approx")
                if trial % 2 == 0
                else ("sc_approx", "strc_approx")
            )
            for method in approximate_order:
                gc.collect()
                if method == "strc_approx":
                    timings[method] = time_strc_approx(graph, K, projection_seed)
                else:
                    timings[method] = time_sc_approx(graph, K, projection_seed)

            if n_nodes <= exact_max_n:
                gc.collect()
                timings["fast"] = time_fast_exact(graph)
            if n_nodes <= original_max_n:
                gc.collect()
                timings["orig"] = time_original_exact(graph)

            for method, seconds in timings.items():
                raw_rows.append(
                    {
                        "N": n_nodes,
                        "M": graph.number_of_edges(),
                        "trial": trial + 1,
                        "graph_seed": graph_seed,
                        "projection_seed": projection_seed,
                        "K": K,
                        "tolerance": TOLERANCE,
                        "method": method,
                        "seconds": seconds,
                    }
                )
            # Checkpoint after every paired trial because the paper run is long.
            _write_csv(raw_path, raw_rows, RAW_FIELDS)

            completed += 1
            ratio = timings["strc_approx"] / timings["sc_approx"]
            print(
                f"[{completed:>{len(str(total_settings))}}/{total_settings}] "
                f"N={n_nodes:>6}, trial={trial + 1}: "
                f"Approx-STRC={timings['strc_approx']:.3f}s, "
                f"Approx-SC={timings['sc_approx']:.3f}s, ratio={ratio:.3f}"
            )

    summary_rows = _summarize(raw_rows, sizes)
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    metadata_path = output_prefix.with_name(output_prefix.name + "_metadata.json")
    _write_csv(raw_path, raw_rows, RAW_FIELDS)
    _write_csv(
        summary_path,
        summary_rows,
        ["N", "method", "mean_seconds", "std_seconds", "n_trials"],
    )

    metadata = {
        "experiment": "Reviewer 2 comment 4: Approx-STRC versus Approx-SC runtime",
        "graph_model": "Barabasi-Albert (m=3)",
        "sizes": sizes,
        "trials": trials,
        "projection_dimension_K": K,
        "solver_tolerance": TOLERANCE,
        "base_seed": seed,
        "original_strc_max_N": original_max_n,
        "exact_strc_star_max_N": exact_max_n,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "matplotlib": matplotlib.__version__,
        "pyamg": pyamg.__version__,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _plot_results(summary_rows, sizes, output_prefix, show)
    print(f"Saved figure and data with prefix: {output_prefix}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired runtime benchmark for Approx-STRC and Approx-SC."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--k", type=int, default=400, dest="K")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--original-max-n", type=int, default=1600)
    parser.add_argument("--exact-max-n", type=int, default=12800)
    parser.add_argument(
        "--output-prefix",
        default=str(Path(__file__).resolve().parent.parent / "results" / "complexity_with_sc"),
        help="Output path without an extension.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from OUTPUT_PREFIX_summary.csv without rerunning timings.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test with N=100,200,400; one trial; K=50.",
    )
    args = parser.parse_args()
    if args.quick:
        args.sizes = [100, 200, 400]
        args.trials = 1
        args.K = 50
        args.original_max_n = 200
        args.exact_max_n = 400
        if args.output_prefix == "complexity_with_sc":
            args.output_prefix = "complexity_with_sc_quick"
    if (
        args.trials < 1
        or args.K < 1
        or any(size < 4 for size in args.sizes)
    ):
        parser.error("trials and K must be positive, and all sizes must be at least 4")
    args.sizes = sorted(set(args.sizes))
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.plot_only:
        prefix = Path(arguments.output_prefix).resolve()
        summary = _read_summary(prefix.with_name(prefix.name + "_summary.csv"))
        summary_sizes = sorted({row["N"] for row in summary})
        _plot_results(summary, summary_sizes, prefix, arguments.show)
        print(f"Regenerated figure from existing summary: {prefix}")
    else:
        run_runtime_experiment(
            sizes=arguments.sizes,
            trials=arguments.trials,
            K=arguments.K,
            seed=arguments.seed,
            original_max_n=arguments.original_max_n,
            exact_max_n=arguments.exact_max_n,
            output_prefix=arguments.output_prefix,
            show=arguments.show,
        )
