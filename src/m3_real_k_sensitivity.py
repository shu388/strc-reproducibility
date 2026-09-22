"""Reviewer 5 M3: K sensitivity on the six real-world networks.

This companion experiment deliberately does not claim an exact STRC reference
for the large real networks. It evaluates the downstream quantities used in
the manuscript: normalized GCC AUC for targeted dismantling and mean SIR
final size for spreading. K in {50, 200, 400} is obtained from nested
prefixes of one projection for each projection seed. The uncertainty unit is
the paired projection seed within a fixed real network, not an independent
graph realization.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
M3_PATH = SCRIPT_DIR / "m3_k_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("m3_k_sensitivity", M3_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load shared M3 implementation from {M3_PATH}")
m3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m3
SPEC.loader.exec_module(m3)

SOLVER_PATH = SCRIPT_DIR / "laplacian_solver.py"
SOLVER_SPEC = importlib.util.spec_from_file_location("m3_laplacian_solver", SOLVER_PATH)
if SOLVER_SPEC is None or SOLVER_SPEC.loader is None:
    raise ImportError(f"Could not load Laplacian solver from {SOLVER_PATH}")
solver_module = importlib.util.module_from_spec(SOLVER_SPEC)
SOLVER_SPEC.loader.exec_module(solver_module)
solve_laplacian_sdd = solver_module.solve_laplacian_sdd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DATASETS = (
    "CA-HepPh.txt",
    "Email-Enron.txt",
    "cit-HepTh.txt",
    "bio-dmela.txt",
    "Wiki-Vote.txt",
    "tech-as-caida.txt",
)
K_VALUES = (50, 200, 400)
BETA_MULTIPLIERS = (0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 2.5)


def load_lcc(path: Path) -> nx.Graph:
    """Read an edge list, discard self-loops, and return an integer LCC."""
    graph = nx.Graph()
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            graph.add_edge(fields[0], fields[1])
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if not graph:
        raise ValueError(f"No edges were read from {path}")
    component = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(component).copy()
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def projected_embedding_amg(graph: nx.Graph, K: int, seed: int):
    """Compute Z with the manuscript's reusable AMG Laplacian solver."""
    nodes, _ = m3._ordered_graph(graph)
    incidence = nx.incidence_matrix(
        graph, nodelist=nodes, oriented=True, dtype=float
    ).T.tocsr()
    laplacian = nx.laplacian_matrix(graph, nodelist=nodes).tocsr().astype(float)
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(K, graph.number_of_edges()), dtype=np.int8)
    projection = (2.0 * signs.astype(float) - 1.0) / math.sqrt(K)
    rhs = incidence.T @ projection.T
    try:
        embedding, stats = solve_laplacian_sdd(
            laplacian, rhs, tolerance=1e-5, max_cycles=100, cycle="V", strict=True
        )
    except RuntimeError:
        embedding, stats = solve_laplacian_sdd(
            laplacian, rhs, tolerance=1e-4, max_cycles=300, cycle="W", strict=False
        )
    return nodes, np.asarray(embedding), stats


def _directed_edge_index(graph: nx.Graph) -> dict:
    directed_edges = []
    for u, v in graph.edges():
        directed_edges.extend(((u, v), (v, u)))
    return {edge: index for index, edge in enumerate(directed_edges)}


def paired_sir(
    graph: nx.Graph,
    scores: dict,
    seed_fraction: float,
    beta_multipliers: tuple[float, ...],
    random_numbers: np.ndarray,
    edge_index: dict,
) -> list[dict]:
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    beta_c = float(degrees.mean() / np.mean(degrees**2))
    seed_count = max(1, int(len(graph) * seed_fraction))
    seeds = set(m3._ranked_nodes(scores)[:seed_count])
    output = []
    for multiplier in beta_multipliers:
        beta = float(np.clip(beta_c * multiplier, np.finfo(float).eps, 1.0))
        trial_values = []
        for trial in range(random_numbers.shape[0]):
            infected, recovered = set(seeds), set()
            while infected:
                new_infected = set()
                for source in infected:
                    for target in graph.neighbors(source):
                        if target in recovered or target in infected:
                            continue
                        if random_numbers[trial, edge_index[(source, target)]] < beta:
                            new_infected.add(target)
                recovered.update(infected)
                infected = new_infected
            trial_values.append(len(recovered) / len(graph))
        values = np.asarray(trial_values, dtype=float)
        output.append(
            {
                "beta_multiplier": multiplier,
                "beta_c": beta_c,
                "beta": beta,
                "sir_final_size": float(values.mean()),
                "sir_trial_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return output


def ci_half_width(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    from scipy.stats import t

    return float(t.ppf(0.975, values.size - 1) * np.std(values, ddof=1) / math.sqrt(values.size))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], group_fields: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups = sorted({tuple(row[field] for field in group_fields) for row in rows})
    summary = []
    for group in groups:
        selected = [row for row in rows if tuple(row[field] for field in group_fields) == group]
        output = dict(zip(group_fields, group))
        output["projection_seeds"] = len({row["projection_seed"] for row in selected})
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_95ci"] = ci_half_width(values)
            output[f"{metric}_ci_lower"] = output[f"{metric}_mean"] - output[f"{metric}_95ci"]
            output[f"{metric}_ci_upper"] = output[f"{metric}_mean"] + output[f"{metric}_95ci"]
        summary.append(output)
    return summary


def paired_differences(rows: list[dict]) -> list[dict]:
    output = []
    for network in sorted({row["network"] for row in rows}):
        selected = [row for row in rows if row["network"] == network]
        for metric in ("dismantling_auc", "spreading_final_size"):
            differences = []
            for seed in sorted({row["projection_seed"] for row in selected}):
                lookup = {int(row["K"]): float(row[metric]) for row in selected if row["projection_seed"] == seed}
                if 200 in lookup and 400 in lookup:
                    differences.append(lookup[400] - lookup[200])
            values = np.asarray(differences, dtype=float)
            half_width = ci_half_width(values)
            output.append(
                {
                    "network": network,
                    "metric": metric,
                    "comparison": "K=400 minus K=200",
                    "paired_projection_seeds": values.size,
                    "mean_difference": float(values.mean()),
                    "difference_95ci": half_width,
                    "difference_ci_lower": float(values.mean() - half_width),
                    "difference_ci_upper": float(values.mean() + half_width),
                }
            )
    return output


def plot_results(summary: list[dict], output: Path) -> None:
    if plt is None:
        return
    colors = {50: "#0072B2", 200: "#009E73", 400: "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    networks = sorted({row["network"] for row in summary})
    positions = np.arange(len(networks))
    width = 0.24
    for offset, K in enumerate(K_VALUES):
        selected = {row["network"]: row for row in summary if int(row["K"]) == K}
        for axis, metric, label in (
            (axes[0], "dismantling_auc", "Dismantling AUC"),
            (axes[1], "spreading_final_size", "Mean SIR final size"),
        ):
            means = [selected[network][f"{metric}_mean"] for network in networks]
            errors = [selected[network][f"{metric}_95ci"] for network in networks]
            axis.bar(positions + (offset - 1) * width, means, width, yerr=errors, capsize=3, label=f"K={K}", color=colors[K])
            axis.set_xticks(positions, networks, rotation=30, ha="right")
            axis.set_ylabel(label)
            axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Real-network dismantling")
    axes[1].set_title("Real-network spreading")
    axes[1].legend(frameon=False)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(args) -> None:
    if args.projection_seeds < 1 or args.sir_trials < 1:
        raise ValueError("Projection seeds and SIR trials must be positive.")
    selected_files = tuple(args.datasets.split(",")) if args.datasets else DATASETS
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows, sir_rows = [], []
    started = time.perf_counter()

    for network_index, filename in enumerate(selected_files):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        graph = load_lcc(path)
        network = path.stem
        print(f"[M3-real] {network}: N={len(graph)}, M={graph.number_of_edges()}", flush=True)
        edge_index = _directed_edge_index(graph)
        sir_trials = args.sir_trials if len(graph) <= 10000 else min(args.sir_trials, 5)
        for repeat in range(args.projection_seeds):
            projection_seed = 9100000 + network_index * 1000 + repeat
            projection_started = time.perf_counter()
            nodes, full_embedding, solver_stats = projected_embedding_amg(
                graph, max(K_VALUES), projection_seed
            )
            projection_seconds = time.perf_counter() - projection_started
            sir_seed = 9200000 + network_index * 1000 + repeat
            random_numbers = np.random.default_rng(sir_seed).random(
                (sir_trials, 2 * graph.number_of_edges())
            )
            for K in K_VALUES:
                embedding = full_embedding[:, :K] * math.sqrt(max(K_VALUES) / K)
                scores = m3.strc_from_embedding(graph, nodes, embedding)
                common = {
                    "network": network,
                    "N": len(graph),
                    "M": graph.number_of_edges(),
                    "K": K,
                    "projection_seed": projection_seed,
                    "solver_max_cycles": solver_stats.max_cycles,
                    "solver_max_relative_residual": solver_stats.max_relative_residual,
                    "projection_seconds": projection_seconds,
                }
                dismantling = m3.dismantling_auc(graph, scores, args.attack_fraction)
                sir_results = paired_sir(
                    graph, scores, args.seed_fraction, BETA_MULTIPLIERS,
                    random_numbers, edge_index
                )
                for item in sir_results:
                    sir_rows.append({**common, "sir_seed": sir_seed, "sir_trials": sir_trials, **item})
                task_rows.append(
                    {
                        **common,
                        "attack_fraction": args.attack_fraction,
                        "seed_fraction": args.seed_fraction,
                        "sir_trials_per_beta": sir_trials,
                        "dismantling_auc": dismantling,
                        "spreading_final_size": float(np.mean([item["sir_final_size"] for item in sir_results])),
                    }
                )
            del full_embedding
            print(f"[M3-real] {network} seed {repeat + 1}/{args.projection_seeds} complete", flush=True)

        write_csv(output_dir / "M3_real_task_sensitivity.csv", task_rows)
        write_csv(output_dir / "M3_real_sir_sensitivity.csv", sir_rows)

    task_summary = summarize(task_rows, ("network", "K"), ("dismantling_auc", "spreading_final_size"))
    sir_summary = summarize(sir_rows, ("network", "K", "beta_multiplier"), ("sir_final_size",))
    paired = paired_differences(task_rows)
    write_csv(output_dir / "M3_real_task_sensitivity_summary.csv", task_summary)
    write_csv(output_dir / "M3_real_sir_sensitivity_summary.csv", sir_summary)
    write_csv(output_dir / "M3_real_K200_K400_paired_differences.csv", paired)
    metadata = {
        "protocol": {
            "networks": list(selected_files),
            "K_values": list(K_VALUES),
            "projection_seeds_per_network": args.projection_seeds,
            "uncertainty_unit": "paired projection seed within one fixed real network",
            "confidence_interval": "two-sided Student-t 95%",
            "solver": "reusable smoothed-aggregation AMG",
            "attack_fraction": args.attack_fraction,
            "seed_fraction": args.seed_fraction,
            "beta_multipliers": list(BETA_MULTIPLIERS),
            "paired_SIR_random_numbers_across_K": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "task_summary": task_summary,
        "sir_summary": sir_summary,
        "paired_K200_K400": paired,
    }
    (output_dir / "M3_real_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(task_summary, output_dir / "M3_real_K_sensitivity")
    print(f"[M3-real] Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR.parent / "data"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR.parent / "results" / "m3_real_k_sensitivity"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--projection-seeds", type=int, default=5)
    parser.add_argument("--sir-trials", type=int, default=20)
    parser.add_argument("--attack-fraction", type=float, default=0.1)
    parser.add_argument("--seed-fraction", type=float, default=0.05)
    run(parser.parse_args())
