"""Reviewer 5 M3: projection-dimension and downstream-task sensitivity.

The default experiment uses 50 independent BA, ER, and WS graphs with
N=200 and exactly M=400 edges. For each graph, K in {50, 200, 400} is
obtained from nested prefixes of one Rademacher projection. The graph is the
independent statistical unit for all Student-t confidence intervals.

The script reports rank agreement with exact STRC, exact local H_i spectral
diagnostics, network-dismantling AUC, and SIR final outbreak size under the
same seven normalized infection rates used in the manuscript. SIR random
numbers are paired across K within every graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla
from scipy.stats import kendalltau, spearmanr, t

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_K_VALUES = (50, 200, 400)
DEFAULT_BETA_MULTIPLIERS = (0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 2.5)
NETWORK_ORDER = ("BA", "ER", "WS")


def _parse_k_values(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("K values must be positive integers.")
    return values


def _ci_half_width(values: np.ndarray, confidence: float = 0.95) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    critical = float(t.ppf((1.0 + confidence) / 2.0, values.size - 1))
    return critical * standard_error


def _largest_connected(graph: nx.Graph) -> nx.Graph:
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if not graph:
        return graph
    if not nx.is_connected(graph):
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _add_random_missing_edges(graph: nx.Graph, target_edges: int, rng: np.random.Generator) -> None:
    nodes = np.asarray(list(graph.nodes()), dtype=int)
    while graph.number_of_edges() < target_edges:
        u, v = (int(item) for item in rng.choice(nodes, size=2, replace=False))
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)


def _make_ba(nodes: int, edges: int, seed: int) -> nx.Graph:
    attachment = 1
    for candidate in range(1, nodes):
        if candidate * (nodes - candidate) <= edges:
            attachment = candidate
        else:
            break
    graph = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    _add_random_missing_edges(graph, edges, np.random.default_rng(seed + 1))
    return _largest_connected(graph)


def _make_er(nodes: int, edges: int, seed: int) -> nx.Graph:
    for attempt in range(10000):
        graph = nx.gnm_random_graph(nodes, edges, seed=seed + attempt)
        if nx.is_connected(graph):
            return _largest_connected(graph)
    raise RuntimeError("Could not generate a connected ER graph.")


def _make_ws(nodes: int, edges: int, seed: int) -> nx.Graph:
    if 2 * edges % nodes != 0:
        raise ValueError("WS requires 2*M/N to be an integer.")
    degree = 2 * edges // nodes
    if degree <= 0 or degree >= nodes or degree % 2:
        raise ValueError("WS requires an even ring degree in [2, N-1].")
    graph = nx.connected_watts_strogatz_graph(
        nodes, degree, 0.05, tries=1000, seed=seed
    )
    if graph.number_of_edges() != edges:
        raise RuntimeError("WS generator did not preserve the requested edge count.")
    return _largest_connected(graph)


def make_graphs(
    nodes: int = 200,
    edges: int = 400,
    realization: int = 0,
) -> list[tuple[str, nx.Graph, str, int]]:
    """Return one reproducible realization of each controlled graph model."""
    if nodes < 4:
        raise ValueError("At least four nodes are required.")
    if edges < nodes - 1 or edges > nodes * (nodes - 1) // 2:
        raise ValueError("M must permit a connected simple graph.")
    base_seed = 100000 + 1000 * realization
    specifications = (
        ("BA", _make_ba, base_seed + 101),
        ("ER", _make_er, base_seed + 202),
        ("WS", _make_ws, base_seed + 303),
    )
    graphs = []
    for name, generator, seed in specifications:
        graph = generator(nodes, edges, seed)
        if len(graph) != nodes or graph.number_of_edges() != edges or not nx.is_connected(graph):
            raise RuntimeError(f"{name} realization does not satisfy N={nodes}, M={edges}.")
        graphs.append((name, graph, "synthetic", seed))
    return graphs


def _ordered_graph(graph: nx.Graph) -> tuple[list, dict]:
    nodes = list(graph.nodes())
    return nodes, {node: index for index, node in enumerate(nodes)}


def _laplacian_pseudoinverse(graph: nx.Graph, nodes: list | None = None) -> np.ndarray:
    if nodes is None:
        nodes, _ = _ordered_graph(graph)
    laplacian = nx.laplacian_matrix(graph, nodelist=nodes).toarray().astype(float)
    return np.linalg.pinv(laplacian, hermitian=True, rcond=1e-10)


def _local_h_matrix(
    graph: nx.Graph,
    node,
    nodes: list,
    index: dict,
    laplacian_pseudoinverse: np.ndarray,
) -> np.ndarray:
    center = index[node]
    neighbours = np.asarray([index[value] for value in graph.neighbors(node)], dtype=int)
    return (
        laplacian_pseudoinverse[np.ix_(neighbours, neighbours)]
        - laplacian_pseudoinverse[neighbours, center][:, None]
        - laplacian_pseudoinverse[center, neighbours][None, :]
        + laplacian_pseudoinverse[center, center]
    )


def exact_strc_scores(graph: nx.Graph, Lp: np.ndarray | None = None) -> dict:
    """Compute exact STRC scores using the Laplacian pseudoinverse."""
    nodes, index = _ordered_graph(graph)
    if Lp is None:
        Lp = _laplacian_pseudoinverse(graph, nodes)
    scores = {}
    for node in nodes:
        degree = graph.degree(node)
        if degree <= 1:
            scores[node] = 0.0
            continue
        local_h = _local_h_matrix(graph, node, nodes, index, Lp)
        matrix = np.eye(degree) - local_h + np.ones((degree, degree)) / degree
        ratio = float(np.linalg.det(matrix) / degree)
        scores[node] = 1.0 - float(np.clip(ratio, 0.0, 1.0))
    return scores


def exact_h_diagnostics(
    graph: nx.Graph,
    laplacian_pseudoinverse: np.ndarray,
    exact_scores: dict,
) -> dict:
    """Summarize exact local H_i spectra and exact-score separation."""
    nodes, index = _ordered_graph(graph)
    effective_ranks, energy90, energy95, spectra = [], [], [], []
    for node in nodes:
        if graph.degree(node) <= 1:
            continue
        local_h = _local_h_matrix(graph, node, nodes, index, laplacian_pseudoinverse)
        eigenvalues = np.clip(np.linalg.eigvalsh(local_h)[::-1], 0.0, None)
        total = float(eigenvalues.sum())
        if total <= np.finfo(float).tiny:
            continue
        normalized = eigenvalues / total
        positive = normalized[normalized > 0]
        effective_ranks.append(float(np.exp(-np.sum(positive * np.log(positive)))))
        cumulative = np.cumsum(normalized)
        energy90.append(int(np.searchsorted(cumulative, 0.90) + 1))
        energy95.append(int(np.searchsorted(cumulative, 0.95) + 1))
        spectra.append(np.pad(normalized[:10], (0, max(0, 10 - normalized.size)))[:10])
    score_values = np.sort(np.asarray(list(exact_scores.values()), dtype=float))
    adjacent_gaps = np.diff(score_values)
    result = {
        "effective_rank_median": float(np.median(effective_ranks)),
        "effective_rank_p90": float(np.percentile(effective_ranks, 90)),
        "energy90_dim_median": float(np.median(energy90)),
        "energy95_dim_median": float(np.median(energy95)),
        "score_iqr": float(np.percentile(score_values, 75) - np.percentile(score_values, 25)),
        "median_adjacent_score_gap": float(np.median(adjacent_gaps)),
    }
    median_spectrum = np.median(np.asarray(spectra), axis=0)
    result.update(
        {f"lambda_norm_{position}": float(median_spectrum[position - 1]) for position in range(1, 11)}
    )
    return result


def build_projection_context(graph: nx.Graph) -> dict:
    """Factor the grounded Laplacian once and reuse it for all right sides."""
    nodes, _ = _ordered_graph(graph)
    incidence = nx.incidence_matrix(
        graph, nodelist=nodes, oriented=True, dtype=float
    ).T.tocsr()
    laplacian = nx.laplacian_matrix(graph, nodelist=nodes).tocsc().astype(float)
    factorization = spla.splu(laplacian[1:, 1:])
    return {
        "nodes": nodes,
        "incidence": incidence,
        "factorization": factorization,
        "edges": graph.number_of_edges(),
    }


def projected_embedding(
    graph: nx.Graph,
    K: int,
    seed: int,
    context: dict | None = None,
) -> tuple[list, np.ndarray]:
    """Solve the projected Laplacian systems for a Rademacher projection."""
    if context is None:
        context = build_projection_context(graph)
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(K, context["edges"]), dtype=np.int8)
    projection = (2.0 * signs.astype(float) - 1.0) / math.sqrt(K)
    right_hand_side = context["incidence"].T @ projection.T
    embedding = np.zeros((len(context["nodes"]), K), dtype=float)
    embedding[1:, :] = context["factorization"].solve(
        np.asarray(right_hand_side[1:, :], dtype=float)
    )
    embedding -= embedding.mean(axis=0, keepdims=True)
    return context["nodes"], embedding


def strc_from_embedding(graph: nx.Graph, nodes: list, embedding: np.ndarray) -> dict:
    index = {node: position for position, node in enumerate(nodes)}
    scores = {}
    for node in nodes:
        degree = graph.degree(node)
        if degree <= 1:
            scores[node] = 0.0
            continue
        center = index[node]
        neighbours = [index[value] for value in graph.neighbors(node)]
        differences = embedding[neighbours, :] - embedding[center, :]
        local_h = differences @ differences.T
        matrix = np.eye(degree) - local_h + np.ones((degree, degree)) / degree
        ratio = float(np.linalg.det(matrix) / degree)
        scores[node] = 1.0 - float(np.clip(ratio, 0.0, 1.0))
    return scores


def _ranked_nodes(scores: dict) -> list:
    return [
        node
        for node, _ in sorted(
            scores.items(), key=lambda item: (-float(item[1]), str(item[0]))
        )
    ]


def rank_metrics(
    reference: dict,
    candidate: dict,
    top_fraction: float = 0.1,
) -> tuple[float, float, float]:
    common = [node for node in reference if node in candidate]
    exact = np.asarray([reference[node] for node in common], dtype=float)
    approximate = np.asarray([candidate[node] for node in common], dtype=float)
    rho = float(spearmanr(exact, approximate).statistic)
    tau = float(kendalltau(exact, approximate).statistic)
    if not np.isfinite(rho):
        rho = 1.0 if np.allclose(exact, approximate) else 0.0
    if not np.isfinite(tau):
        tau = 1.0 if np.allclose(exact, approximate) else 0.0
    count = max(1, int(len(common) * top_fraction))
    exact_top = set(_ranked_nodes(reference)[:count])
    approximate_top = set(_ranked_nodes(candidate)[:count])
    return rho, tau, len(exact_top & approximate_top) / count


def dismantling_auc(graph: nx.Graph, scores: dict, fraction: float = 0.1) -> float:
    node_count = len(graph)
    removal_count = max(1, int(node_count * fraction))
    remaining = graph.copy()
    curve = [len(max(nx.connected_components(remaining), key=len)) / node_count]
    for node in _ranked_nodes(scores)[:removal_count]:
        remaining.remove_node(node)
        largest = max((len(component) for component in nx.connected_components(remaining)), default=0)
        curve.append(largest / node_count)
    x_values = np.linspace(0.0, fraction, len(curve))
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(curve, x_values) / fraction)


def _directed_edge_index(graph: nx.Graph) -> dict:
    directed_edges = []
    for u, v in graph.edges():
        directed_edges.extend(((u, v), (v, u)))
    return {edge: index for index, edge in enumerate(directed_edges)}


def paired_sir_results(
    graph: nx.Graph,
    scores: dict,
    seed_fraction: float,
    beta_multipliers: tuple[float, ...],
    trials: int,
    random_numbers: np.ndarray,
    edge_index: dict,
) -> list[dict]:
    """Run one-step SIR (gamma=1) using edge-level common random numbers."""
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    beta_c = float(degrees.mean() / np.mean(degrees**2)) if np.any(degrees) else 0.05
    seed_count = max(1, int(len(graph) * seed_fraction))
    initial_seeds = set(_ranked_nodes(scores)[:seed_count])
    results = []
    for multiplier in beta_multipliers:
        beta = float(np.clip(beta_c * multiplier, np.finfo(float).eps, 1.0))
        final_sizes = []
        for trial in range(trials):
            infected = set(initial_seeds)
            recovered = set()
            while infected:
                newly_infected = set()
                for source in infected:
                    for target in graph.neighbors(source):
                        if target in recovered or target in infected:
                            continue
                        if random_numbers[trial, edge_index[(source, target)]] < beta:
                            newly_infected.add(target)
                recovered.update(infected)
                infected = newly_infected
            final_sizes.append(len(recovered) / len(graph))
        values = np.asarray(final_sizes, dtype=float)
        results.append(
            {
                "beta_multiplier": multiplier,
                "beta_c": beta_c,
                "beta": beta,
                "sir_final_size": float(values.mean()),
                "sir_trial_std": float(values.std(ddof=1)) if trials > 1 else 0.0,
            }
        )
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    rows: list[dict],
    group_fields: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict]:
    keys = sorted({tuple(row[field] for field in group_fields) for row in rows})
    summary = []
    for key in keys:
        selected = [
            row
            for row in rows
            if tuple(row[field] for field in group_fields) == key
        ]
        output = dict(zip(group_fields, key))
        output["graph_repetitions"] = len({row["graph_repeat"] for row in selected})
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_95ci"] = _ci_half_width(values)
            output[f"{metric}_ci_lower"] = output[f"{metric}_mean"] - output[f"{metric}_95ci"]
            output[f"{metric}_ci_upper"] = output[f"{metric}_mean"] + output[f"{metric}_95ci"]
        summary.append(output)
    return summary


def _paired_comparison(task_rows: list[dict], low_k: int = 200, high_k: int = 400) -> list[dict]:
    comparisons = []
    for network in NETWORK_ORDER:
        network_rows = [row for row in task_rows if row["network"] == network]
        for metric in ("dismantling_auc", "spreading_final_size"):
            differences = []
            repeats = sorted({int(row["graph_repeat"]) for row in network_rows})
            for repeat in repeats:
                lookup = {
                    int(row["K"]): float(row[metric])
                    for row in network_rows
                    if int(row["graph_repeat"]) == repeat
                }
                if low_k in lookup and high_k in lookup:
                    differences.append(lookup[high_k] - lookup[low_k])
            values = np.asarray(differences, dtype=float)
            if not values.size:
                continue
            half_width = _ci_half_width(values)
            comparisons.append(
                {
                    "network": network,
                    "metric": metric,
                    "comparison": f"K={high_k} minus K={low_k}",
                    "paired_graphs": values.size,
                    "mean_difference": float(values.mean()),
                    "difference_95ci": half_width,
                    "difference_ci_lower": float(values.mean()) - half_width,
                    "difference_ci_upper": float(values.mean()) + half_width,
                }
            )
    return comparisons


def _diagnostic_summary(rows: list[dict]) -> list[dict]:
    metrics = (
        "effective_rank_median",
        "effective_rank_p90",
        "energy90_dim_median",
        "energy95_dim_median",
        "score_iqr",
        "median_adjacent_score_gap",
        *(f"lambda_norm_{position}" for position in range(1, 11)),
    )
    return _summarize(rows, ("network",), metrics)


def plot_results(
    rank_summary: list[dict],
    task_summary: list[dict],
    output: Path,
) -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.25), constrained_layout=True)
    colors = {"BA": "#0072B2", "ER": "#009E73", "WS": "#D55E00"}
    panels = (
        (axes[0], rank_summary, "spearman", "Spearman $\\rho$", "Rank agreement"),
        (axes[1], task_summary, "dismantling_auc", "Dismantling AUC", "Dismantling"),
        (axes[2], task_summary, "spreading_final_size", "Mean final size", "SIR spreading"),
    )
    for axis, rows, metric, ylabel, title in panels:
        for network in NETWORK_ORDER:
            selected = sorted(
                (row for row in rows if row["network"] == network),
                key=lambda row: int(row["K"]),
            )
            axis.errorbar(
                [int(row["K"]) for row in selected],
                [float(row[f"{metric}_mean"]) for row in selected],
                yerr=[float(row[f"{metric}_95ci"]) for row in selected],
                marker="o",
                capsize=3,
                label=network,
                color=colors[network],
            )
        axis.set_xlabel("Projection dimension K")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].axhline(0.95, color="#666666", linestyle="--", linewidth=1)
    axes[2].legend(frameon=False)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _mean_ci(row: dict, metric: str, digits: int = 4) -> str:
    return (
        f"${float(row[f'{metric}_mean']):.{digits}f}"
        f"\\pm{float(row[f'{metric}_95ci']):.{digits}f}$"
    )


def write_latex_tables(
    output_dir: Path,
    rank_summary: list[dict],
    task_summary: list[dict],
    diagnostic_summary: list[dict],
) -> None:
    """Write compact tables whose values are sourced from the final CSVs."""
    rank_lookup = {
        (row["network"], int(row["K"])): row for row in rank_summary
    }
    task_lookup = {
        (row["network"], int(row["K"])): row for row in task_summary
    }
    sensitivity_lines = [
        "% Auto-generated by the M3 sensitivity script; do not edit values manually.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Sensitivity of approximate STRC to projection dimension $K$. "
        "Entries are graph-ensemble means $\\pm$ Student-$t$ 95\\% confidence-interval half-widths.}",
        "\\label{tab:m3-k-sensitivity}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Network & $K$ & Spearman $\\rho$ & Dismantling AUC & Mean SIR final size \\\\",
        "\\midrule",
    ]
    for network in NETWORK_ORDER:
        for row_index, K in enumerate(DEFAULT_K_VALUES):
            rank = rank_lookup[(network, K)]
            task = task_lookup[(network, K)]
            network_label = network if row_index == 0 else ""
            sensitivity_lines.append(
                f"{network_label} & {K} & {_mean_ci(rank, 'spearman')} & "
                f"{_mean_ci(task, 'dismantling_auc')} & "
                f"{_mean_ci(task, 'spreading_final_size')} \\\\"
            )
        if network != NETWORK_ORDER[-1]:
            sensitivity_lines.append("\\addlinespace")
    sensitivity_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (output_dir / "M3_appendix_K_sensitivity_table.tex").write_text(
        "\n".join(sensitivity_lines), encoding="utf-8"
    )

    diagnostics_lines = [
        "% Auto-generated by the M3 sensitivity script; do not edit values manually.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Exact local-spectrum and STRC score-separation diagnostics. "
        "Entries are graph-ensemble means $\\pm$ Student-$t$ 95\\% confidence-interval half-widths.}",
        "\\label{tab:m3-hi-diagnostics}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Network & Median effective rank & 95\\% energy dimension & STRC score IQR & Median adjacent gap \\\\",
        "\\midrule",
    ]
    diagnostic_lookup = {row["network"]: row for row in diagnostic_summary}
    for network in NETWORK_ORDER:
        row = diagnostic_lookup[network]
        diagnostics_lines.append(
            f"{network} & {_mean_ci(row, 'effective_rank_median', 3)} & "
            f"{_mean_ci(row, 'energy95_dim_median', 2)} & "
            f"{_mean_ci(row, 'score_iqr', 4)} & "
            f"{_mean_ci(row, 'median_adjacent_score_gap', 6)} \\\\"
        )
    diagnostics_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (output_dir / "M3_Hi_diagnostics_table.tex").write_text(
        "\n".join(diagnostics_lines), encoding="utf-8"
    )


def _write_outputs(
    output_dir: Path,
    rank_rows: list[dict],
    task_rows: list[dict],
    sir_rows: list[dict],
    diagnostic_rows: list[dict],
) -> dict:
    rank_summary = _summarize(
        rank_rows, ("network", "K"), ("spearman", "kendall", "top10_overlap")
    )
    task_summary = _summarize(
        task_rows,
        ("network", "K"),
        ("dismantling_auc", "spreading_final_size"),
    )
    sir_summary = _summarize(
        sir_rows, ("network", "K", "beta_multiplier"), ("sir_final_size",)
    )
    diagnostic_summary = _diagnostic_summary(diagnostic_rows)
    paired = _paired_comparison(task_rows)
    outputs = {
        "M3_rank_sensitivity.csv": rank_rows,
        "M3_rank_sensitivity_summary.csv": rank_summary,
        "M3_task_sensitivity.csv": task_rows,
        "M3_task_sensitivity_summary.csv": task_summary,
        "M3_sir_sensitivity.csv": sir_rows,
        "M3_sir_sensitivity_summary.csv": sir_summary,
        "M3_Hi_diagnostics.csv": diagnostic_rows,
        "M3_Hi_diagnostics_summary.csv": diagnostic_summary,
        "M3_K200_K400_paired_differences.csv": paired,
    }
    for filename, rows in outputs.items():
        write_csv(output_dir / filename, rows)
    if all(
        (network, K) in {(row["network"], int(row["K"])) for row in rank_summary}
        for network in NETWORK_ORDER
        for K in DEFAULT_K_VALUES
    ):
        write_latex_tables(
            output_dir, rank_summary, task_summary, diagnostic_summary
        )
    return {
        "rank_summary": rank_summary,
        "task_summary": task_summary,
        "sir_summary": sir_summary,
        "diagnostic_summary": diagnostic_summary,
        "paired_K200_K400": paired,
    }


def run(args) -> None:
    if args.graph_repeats < 1 or args.sir_trials < 1:
        raise ValueError("Graph repetitions and SIR trials must be positive.")
    if args.nodes > args.exact_max_n:
        raise ValueError("Exact STRC is required; increase --exact-max-n or reduce --nodes.")
    k_values = _parse_k_values(args.k_values)
    beta_multipliers = tuple(float(value) for value in args.beta_multipliers.split(","))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_rows, task_rows, sir_rows, diagnostic_rows = [], [], [], []
    experiment_started = time.perf_counter()

    for graph_repeat in range(args.graph_repeats):
        for network_index, (network, graph, graph_kind, graph_seed) in enumerate(
            make_graphs(args.nodes, args.edges, graph_repeat)
        ):
            print(
                f"[M3] graph {graph_repeat + 1}/{args.graph_repeats} {network}: "
                f"N={len(graph)}, M={graph.number_of_edges()}",
                flush=True,
            )
            nodes, _ = _ordered_graph(graph)
            laplacian_pseudoinverse = _laplacian_pseudoinverse(graph, nodes)
            exact_scores = exact_strc_scores(graph, Lp=laplacian_pseudoinverse)
            diagnostic_rows.append(
                {
                    "network": network,
                    "graph_repeat": graph_repeat,
                    "graph_seed": graph_seed,
                    "N": len(graph),
                    "M": graph.number_of_edges(),
                    **exact_h_diagnostics(graph, laplacian_pseudoinverse, exact_scores),
                }
            )

            projection_seed = 5000000 + 100 * graph_repeat + network_index
            context = build_projection_context(graph)
            projection_started = time.perf_counter()
            projected_nodes, full_embedding = projected_embedding(
                graph, max(k_values), projection_seed, context=context
            )
            projection_seconds = time.perf_counter() - projection_started

            edge_index = _directed_edge_index(graph)
            sir_seed = 7000000 + 100 * graph_repeat + network_index
            sir_random_numbers = np.random.default_rng(sir_seed).random(
                (args.sir_trials, 2 * graph.number_of_edges())
            )
            for K in k_values:
                score_started = time.perf_counter()
                embedding = full_embedding[:, :K] * math.sqrt(max(k_values) / K)
                approximate_scores = strc_from_embedding(graph, projected_nodes, embedding)
                score_seconds = time.perf_counter() - score_started
                rho, tau, overlap = rank_metrics(exact_scores, approximate_scores)
                common = {
                    "network": network,
                    "graph_repeat": graph_repeat,
                    "graph_seed": graph_seed,
                    "N": len(graph),
                    "M": graph.number_of_edges(),
                    "K": K,
                    "projection_seed": projection_seed,
                }
                rank_rows.append(
                    {
                        **common,
                        "spearman": rho,
                        "kendall": tau,
                        "top10_overlap": overlap,
                        "K_max_projection_seconds": projection_seconds,
                        "score_seconds": score_seconds,
                        "reference": "exact",
                    }
                )
                dismantling = dismantling_auc(graph, approximate_scores, args.attack_fraction)
                sir_results = paired_sir_results(
                    graph,
                    approximate_scores,
                    args.seed_fraction,
                    beta_multipliers,
                    args.sir_trials,
                    sir_random_numbers,
                    edge_index,
                )
                for sir_result in sir_results:
                    sir_rows.append(
                        {
                            **common,
                            "sir_seed": sir_seed,
                            "sir_trials": args.sir_trials,
                            **sir_result,
                        }
                    )
                task_rows.append(
                    {
                        **common,
                        "attack_fraction": args.attack_fraction,
                        "seed_fraction": args.seed_fraction,
                        "sir_trials_per_beta": args.sir_trials,
                        "dismantling_auc": dismantling,
                        "spreading_final_size": float(
                            np.mean([row["sir_final_size"] for row in sir_results])
                        ),
                    }
                )

        # Preserve recoverable tables if a long run is interrupted later.
        _write_outputs(output_dir, rank_rows, task_rows, sir_rows, diagnostic_rows)

    summaries = _write_outputs(output_dir, rank_rows, task_rows, sir_rows, diagnostic_rows)
    metadata = {
        "protocol": {
            "networks": list(NETWORK_ORDER),
            "network_parameters": {
                "BA": "m=2, followed by uniformly sampled missing edges until M=400",
                "ER": "connected G(N,M)",
                "WS": "k=4, rewiring probability p=0.05",
            },
            "N": args.nodes,
            "M": args.edges,
            "graph_repetitions": args.graph_repeats,
            "K_values": list(k_values),
            "nested_projection": True,
            "rank_reference": "exact STRC",
            "projection_system_solver": "grounded sparse LU",
            "solver_rationale": "isolate projection-dimension error from iterative-solver error",
            "confidence_interval_unit": "independent graph realization",
            "confidence_interval": "two-sided Student-t 95%",
            "attack_fraction": args.attack_fraction,
            "seed_fraction": args.seed_fraction,
            "beta_multipliers": list(beta_multipliers),
            "sir_trials_per_beta": args.sir_trials,
            "sir_random_numbers_paired_across_K": True,
            "graph_seed_scheme": "100000 + 1000*graph_repeat + {101,202,303}",
            "projection_seed_scheme": "5000000 + 100*graph_repeat + network_index",
            "sir_seed_scheme": "7000000 + 100*graph_repeat + network_index",
        },
        "elapsed_seconds": time.perf_counter() - experiment_started,
        **summaries,
    }
    (output_dir / "M3_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(
        summaries["rank_summary"],
        summaries["task_summary"],
        output_dir / "M3_K_sensitivity",
    )
    print(f"[M3] Wrote complete results to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR.parent / "results" / "m3_k_sensitivity"))
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--edges", type=int, default=400)
    parser.add_argument("--exact-max-n", type=int, default=300)
    parser.add_argument("--graph-repeats", type=int, default=50)
    parser.add_argument("--k-values", default=",".join(map(str, DEFAULT_K_VALUES)))
    parser.add_argument(
        "--beta-multipliers",
        default=",".join(map(str, DEFAULT_BETA_MULTIPLIERS)),
    )
    parser.add_argument("--sir-trials", type=int, default=20)
    parser.add_argument("--attack-fraction", type=float, default=0.1)
    parser.add_argument("--seed-fraction", type=float, default=0.05)
    run(parser.parse_args())
