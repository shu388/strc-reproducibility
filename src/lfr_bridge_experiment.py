"""Controlled LFR experiments for the topology-dependent STRC hypothesis.

The experiment has three independent sweeps:
1. native LFR mixing parameter ``mu``;
2. inter-community edge ratio under a fixed LFR community assignment;
3. bridge-node fraction under fixed communities and inter-community density.

For every graph we report the targeted-dismantling AUC, the absolute AUC
advantage, and the relative gain of STRC over the best competing baseline.
The generated graphs and measurements are written to CSV so the analysis is
reproducible without re-running plots.
"""

from __future__ import annotations

import csv
import importlib.util
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from advanced_baselines import (
    closeness_energy_scores,
    collective_influence_scores,
    radiation_theory_scores,
)


SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_CSV = RESULTS_DIR / "lfr_bridge_results.csv"
RESULT_FIG = RESULTS_DIR / "LFR_STRC_bridge_sensitivity.png"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _load_dismantling_helpers():
    """Reuse the tested STRC and attack implementations from the main script."""
    path = SCRIPT_DIR / "dismantling_experiment.py"
    spec = importlib.util.spec_from_file_location("dismantling_main", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN = _load_dismantling_helpers()


METHODS = ["STRC", "DC", "BC", "CC", "PR", "EC", "K-core", "NC", "CE", "RT", "CI"]
MU_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40]
EDGE_RATIO_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40]
BRIDGE_FRACTION_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40]
EDGE_SWEEP_BRIDGE_FRACTION = 0.20
BRIDGE_SWEEP_EDGE_RATIO = 0.20
ATTACK_FRACTION = 0.20
DEFAULT_N = 2000
REPEATS = 5


def generate_lfr(mu: float, seed: int, n: int = DEFAULT_N) -> nx.Graph:
    """Generate a connected LFR graph and preserve its community labels."""
    for attempt in range(100):
        try:
            graph = nx.LFR_benchmark_graph(
                n,
                tau1=3.0,
                tau2=1.5,
                mu=mu,
                # NetworkX treats min_degree and average_degree as mutually
                # exclusive alternatives.  Use average_degree alone here;
                # adding min_degree=5 reproduces the reported exception.
                average_degree=12,
                min_community=40,
                max_community=250,
                seed=seed + attempt,
            )
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            if not nx.is_connected(graph):
                graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
            if graph.number_of_nodes() < 0.85 * n:
                continue
            _attach_community_ids(graph)
            return graph
        except nx.ExceededMaxIterations:
            continue
        except ValueError:
            continue
    raise RuntimeError(f"Unable to generate an LFR graph for mu={mu}")


def _attach_community_ids(graph: nx.Graph) -> dict:
    """Convert LFR's set-valued community attribute to integer labels."""
    groups = {}
    for node, attrs in graph.nodes(data=True):
        community = attrs.get("community")
        if isinstance(community, (set, frozenset, list, tuple)):
            community = min(community) if community else node
        groups[node] = community
    labels = {value: index for index, value in enumerate(sorted(set(groups.values()), key=str))}
    groups = {node: labels[value] for node, value in groups.items()}
    nx.set_node_attributes(graph, groups, "community_id")
    return groups


def _sample_pairs(nodes: list, count: int, existing: set, rng: random.Random) -> None:
    """Add random undirected pairs to an existing edge-key set."""
    if count <= 0 or len(nodes) < 2:
        return
    attempts = 0
    max_attempts = max(1000, count * 100)
    while count > 0 and attempts < max_attempts:
        u, v = rng.sample(nodes, 2)
        key = (u, v) if u < v else (v, u)
        if key not in existing:
            existing.add(key)
            count -= 1
        attempts += 1


def controlled_community_graph(
    base: nx.Graph,
    inter_edge_ratio: float,
    bridge_fraction: float | None,
    seed: int,
) -> nx.Graph:
    """Generate a graph with a fixed LFR community assignment.

    ``bridge_fraction`` restricts cross-community edges to a selected fraction
    of nodes. The result is a controlled graph rather than a native LFR draw;
    this isolates the structural variable being swept.
    """
    rng = random.Random(seed)
    groups = nx.get_node_attributes(base, "community_id")
    by_group = {}
    for node, group in groups.items():
        by_group.setdefault(group, []).append(node)
    nodes = list(base.nodes())
    group_ids = list(by_group)
    total_edges = base.number_of_edges()

    # A connected graph with k communities needs at least k-1 cross-community
    # edges. This is the only unavoidable deviation from a very small target.
    target_cross = max(len(group_ids) - 1, int(round(total_edges * inter_edge_ratio)))
    edge_keys = set()

    if bridge_fraction is None:
        bridge_nodes = nodes.copy()
    else:
        bridge_count = max(len(group_ids), int(round(len(nodes) * bridge_fraction)))
        bridge_nodes = [rng.choice(members) for members in by_group.values()]
        selected = set(bridge_nodes)
        remaining = [node for node in nodes if node not in selected]
        bridge_nodes.extend(rng.sample(remaining, min(bridge_count - len(bridge_nodes), len(remaining))))

    bridge_by_group = {}
    for node in bridge_nodes:
        bridge_by_group.setdefault(groups[node], []).append(node)
    group_ids = [group for group in group_ids if group in bridge_by_group]

    def add_edge(u, v) -> bool:
        if u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in edge_keys:
            return False
        edge_keys.add(key)
        return True

    def cross_edge_count() -> int:
        return sum(groups[u] != groups[v] for u, v in edge_keys)

    # Build connected spanning trees inside communities first.
    for members in by_group.values():
        ordered = members.copy()
        rng.shuffle(ordered)
        for u, v in zip(ordered, ordered[1:]):
            add_edge(u, v)

    # Connect communities through one representative per community. This
    # guarantees connectivity without a post-hoc edge injection step.
    representatives = [bridge_by_group[group][0] for group in group_ids]
    for u, v in zip(representatives, representatives[1:]):
        add_edge(u, v)

    # Make every selected bridge node incident to a cross-community edge when
    # the requested edge budget permits it.
    for index, node in enumerate(bridge_nodes):
        if cross_edge_count() >= target_cross:
            break
        own_group = groups[node]
        other_groups = [group for group in group_ids if group != own_group]
        if not other_groups:
            break
        other_group = other_groups[index % len(other_groups)]
        add_edge(node, rng.choice(bridge_by_group[other_group]))

    # Fill the remaining cross-community budget using only bridge candidates.
    attempts = 0
    max_attempts = max(5000, target_cross * 200)
    while cross_edge_count() < target_cross and attempts < max_attempts:
        group_a, group_b = rng.sample(group_ids, 2)
        add_edge(rng.choice(bridge_by_group[group_a]), rng.choice(bridge_by_group[group_b]))
        attempts += 1
    if cross_edge_count() < target_cross:
        raise RuntimeError("Unable to realize the requested cross-community edge budget")

    # Fill the remaining edge budget with within-community edges. The spanning
    # trees above ensure that this never disconnects the graph.
    intra_edges = [(u, v) for u, v in base.edges() if groups.get(u) == groups.get(v)]
    rng.shuffle(intra_edges)
    for u, v in intra_edges:
        if len(edge_keys) >= total_edges:
            break
        add_edge(u, v)

    attempts = 0
    max_attempts = max(10000, total_edges * 100)
    while len(edge_keys) < total_edges and attempts < max_attempts:
        members = by_group[rng.choice(group_ids)]
        if len(members) >= 2:
            add_edge(*rng.sample(members, 2))
        attempts += 1
    if len(edge_keys) < total_edges:
        raise RuntimeError("Unable to realize the requested total edge budget")

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_keys)
    nx.set_node_attributes(graph, groups, "community_id")
    return graph


def structural_descriptors(graph: nx.Graph) -> tuple[float, float]:
    groups = nx.get_node_attributes(graph, "community_id")
    cross = sum(groups.get(u) != groups.get(v) for u, v in graph.edges())
    inter_ratio = cross / max(1, graph.number_of_edges())
    bridge_nodes = {
        node for node in graph if any(groups.get(node) != groups.get(neighbor) for neighbor in graph[node])
    }
    bridge_fraction = len(bridge_nodes) / max(1, graph.number_of_nodes())
    return inter_ratio, bridge_fraction


def score_methods(graph: nx.Graph) -> dict:
    n = graph.number_of_nodes()
    return {
        "STRC": MAIN.million_node_stc_approximation(graph, K=256),
        "DC": nx.degree_centrality(graph),
        "BC": nx.betweenness_centrality(graph, k=min(n, 200), seed=20260815),
        "CC": MAIN.approx_closeness_centrality(graph, sample_size=min(n, 200)),
        "PR": nx.pagerank(graph),
        "EC": MAIN.approx_eigenvector_centrality(graph),
        "K-core": nx.core_number(graph),
        "NC": MAIN.approx_natural_connectivity_centrality(graph),
        "CE": closeness_energy_scores(graph, landmark_count=min(n, 256)),
        "RT": radiation_theory_scores(graph, exact_node_limit=2000, landmark_count=128),
        "CI": collective_influence_scores(graph, radius=2),
    }


def evaluate_graph(graph: nx.Graph, top_fraction: float = ATTACK_FRACTION) -> dict:
    scores = score_methods(graph)
    auc = {}
    for method, node_scores in scores.items():
        curve = MAIN.simulate_attack(graph, node_scores, top_fraction=top_fraction)
        x = np.linspace(0.0, top_fraction, len(curve))
        try:
            auc[method] = float(np.trapezoid(curve, x) / top_fraction)
        except AttributeError:
            auc[method] = float(np.trapz(curve, x) / top_fraction)
    best_baseline = min(auc[method] for method in METHODS if method != "STRC")
    delta_auc = best_baseline - auc["STRC"]
    gain = 100.0 * delta_auc / best_baseline if best_baseline > 1e-12 else float("nan")
    inter_ratio, bridge_fraction = structural_descriptors(graph)
    return {
        "auc": auc,
        "best_baseline_auc": best_baseline,
        "strc_delta_auc": delta_auc,
        "strc_gain_percent": gain,
        "inter_edge_ratio": inter_ratio,
        "bridge_node_fraction": bridge_fraction,
    }


def run_sweep(name: str, values: list[float], repeats: int = REPEATS, n: int = DEFAULT_N) -> list[dict]:
    rows = []
    for value in values:
        for repeat in range(repeats):
            seed = 20260815 + repeat * 1000 + int(value * 1000)
            if name == "mu":
                graph = generate_lfr(value, seed, n=n)
            elif name == "inter_edge_ratio":
                base = generate_lfr(0.15, seed, n=n)
                graph = controlled_community_graph(base, value, EDGE_SWEEP_BRIDGE_FRACTION, seed)
            elif name == "bridge_node_fraction":
                base = generate_lfr(0.15, seed, n=n)
                graph = controlled_community_graph(base, BRIDGE_SWEEP_EDGE_RATIO, value, seed)
            else:
                raise ValueError(name)

            # STRC uses randomized probing vectors; seed all NumPy-based
            # scoring for reproducible repeat-level comparisons.
            np.random.seed(seed)
            result = evaluate_graph(graph, top_fraction=ATTACK_FRACTION)
            if name == "mu":
                measured_parameter = value
            elif name == "inter_edge_ratio":
                measured_parameter = result["inter_edge_ratio"]
            else:
                measured_parameter = result["bridge_node_fraction"]
            for method, value_auc in result["auc"].items():
                rows.append(
                    {
                        "experiment": name,
                        "target_parameter": value,
                        "parameter": measured_parameter,
                        "repeat": repeat,
                        "N": graph.number_of_nodes(),
                        "M": graph.number_of_edges(),
                        "inter_edge_ratio": result["inter_edge_ratio"],
                        "bridge_node_fraction": result["bridge_node_fraction"],
                        "method": method,
                        "auc": value_auc,
                        "best_baseline_auc": result["best_baseline_auc"],
                        "strc_delta_auc": result["strc_delta_auc"],
                        "strc_gain_percent": result["strc_gain_percent"],
                    }
                )
    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_gain(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, constrained_layout=True)
    labels = {
        "mu": ("LFR mixing parameter $\\mu$", "Input $\\mu$"),
        "inter_edge_ratio": (
            "Inter-community edge ratio\n(bridge fraction fixed at 0.20)",
            "Measured inter-edge ratio",
        ),
        "bridge_node_fraction": (
            "Bridge-node fraction\n(inter-edge ratio fixed at 0.20)",
            "Measured bridge-node fraction",
        ),
    }
    for ax, experiment in zip(axes, labels):
        subset = [row for row in rows if row["experiment"] == experiment and row["method"] == "STRC"]
        targets = sorted(set(float(row["target_parameter"]) for row in subset))
        x_values, x_errors, advantages, advantage_stds = [], [], [], []
        for target in targets:
            group = [row for row in subset if float(row["target_parameter"]) == target]
            measured = np.array([float(row["parameter"]) for row in group])
            delta = np.array([float(row["strc_delta_auc"]) for row in group])
            x_values.append(float(np.mean(measured)))
            x_errors.append(float(np.std(measured)))
            advantages.append(float(np.mean(delta)))
            advantage_stds.append(float(np.std(delta)))
        ax.errorbar(
            x_values,
            advantages,
            xerr=x_errors if experiment != "mu" else None,
            yerr=advantage_stds,
            color="#E64B35",
            marker="o",
            linewidth=2,
            capsize=3,
        )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xlabel(labels[experiment][1])
        ax.set_ylabel("STRC advantage (AUC points)")
        ax.set_title(labels[experiment][0], loc="left")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(RESULT_FIG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = []
    rows.extend(run_sweep("mu", MU_VALUES, repeats=REPEATS, n=DEFAULT_N))
    rows.extend(run_sweep("inter_edge_ratio", EDGE_RATIO_VALUES, repeats=REPEATS, n=DEFAULT_N))
    rows.extend(run_sweep("bridge_node_fraction", BRIDGE_FRACTION_VALUES, repeats=REPEATS, n=DEFAULT_N))
    write_csv(rows)
    plot_gain(rows)
    print(f"Wrote {len(rows)} rows to {RESULT_CSV}")
    print(f"Wrote figure to {RESULT_FIG}")


if __name__ == "__main__":
    main()
