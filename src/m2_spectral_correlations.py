"""Reviewer 5 M2: spectral correlations for STRC, NC, and spectral radius.

This is a standalone experiment for the six real-world networks used in the
manuscript.  It reports node-level Spearman correlations between:

* STRC: the JL/Laplacian approximation of spanning-tree reduction centrality;
* NC: the first-order node-deletion contribution to natural connectivity,
  ``log(mean(exp(adjacency eigenvalues)))``;
* SR: the first-order node-deletion contribution to the adjacency spectral
  radius, ``2 * lambda_1 * x_1(i)^2``.

For large sparse networks, NC and SR use leading adjacency eigenpairs and
STRC uses a fixed Rademacher projection.  For an independently selected small
connected induced subgraph from every real network, the same node scores are
also recomputed exactly (dense eigendecomposition and Matrix--Tree theorem).
The approximation-vs-exact validation is written alongside the main results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr

from laplacian_solver import solve_laplacian_sdd


NETWORK_FILES = (
    "CA-HepPh.txt",
    "Email-Enron.txt",
    "cit-HepTh.txt",
    "bio-dmela.txt",
    "Wiki-Vote.txt",
    "tech-as-caida.txt",
)
METHODS = ("STRC", "NC", "SR")


def load_network(path: Path) -> nx.Graph:
    """Read an edge list and retain its largest connected component."""
    edges = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                edges.append((fields[0], fields[1]))
    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() == 0:
        raise ValueError(f"empty network: {path}")
    if not nx.is_connected(graph):
        component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(component).copy()
    return graph


def _solve_laplacian(laplacian: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve a singular Laplacian system with the project solver and fallback."""
    try:
        solution, _ = solve_laplacian_sdd(
            laplacian,
            rhs,
            tolerance=1e-4,
            max_cycles=150,
            strict=False,
        )
        return np.asarray(solution)
    except Exception as exc:
        print(f"[STRC] AMG fallback ({type(exc).__name__}: {exc})")
        shifted = laplacian + sp.eye(laplacian.shape[0], format="csr") * 1e-7
        columns = []
        for column in range(rhs.shape[1]):
            result = spla.lsmr(shifted, rhs[:, column], atol=1e-5, btol=1e-5)[0]
            columns.append(result)
        return np.column_stack(columns)


def strc_scores(graph: nx.Graph, projections: int, seed: int) -> dict:
    """Compute the scalable STRC approximation used by the manuscript."""
    nodes = list(graph)
    n = len(nodes)
    if n < 2:
        return {node: 0.0 for node in nodes}

    index = {node: i for i, node in enumerate(nodes)}
    laplacian = nx.laplacian_matrix(graph, nodelist=nodes).astype(float).tocsr()
    incidence = nx.incidence_matrix(
        graph, nodelist=nodes, oriented=True, weight=None
    ).astype(float).tocsr()
    rng = np.random.default_rng(seed)
    projections = max(8, int(projections))
    edge_probes = rng.choice(
        np.array([-1.0, 1.0]), size=(incidence.shape[1], projections)
    ) / np.sqrt(projections)
    rhs = incidence @ edge_probes
    embedding = _solve_laplacian(laplacian, rhs)
    embedding -= embedding.mean(axis=0, keepdims=True)

    scores = {}
    for node in nodes:
        i = index[node]
        neighbors = list(graph.neighbors(node))
        degree = len(neighbors)
        if degree <= 1:
            scores[node] = 0.0
            continue
        neighbor_indices = [index[neighbor] for neighbor in neighbors]
        differences = embedding[neighbor_indices] - embedding[i]
        gram = differences @ differences.T
        matrix = np.eye(degree) - gram + np.ones((degree, degree)) / degree
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0 or not np.isfinite(logdet):
            scores[node] = 0.0
            continue
        ratio = float(np.exp(np.clip(logdet - np.log(degree), -745.0, 0.0)))
        scores[node] = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return scores


def _leading_adjacency_eigenpairs(
    graph: nx.Graph, eigenpairs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return leading adjacency eigenvalues and eigenvectors."""
    nodes = list(graph)
    adjacency = nx.adjacency_matrix(graph, nodelist=nodes).astype(float).tocsr()
    n = len(nodes)
    if n <= 2:
        values, vectors = np.linalg.eigh(adjacency.toarray())
        order = np.argsort(values)[::-1]
        return values[order], vectors[:, order]
    k = min(max(2, int(eigenpairs)), n - 2)
    if k >= n - 1:
        values, vectors = np.linalg.eigh(adjacency.toarray())
        order = np.argsort(values)[::-1][:k]
        return values[order], vectors[:, order]
    values, vectors = spla.eigsh(adjacency, k=k, which="LA", tol=1e-5, maxiter=1000)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def natural_connectivity_scores(
    graph: nx.Graph, eigenpairs: int
) -> tuple[dict, float]:
    """Return node-deletion NC scores and the graph-level NC estimate.

    The node score uses the same trace/eigenvector first-order approximation
    as the existing NC baseline: removing node ``i`` removes its weighted
    diagonal contribution to ``tr(exp(A))`` and changes the normalization from
    ``N`` to ``N-1``.
    """
    nodes = list(graph)
    n = len(nodes)
    values, vectors = _leading_adjacency_eigenpairs(graph, eigenpairs)
    shift = float(values[0])
    weights = np.exp(np.clip(values - shift, -745.0, 0.0))
    total = float(weights.sum())
    contributions = (vectors * vectors) @ weights
    denominator = np.maximum(total - contributions, np.finfo(float).tiny)
    scores_array = np.log(np.maximum(total * (n - 1), np.finfo(float).tiny) / (n * denominator))
    graph_nc = float(shift + np.log(total / n))
    return {node: float(score) for node, score in zip(nodes, scores_array)}, graph_nc


def spectral_radius_scores(graph: nx.Graph) -> tuple[dict, float]:
    """Return first-order node-deletion spectral-radius scores."""
    nodes = list(graph)
    adjacency = nx.adjacency_matrix(graph, nodelist=nodes).astype(float).tocsr()
    value, vector = spla.eigsh(adjacency, k=1, which="LA", tol=1e-5, maxiter=1000)
    radius = float(value[0])
    scores = 2.0 * radius * np.square(np.asarray(vector[:, 0]).ravel())
    return {node: float(score) for node, score in zip(nodes, scores)}, radius


def _natural_connectivity(adjacency: np.ndarray) -> float:
    """Compute NC exactly from all adjacency eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(adjacency)
    # log-sum-exp avoids overflow for dense validation graphs.
    shift = float(np.max(eigenvalues))
    return float(shift + np.log(np.exp(eigenvalues - shift).sum() / len(eigenvalues)))


def exact_natural_connectivity_scores(graph: nx.Graph) -> tuple[dict, float]:
    """Recompute every leave-one-node NC value using a dense eigendecomposition."""
    nodes = list(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    base = _natural_connectivity(adjacency)
    scores = {}
    for index, node in enumerate(nodes):
        reduced = np.delete(np.delete(adjacency, index, axis=0), index, axis=1)
        scores[node] = float(base - _natural_connectivity(reduced))
    return scores, base


def exact_spectral_radius_scores(graph: nx.Graph) -> tuple[dict, float]:
    """Recompute every leave-one-node spectral radius by dense eigendecomposition."""
    nodes = list(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    base = float(np.max(np.linalg.eigvalsh(adjacency)))
    scores = {}
    for index, node in enumerate(nodes):
        reduced = np.delete(np.delete(adjacency, index, axis=0), index, axis=1)
        reduced_radius = float(np.max(np.linalg.eigvalsh(reduced))) if reduced.size else 0.0
        scores[node] = float(base - reduced_radius)
    return scores, base


def _log_spanning_tree_count(graph: nx.Graph) -> float:
    """Return log(tau(G)) using one reduced Laplacian cofactor."""
    if graph.number_of_nodes() <= 1:
        return 0.0
    if not nx.is_connected(graph):
        return -np.inf
    nodes = list(graph)
    laplacian = nx.laplacian_matrix(graph, nodelist=nodes).astype(float).toarray()
    sign, logdet = np.linalg.slogdet(laplacian[:-1, :-1])
    return float(logdet) if sign > 0 and np.isfinite(logdet) else -np.inf


def exact_strc_scores(graph: nx.Graph) -> dict:
    """Compute exact relative spanning-tree reductions for a small graph."""
    nodes = list(graph)
    base_log_tau = _log_spanning_tree_count(graph)
    scores = {}
    for node in nodes:
        reduced = graph.copy()
        reduced.remove_node(node)
        reduced_log_tau = _log_spanning_tree_count(reduced)
        if not np.isfinite(base_log_tau) or not np.isfinite(reduced_log_tau):
            scores[node] = 1.0 if np.isfinite(base_log_tau) else 0.0
        else:
            # Work in log space because spanning-tree counts are enormous.
            ratio = np.exp(np.clip(reduced_log_tau - base_log_tau, -745.0, 0.0))
            scores[node] = float(1.0 - ratio)
    return scores


def connected_induced_sample(graph: nx.Graph, max_nodes: int, seed: int) -> nx.Graph:
    """Select a deterministic connected induced subgraph for exact validation."""
    if graph.number_of_nodes() <= max_nodes:
        return graph.copy()
    nodes = list(graph)
    start = nodes[np.random.default_rng(seed).integers(0, len(nodes))]
    selected = []
    seen = {start}
    queue = [start]
    while queue and len(selected) < max_nodes:
        node = queue.pop(0)
        selected.append(node)
        neighbors = sorted(graph.neighbors(node), key=lambda value: str(value))
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return graph.subgraph(selected).copy()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    data_dir: Path,
    output_dir: Path,
    projections: int,
    eigenpairs: int,
    seed: int,
    validation_nodes: int = 80,
    network_files: tuple[str, ...] = NETWORK_FILES,
) -> tuple[list[dict], list[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    correlation_rows = []
    wide_rows = []
    validation_rows_summary = []
    metadata = {
        "experiment": "Reviewer 5 M2 spectral relationship",
        "methods": list(METHODS),
        "strc_projections": projections,
        "adjacency_eigenpairs": eigenpairs,
        "exact_validation_nodes": validation_nodes,
        "seed": seed,
        "node_score_definitions": {
            "STRC": "JL/Laplacian approximation of spanning-tree reduction centrality",
            "NC": "first-order Delta log(mean(exp(mu))) under node deletion",
            "SR": "first-order Delta spectral radius: 2*lambda_1*x_1(i)^2",
        },
        "networks": [],
    }

    for network_index, filename in enumerate(network_files):
        path = data_dir / filename
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        print(f"[network {network_index + 1}/{len(network_files)}] loading {filename}")
        graph = load_network(path)
        nodes = list(graph)
        name = path.stem
        print(f"  N={graph.number_of_nodes()}, M={graph.number_of_edges()}; STRC")
        scores = {
            "STRC": strc_scores(graph, projections, seed + network_index),
        }
        print("  NC and spectral radius")
        scores["NC"], graph_nc = natural_connectivity_scores(graph, eigenpairs)
        scores["SR"], radius = spectral_radius_scores(graph)

        node_rows = []
        for node in nodes:
            row = {"node": node}
            row.update({method: scores[method][node] for method in METHODS})
            node_rows.append(row)
        _write_csv(
            output_dir / f"m2_node_scores_{name}.csv",
            node_rows,
            ["node", *METHODS],
        )

        matrix = np.array([[scores[method][node] for method in METHODS] for node in nodes])
        wide = {"network": name, "N": len(nodes), "M": graph.number_of_edges()}
        for left_index, left in enumerate(METHODS):
            for right in METHODS[left_index + 1 :]:
                rho, pvalue = spearmanr(matrix[:, left_index], matrix[:, METHODS.index(right)])
                key = f"{left}_vs_{right}"
                wide[f"{key}_rho"] = float(rho)
                wide[f"{key}_pvalue"] = float(pvalue)
                correlation_rows.append(
                    {
                        "network": name,
                        "N": len(nodes),
                        "M": graph.number_of_edges(),
                        "metric_a": left,
                        "metric_b": right,
                        "spearman_rho": float(rho),
                        "pvalue": float(pvalue),
                    }
                )
        wide_rows.append(wide)

        # Exact validation is deliberately restricted to a small connected
        # induced subgraph.  This keeps the validation finite while testing
        # the same node-level quantities used in the six-network analysis.
        validation_graph = connected_induced_sample(
            graph, validation_nodes, seed + 1000 + network_index
        )
        print(
            f"  exact validation on connected induced subgraph "
            f"N={validation_graph.number_of_nodes()}, "
            f"M={validation_graph.number_of_edges()}"
        )
        exact_scores = {
            "STRC": exact_strc_scores(validation_graph),
        }
        exact_scores["NC"], exact_nc = exact_natural_connectivity_scores(
            validation_graph
        )
        exact_scores["SR"], exact_radius = exact_spectral_radius_scores(
            validation_graph
        )
        approx_validation = {
            "STRC": strc_scores(
                validation_graph, projections, seed + 2000 + network_index
            ),
            "NC": natural_connectivity_scores(validation_graph, eigenpairs)[0],
            "SR": spectral_radius_scores(validation_graph)[0],
        }
        validation_rows = []
        for node in validation_graph:
            row = {"node": node}
            for method in METHODS:
                row[f"{method}_approx"] = approx_validation[method][node]
                row[f"{method}_exact"] = exact_scores[method][node]
            validation_rows.append(row)
        _write_csv(
            output_dir / f"m2_exact_validation_scores_{name}.csv",
            validation_rows,
            [
                "node",
                *[field for method in METHODS for field in (f"{method}_approx", f"{method}_exact")],
            ],
        )
        for method in METHODS:
            values_approx = np.array(
                [approx_validation[method][node] for node in validation_graph]
            )
            values_exact = np.array(
                [exact_scores[method][node] for node in validation_graph]
            )
            validation_rho, validation_p = spearmanr(values_approx, values_exact)
            validation_rows_summary.append(
                {
                    "network": name,
                    "validation_nodes": validation_graph.number_of_nodes(),
                    "validation_edges": validation_graph.number_of_edges(),
                    "metric": method,
                    "spearman_rho_approx_vs_exact": float(validation_rho),
                    "pvalue": float(validation_p),
                }
            )
        metadata["networks"].append(
            {
                "name": name,
                "N": len(nodes),
                "M": graph.number_of_edges(),
                "natural_connectivity": graph_nc,
                "spectral_radius": radius,
                "exact_validation": {
                    "N": validation_graph.number_of_nodes(),
                    "M": validation_graph.number_of_edges(),
                    "natural_connectivity": exact_nc,
                    "spectral_radius": exact_radius,
                },
            }
        )

    _write_csv(
        output_dir / "m2_spectral_correlations.csv",
        correlation_rows,
        ["network", "N", "M", "metric_a", "metric_b", "spearman_rho", "pvalue"],
    )
    wide_fields = ["network", "N", "M"]
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1 :]:
            wide_fields.extend([f"{left}_vs_{right}_rho", f"{left}_vs_{right}_pvalue"])
    _write_csv(output_dir / "m2_spectral_correlations_wide.csv", wide_rows, wide_fields)
    _write_csv(
        output_dir / "m2_approximation_validation.csv",
        validation_rows_summary,
        [
            "network",
            "validation_nodes",
            "validation_edges",
            "metric",
            "spearman_rho_approx_vs_exact",
            "pvalue",
        ],
    )
    metadata["completed_networks"] = len(metadata["networks"])
    (output_dir / "m2_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return correlation_rows, wide_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=script_dir.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "results" / "m2_spectral")
    parser.add_argument("--strc-projections", type=int, default=400)
    parser.add_argument("--adjacency-eigenpairs", type=int, default=64)
    parser.add_argument(
        "--exact-validation-nodes",
        type=int,
        default=80,
        help="nodes in each connected induced subgraph used for exact validation",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--networks", nargs="*", default=list(NETWORK_FILES))
    args = parser.parse_args()
    if args.exact_validation_nodes < 3:
        parser.error("--exact-validation-nodes must be at least 3")
    run_experiment(
        args.data_dir,
        args.output_dir,
        args.strc_projections,
        args.adjacency_eigenpairs,
        args.seed,
        args.exact_validation_nodes,
        tuple(args.networks),
    )


if __name__ == "__main__":
    main()
