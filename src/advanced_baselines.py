"""Advanced node-ranking baselines shared by the dismantling and SIR scripts.

CE and RT are evaluated from shortest-path information. Exact all-pairs
evaluation is used on small graphs; for larger graphs, a fixed landmark sample
keeps the ranking runs reproducible and computationally feasible. The M1
baselines include an exact leave-one-out implementation and a JL/Laplacian-
solver approximation for effective-resistance efficiency. The latter is used
for large real-world networks; the Kirchhoff-index and average-resistance
versions remain exact-only in the experiment drivers.
"""

from __future__ import annotations

from collections import defaultdict
import math
import random

import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla
from laplacian_solver import solve_laplacian_sdd


DEFAULT_SEED = 20260815


def _largest_connected_subgraph(G: nx.Graph) -> nx.Graph:
    """Return a copy of the largest connected component of ``G``."""
    if G.number_of_nodes() == 0 or nx.is_connected(G):
        return G.copy()
    component = max(nx.connected_components(G), key=len)
    return G.subgraph(component).copy()


def _global_resistance_indices(G: nx.Graph) -> tuple[float, float]:
    """Return effective-resistance efficiency and Kirchhoff index exactly.

    The calculation is intentionally dense and is therefore reserved for the
    small generated networks in the M1 experiments.  Disconnected inputs are
    reduced to their largest connected component before the pseudoinverse is
    evaluated.
    """
    H = _largest_connected_subgraph(G)
    nodes = list(H)
    n = len(nodes)
    if n < 2:
        return 0.0, 0.0

    laplacian = nx.laplacian_matrix(H, nodelist=nodes).astype(float).toarray()
    laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(laplacian_pinv)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * laplacian_pinv
    np.maximum(resistance, 0.0, out=resistance)
    off_diagonal = ~np.eye(n, dtype=bool)
    values = resistance[off_diagonal]
    reciprocal = np.divide(
        1.0,
        values,
        out=np.zeros_like(values),
        where=values > 1e-12,
    )
    efficiency = float(np.mean(reciprocal))
    kirchhoff_index = float(np.sum(values) / 2.0)
    return efficiency, kirchhoff_index


def effective_resistance_baselines(G: nx.Graph) -> dict[str, dict]:
    """Compute the three M1 baselines for a small graph.

    ``ER`` is the leave-one-out change in effective-resistance network
    efficiency, ``KI`` is the leave-one-out change in Kirchhoff index, and
    ``AER`` is the negative average effective resistance from each node to all
    other nodes (negative so that larger values are ranked first).  The first
    two require one dense pseudoinverse per removed node, so callers should
    use this exact function for generated networks only. Real-network ER is
    handled separately by ``approximate_effective_resistance_efficiency``.
    """
    H = _largest_connected_subgraph(G)
    nodes = list(H)
    n = len(nodes)
    zero = {node: 0.0 for node in nodes}
    if n < 2:
        return {"ER": zero.copy(), "KI": zero.copy(), "AER": zero}

    laplacian = nx.laplacian_matrix(H, nodelist=nodes).astype(float).toarray()
    laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(laplacian_pinv)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * laplacian_pinv
    np.maximum(resistance, 0.0, out=resistance)
    average_resistance = np.sum(resistance, axis=1) / (n - 1)

    base_efficiency, base_kirchhoff = _global_resistance_indices(H)
    efficiency_change = {}
    kirchhoff_change = {}
    for node in nodes:
        reduced = H.copy()
        reduced.remove_node(node)
        reduced_efficiency, reduced_kirchhoff = _global_resistance_indices(reduced)
        efficiency_change[node] = base_efficiency - reduced_efficiency
        kirchhoff_change[node] = base_kirchhoff - reduced_kirchhoff

    return {
        "ER": efficiency_change,
        "KI": kirchhoff_change,
        "AER": {
            node: -float(value)
            for node, value in zip(nodes, average_resistance)
        },
    }


def approximate_effective_resistance_efficiency(
    G: nx.Graph,
    projection_count: int = 64,
    landmark_count: int = 256,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Return a scalable proxy for node-deletion ER efficiency impact.

    A JL embedding of effective resistance is obtained by solving a batch of
    Laplacian systems with Rademacher edge probes.  For each node, the mean of
    ``1 / R_uv`` over a fixed landmark sample estimates
    ``s_i = sum_{v != i} 1/R_iv``. If resistances between surviving node pairs
    are held fixed, the deletion score satisfies
    ``Delta E_i = constant + 2*s_i/((N-1)(N-2))``; ranking by ``s_i`` is
    therefore the corresponding first-order, no-rerouting approximation.
    Unlike the exact generated-network score, this scalable proxy does not
    recompute resistance changes among surviving pairs in every ``G-i``.
    """
    H = _largest_connected_subgraph(G)
    nodes = list(H)
    n = len(nodes)
    if n < 2:
        return {node: 0.0 for node in nodes}

    projection_count = max(8, int(projection_count))
    laplacian = nx.laplacian_matrix(H, nodelist=nodes).astype(float).tocsr()
    incidence = nx.incidence_matrix(
        H,
        nodelist=nodes,
        oriented=True,
        weight=None,
    ).astype(float).tocsr()
    rng = np.random.default_rng(seed)
    edge_probes = rng.choice(
        np.array([-1.0, 1.0]),
        size=(incidence.shape[1], projection_count),
    ) / np.sqrt(projection_count)
    rhs = incidence @ edge_probes
    try:
        embedding, _ = solve_laplacian_sdd(
            laplacian,
            rhs,
            tolerance=1e-4,
            max_cycles=150,
            strict=False,
        )
    except Exception:
        embedding = np.column_stack(
            [
                spla.lsmr(
                    laplacian,
                    rhs[:, column],
                    atol=1e-5,
                    btol=1e-5,
                )[0]
                for column in range(projection_count)
            ]
        )
    embedding -= embedding.mean(axis=0, keepdims=True)

    landmark_count = min(max(16, int(landmark_count)), n)
    landmarks = rng.choice(n, size=landmark_count, replace=False)
    landmark_embedding = embedding[landmarks]
    landmark_position = np.full(n, -1, dtype=int)
    landmark_position[landmarks] = np.arange(landmark_count)
    scores = np.zeros(n, dtype=float)
    for start in range(0, n, 2048):
        stop = min(start + 2048, n)
        differences = embedding[start:stop, None, :] - landmark_embedding[None, :, :]
        resistance = np.sum(differences * differences, axis=2)
        local_indices = np.arange(start, stop)
        positions = landmark_position[local_indices]
        sampled_rows = np.flatnonzero(positions >= 0)
        resistance[sampled_rows, positions[sampled_rows]] = np.inf
        reciprocal = np.divide(
            1.0,
            resistance,
            out=np.zeros_like(resistance),
            where=np.isfinite(resistance) & (resistance > 1e-12),
        )
        scores[start:stop] = np.sum(reciprocal, axis=1) / np.maximum(
            np.sum(reciprocal > 0, axis=1),
            1,
        )
    return {node: float(score) for node, score in zip(nodes, scores)}


def collective_influence_scores(G: nx.Graph, radius: int = 2) -> dict:
    """Return the standard Collective Influence score CI_radius for each node."""
    degree = dict(G.degree())
    scores = {}
    for node in G:
        distances = nx.single_source_shortest_path_length(G, node, cutoff=radius)
        boundary = (other for other, distance in distances.items() if distance == radius)
        scores[node] = (degree[node] - 1) * sum(degree[other] - 1 for other in boundary)
    return scores


def closeness_energy_scores(
    G: nx.Graph, landmark_count: int = 256, seed: int = DEFAULT_SEED
) -> dict:
    """Estimate closeness energy using the distance-decay kernel exp(-d_ij).

    On an undirected graph, summing contributions from sampled landmarks gives
    an unbiased rescaled estimate of the all-pairs score. The scale factor does
    not change the ranking and is retained for comparability across networks.
    """
    nodes = list(G)
    if not nodes:
        return {}

    landmark_count = min(landmark_count, len(nodes))
    rng = random.Random(seed)
    landmarks = nodes if landmark_count == len(nodes) else rng.sample(nodes, landmark_count)
    scores = dict.fromkeys(nodes, 0.0)

    for landmark in landmarks:
        for node, distance in nx.single_source_shortest_path_length(G, landmark).items():
            if node != landmark:
                scores[node] += math.exp(-distance)

    scale = len(nodes) / landmark_count
    return {node: value * scale for node, value in scores.items()}


def _branching_intervening_mass(degree: int, distance: int, mean_excess_degree: float) -> float:
    """Mean-field mass between a source and a landmark target for large graphs."""
    if distance <= 1:
        return 0.0
    if abs(mean_excess_degree - 1.0) < 1e-12:
        return float(degree * (distance - 1))
    return float(degree * (mean_excess_degree ** (distance - 1) - 1.0) / (mean_excess_degree - 1.0))


def radiation_theory_scores(
    G: nx.Graph,
    exact_node_limit: int = 2000,
    landmark_count: int = 128,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Return radiation-theory influence scores.

    The score accumulates the radiation-model flux from a source to all other
    nodes, using node degree as the node mass. For graphs up to
    ``exact_node_limit`` the intervening mass is evaluated exactly from BFS
    shells. Larger graphs use a fixed landmark approximation and a branching
    mean-field estimate of the intervening mass. This is essential for the
    large real-world networks and is reported as an approximation.
    """
    nodes = list(G)
    if not nodes:
        return {}

    degree = dict(G.degree())
    mean_degree = sum(degree.values()) / len(nodes)
    numerator = sum(d * (d - 1) for d in degree.values())
    denominator = sum(degree.values())
    mean_excess_degree = numerator / denominator if denominator else 1.0

    def flux(source, target, intervening_mass):
        source_mass = max(float(degree[source]), 1.0)
        target_mass = max(float(degree[target]), 1.0)
        return source_mass * target_mass / (
            (source_mass + intervening_mass)
            * (source_mass + target_mass + intervening_mass)
        )

    if len(nodes) <= exact_node_limit:
        scores = {}
        for source in nodes:
            distances = nx.single_source_shortest_path_length(G, source)
            shell_mass = defaultdict(float)
            for target, distance in distances.items():
                shell_mass[distance] += degree[target]

            cumulative_mass = 0.0
            score = 0.0
            for distance in sorted(shell_mass):
                if distance == 0:
                    cumulative_mass += shell_mass[distance]
                    continue
                for target, target_distance in distances.items():
                    if target_distance == distance:
                        score += flux(source, target, cumulative_mass)
                cumulative_mass += shell_mass[distance]
            scores[source] = score
        return scores

    sample_count = min(landmark_count, len(nodes))
    rng = random.Random(seed)
    landmarks = rng.sample(nodes, sample_count)
    scores = dict.fromkeys(nodes, 0.0)
    for target in landmarks:
        distances = nx.single_source_shortest_path_length(G, target)
        for source, distance in distances.items():
            if source == target:
                continue
            intervening_mass = _branching_intervening_mass(
                degree[source], distance, mean_excess_degree
            )
            scores[source] += flux(source, target, intervening_mass)

    scale = len(nodes) / sample_count
    return {node: value * scale for node, value in scores.items()}
