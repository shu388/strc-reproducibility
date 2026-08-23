"""Advanced node-ranking baselines shared by the dismantling and SIR scripts.

CE and RT are evaluated from shortest-path information. Exact all-pairs
evaluation is used on small graphs; for larger graphs, a fixed landmark sample
keeps the ranking runs reproducible and computationally feasible.
"""

from __future__ import annotations

from collections import defaultdict
import math
import random

import networkx as nx


DEFAULT_SEED = 20260815


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
