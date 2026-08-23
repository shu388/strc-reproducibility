"""Faithful Python reproduction of Fast-TreeC Algorithm 2.

The original implementation uses CMG/PCG.  This port preserves its streaming
Rademacher projections, 2-core shortcut, scale rule, and one reusable AMG
hierarchy, while replacing the unavailable CMG backend with smoothed
aggregation AMG.  It must therefore be reported as a Fast-TreeC reproduction
with an SA-AMG backend, not as the original CMG wall-clock result.
"""

from time import perf_counter

import networkx as nx
import numpy as np
import scipy.sparse as sp


def _streaming_solver(laplacian, rhs_columns, tolerance, hierarchy):
    """Solve grounded compatible Laplacian systems with one AMG hierarchy."""
    n = laplacian.shape[0]
    grounded = laplacian[1:, 1:].tocsr()
    solution = np.zeros((n, rhs_columns.shape[1]), dtype=float)
    for column in range(rhs_columns.shape[1]):
        rhs = rhs_columns[1:, column]
        residuals = []
        solution[1:, column] = hierarchy.solve(
            rhs,
            tol=tolerance,
            maxiter=300,
            cycle="V",
            residuals=residuals,
        )
    solution -= solution.mean(axis=0, keepdims=True)
    return solution


def _amg_compatible(matrix):
    """Return CSR with 32-bit indices required by the PyAMG C kernels."""
    matrix = sp.csr_matrix(matrix, dtype=float)
    matrix.sum_duplicates()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def fast_treec_scores(G, epsilon=1.0, tolerance=1e-4, seed=0):
    """Return approximate SC scores and elapsed seconds for all graph edges.

    The score order follows ``list(G.edges())``.  Edges outside the 2-core are
    assigned score one as in the paper's Extract2Core speedup.
    """
    if not nx.is_connected(G):
        raise ValueError("Fast-TreeC requires a connected graph")
    if epsilon <= 0 or tolerance <= 0:
        raise ValueError("epsilon and tolerance must be positive")

    start = perf_counter()
    edges = list(G.edges())
    edge_scores = np.ones(len(edges), dtype=float)
    core = nx.k_core(G, k=2)
    if core.number_of_nodes() < 2 or core.number_of_edges() == 0:
        return edge_scores, perf_counter() - start

    core_nodes = list(core.nodes())
    core_index = {node: index for index, node in enumerate(core_nodes)}
    core_edges = list(core.edges())
    laplacian = nx.laplacian_matrix(core, nodelist=core_nodes).tocsr().astype(float)
    incidence = nx.incidence_matrix(
        core,
        nodelist=core_nodes,
        edgelist=core_edges,
        oriented=True,
    ).T.tocsr()

    # Algorithm 2 and EffectiveResistances.m use ceil(log2(n))/epsilon.
    scale = int(np.ceil(np.log2(len(core_nodes))) / epsilon)
    scale = max(scale, 1)

    import pyamg

    hierarchy = pyamg.smoothed_aggregation_solver(
        _amg_compatible(laplacian[1:, 1:]),
        symmetry="symmetric",
    )
    rng = np.random.default_rng(seed)
    core_scores = np.zeros(len(core_edges), dtype=float)
    source = np.fromiter(
        (core_index[u] for u, _ in core_edges), dtype=np.int64
    )
    target = np.fromiter(
        (core_index[v] for _, v in core_edges), dtype=np.int64
    )
    for _ in range(scale):
        q = rng.integers(0, 2, size=len(core_edges), dtype=np.int8).astype(float)
        q *= 2.0
        q -= 1.0
        q /= np.sqrt(scale)
        rhs = np.asarray(incidence.T @ q).reshape(-1, 1)
        embedding = _streaming_solver(laplacian, rhs, tolerance, hierarchy)[:, 0]
        differences = embedding[source] - embedding[target]
        core_scores += differences * differences

    core_edge_index = {tuple(sorted(edge)): index for index, edge in enumerate(core_edges)}
    for edge_index, (u, v) in enumerate(edges):
        key = tuple(sorted((u, v)))
        if key in core_edge_index:
            edge_scores[edge_index] = core_scores[core_edge_index[key]]
    return edge_scores, perf_counter() - start


def time_fast_treec(G, epsilon=1.0, tolerance=1e-4, seed=0):
    """Return only the end-to-end Fast-TreeC reproduction runtime."""
    _, seconds = fast_treec_scores(G, epsilon=epsilon, tolerance=tolerance, seed=seed)
    return seconds
