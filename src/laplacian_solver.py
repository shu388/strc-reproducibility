"""Practical near-linear solver for compatible graph Laplacian systems.

The implementation uses one PyAMG hierarchy per graph and reuses it for all
right-hand sides. It intentionally has no fallback to unpreconditioned CG or
sparse LU: such a fallback would make the experimental implementation differ
from the solver described in the manuscript.
"""

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class LaplacianSolveStats:
    setup_seconds: float
    solve_seconds: float
    cycles: tuple
    relative_residuals: tuple

    @property
    def mean_cycles(self):
        return float(np.mean(self.cycles)) if self.cycles else 0.0

    @property
    def max_cycles(self):
        return int(max(self.cycles, default=0))

    @property
    def max_relative_residual(self):
        return float(max(self.relative_residuals, default=0.0))


def _load_pyamg():
    try:
        import pyamg
    except ImportError as exc:
        raise ImportError(
            "PyAMG is required for the near-linear Laplacian solver. "
            "Install it in the experiment environment with: pip install pyamg"
        ) from exc
    return pyamg


def solve_laplacian_sdd(
    laplacian,
    rhs,
    *,
    tolerance=1e-5,
    max_cycles=100,
    cycle="V",
    strict=True,
):
    """Approximately solve L X = rhs for a connected graph Laplacian.

    A grounded principal minor removes the one-dimensional null space. Since
    every column of B.T @ Q.T is orthogonal to the all-ones vector, solving the
    grounded systems gives a valid solution of the original systems. Centering
    each result selects the minimum-norm gauge without changing the node-pair
    differences used to construct H_i.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_cycles <= 0:
        raise ValueError("max_cycles must be positive")

    matrix = sp.csr_matrix(laplacian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("laplacian must be a square matrix")

    n = matrix.shape[0]
    if n < 2:
        raise ValueError("the Laplacian solver requires at least two nodes")
    if connected_components(matrix, directed=False, return_labels=False) != 1:
        raise ValueError("the Laplacian solver requires a connected graph")

    right_hand_side = np.asarray(rhs, dtype=float)
    was_vector = right_hand_side.ndim == 1
    if was_vector:
        right_hand_side = right_hand_side[:, None]
    if right_hand_side.ndim != 2 or right_hand_side.shape[0] != n:
        raise ValueError("rhs must have shape (N,) or (N, K)")
    if not np.all(np.isfinite(right_hand_side)):
        raise ValueError("rhs contains non-finite values")

    # Remove roundoff-level components in null(L). This is exact for B.T @ Q.T.
    compatible_rhs = right_hand_side - right_hand_side.mean(axis=0, keepdims=True)
    grounded_matrix = matrix[1:, 1:].tocsr()
    grounded_rhs = compatible_rhs[1:, :]

    pyamg = _load_pyamg()
    setup_start = perf_counter()
    hierarchy = pyamg.smoothed_aggregation_solver(
        grounded_matrix,
        symmetry="symmetric",
    )
    setup_seconds = perf_counter() - setup_start

    solution = np.zeros_like(compatible_rhs)
    cycles = []
    solve_start = perf_counter()
    for column in range(grounded_rhs.shape[1]):
        residual_history = []
        solution[1:, column] = hierarchy.solve(
            grounded_rhs[:, column],
            tol=tolerance,
            maxiter=max_cycles,
            cycle=cycle,
            residuals=residual_history,
        )
        cycles.append(max(0, len(residual_history) - 1))
    solve_seconds = perf_counter() - solve_start

    solution -= solution.mean(axis=0, keepdims=True)
    residual = matrix @ solution - compatible_rhs
    rhs_norm = np.linalg.norm(compatible_rhs, axis=0)
    residual_norm = np.linalg.norm(residual, axis=0)
    relative_residuals = residual_norm / np.maximum(
        rhs_norm, np.finfo(float).tiny
    )

    if not np.all(np.isfinite(solution)):
        raise RuntimeError("the Laplacian solver returned non-finite values")
    max_relative_residual = np.max(relative_residuals, initial=0.0)
    if max_relative_residual > 10.0 * tolerance:
        message = (
            "the Laplacian solver did not reach the requested tolerance; "
            f"maximum relative residual={max_relative_residual:.3e}"
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(
            message + "; returning the best finite AMG iterate",
            RuntimeWarning,
            stacklevel=2,
        )

    stats = LaplacianSolveStats(
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        cycles=tuple(int(value) for value in cycles),
        relative_residuals=tuple(float(value) for value in relative_residuals),
    )
    if was_vector:
        return solution[:, 0], stats
    return solution, stats
