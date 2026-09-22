"""Reviewer 5 M3: find the K at which WS rank agreement reaches 0.95.

This experiment complements m3_k_sensitivity.py. It scans K=400...4000 on
50 independent controlled WS graphs. Every K is a nested prefix of the same
K_max Rademacher projection within a graph, exact STRC is the reference, and
graph realizations are the independent units for Student-t 95% intervals.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
M3_MODULE_PATH = SCRIPT_DIR / "m3_k_sensitivity.py"
M3_SPEC = importlib.util.spec_from_file_location("m3_k_sensitivity", M3_MODULE_PATH)
if M3_SPEC is None or M3_SPEC.loader is None:
    raise ImportError(f"Could not load shared M3 implementation from {M3_MODULE_PATH}")
m3 = importlib.util.module_from_spec(M3_SPEC)
sys.modules[M3_SPEC.name] = m3
M3_SPEC.loader.exec_module(m3)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_K_VALUES = (
    400,
    800,
    1200,
    1600,
    1800,
    2000,
    2200,
    2400,
    2600,
    2800,
    3000,
    3200,
    3600,
    4000,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows: list[dict], k_values: tuple[int, ...]) -> list[dict]:
    summary = []
    for K in k_values:
        rows = [row for row in raw_rows if int(row["K"]) == K]
        if not rows:
            continue
        output = {
            "network": "WS",
            "K": K,
            "graph_repetitions": len({int(row["graph_repeat"]) for row in rows}),
        }
        for metric in ("spearman", "kendall", "top10_overlap", "score_seconds"):
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_95ci"] = m3._ci_half_width(values)
        output["spearman_ci_lower"] = output["spearman_mean"] - output["spearman_95ci"]
        output["spearman_ci_upper"] = output["spearman_mean"] + output["spearman_95ci"]
        summary.append(output)
    return summary


def first_threshold(summary: list[dict], conservative: bool = False):
    metric = "spearman_ci_lower" if conservative else "spearman_mean"
    for row in summary:
        if float(row[metric]) >= 0.95:
            return int(row["K"])
    return "not_reached"


def plot_summary(summary: list[dict], output: Path) -> None:
    if plt is None or not summary:
        return
    k_values = np.asarray([row["K"] for row in summary], dtype=int)
    means = np.asarray([row["spearman_mean"] for row in summary], dtype=float)
    errors = np.asarray([row["spearman_95ci"] for row in summary], dtype=float)
    fig, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    axis.errorbar(k_values, means, yerr=errors, marker="o", capsize=3, color="#0072B2")
    axis.axhline(0.95, color="#666666", linestyle="--", linewidth=1)
    mean_threshold = first_threshold(summary)
    conservative_threshold = first_threshold(summary, conservative=True)
    if isinstance(mean_threshold, int):
        axis.axvline(
            mean_threshold,
            color="#D55E00",
            linestyle=":",
            linewidth=1.2,
            label=f"Mean threshold: K={mean_threshold}",
        )
    if isinstance(conservative_threshold, int):
        axis.axvline(
            conservative_threshold,
            color="#009E73",
            linestyle="-.",
            linewidth=1.2,
            label=f"95% CI threshold: K={conservative_threshold}",
        )
    axis.set_xticks(np.arange(400, int(k_values.max()) + 1, 400))
    axis.set_xlabel("Projection dimension K")
    axis.set_ylabel(r"Spearman $\rho$")
    axis.set_title("WS projection-dimension threshold")
    axis.grid(alpha=0.25)
    if isinstance(mean_threshold, int) or isinstance(conservative_threshold, int):
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _write_checkpoint(output_dir: Path, raw_rows: list[dict], k_values: tuple[int, ...]):
    summary = summarize(raw_rows, k_values)
    write_csv(output_dir / "M3_WS_K_threshold_raw.csv", raw_rows)
    write_csv(output_dir / "M3_WS_K_threshold_summary.csv", summary)
    return summary


def run(args) -> None:
    if args.graph_repeats < 1:
        raise ValueError("--graph-repeats must be positive.")
    if args.nodes > args.exact_max_n:
        raise ValueError("Exact STRC is required; increase --exact-max-n or reduce --nodes.")
    k_values = m3._parse_k_values(args.k_values)
    reference_k = max(k_values)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, projection_times = [], []
    experiment_started = time.perf_counter()

    for graph_repeat in range(args.graph_repeats):
        graphs = m3.make_graphs(args.nodes, args.edges, realization=graph_repeat)
        network, graph, graph_kind, graph_seed = next(
            item for item in graphs if item[0] == "WS"
        )
        if network != "WS" or graph_kind != "synthetic":
            raise RuntimeError("The threshold experiment requires a synthetic WS graph.")
        print(
            f"[M3-WS] graph {graph_repeat + 1}/{args.graph_repeats}: "
            f"N={len(graph)}, M={graph.number_of_edges()}, K_max={reference_k}",
            flush=True,
        )
        nodes, _ = m3._ordered_graph(graph)
        laplacian_pseudoinverse = m3._laplacian_pseudoinverse(graph, nodes)
        exact_scores = m3.exact_strc_scores(graph, Lp=laplacian_pseudoinverse)
        context = m3.build_projection_context(graph)
        projection_started = time.perf_counter()
        projected_nodes, full_embedding = m3.projected_embedding(
            graph,
            reference_k,
            seed=9000000 + graph_repeat,
            context=context,
        )
        projection_seconds = time.perf_counter() - projection_started
        projection_times.append(projection_seconds)
        for K in k_values:
            score_started = time.perf_counter()
            embedding = full_embedding[:, :K] * math.sqrt(reference_k / K)
            scores = m3.strc_from_embedding(graph, projected_nodes, embedding)
            rho, tau, overlap = m3.rank_metrics(exact_scores, scores)
            raw_rows.append(
                {
                    "network": "WS",
                    "graph_repeat": graph_repeat,
                    "graph_seed": graph_seed,
                    "N": len(graph),
                    "M": graph.number_of_edges(),
                    "K": K,
                    "projection_seed": 9000000 + graph_repeat,
                    "spearman": rho,
                    "kendall": tau,
                    "top10_overlap": overlap,
                    "score_seconds": time.perf_counter() - score_started,
                    "K_max_projection_seconds": projection_seconds,
                    "reference": "exact",
                }
            )
        _write_checkpoint(output_dir, raw_rows, k_values)
        del laplacian_pseudoinverse, full_embedding, context
        gc.collect()

    summary = _write_checkpoint(output_dir, raw_rows, k_values)
    result = {
        "network": "WS",
        "network_type": "synthetic",
        "N": args.nodes,
        "M": args.edges,
        "graph_repetitions": args.graph_repeats,
        "K_values": list(k_values),
        "nested_projection": True,
        "rank_reference": "exact STRC",
        "projection_system_solver": "grounded sparse LU",
        "solver_rationale": "isolate projection-dimension error from iterative-solver error",
        "confidence_interval_unit": "independent graph realization",
        "mean_rho_threshold": 0.95,
        "first_K_with_mean_rho_at_least_0.95": first_threshold(summary),
        "first_K_with_95ci_lower_bound_at_least_0.95": first_threshold(
            summary, conservative=True
        ),
        "K_max_projection_seconds_mean": float(np.mean(projection_times)),
        "K_max_projection_seconds_95ci": m3._ci_half_width(
            np.asarray(projection_times)
        ),
        "elapsed_seconds": time.perf_counter() - experiment_started,
    }
    (output_dir / "M3_WS_K_threshold.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_summary(summary, output_dir / "M3_WS_K_threshold")
    print(
        "[M3-WS] mean threshold:",
        result["first_K_with_mean_rho_at_least_0.95"],
        "| conservative threshold:",
        result["first_K_with_95ci_lower_bound_at_least_0.95"],
        flush=True,
    )
    print(f"[M3-WS] Wrote complete results to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=str(SCRIPT_DIR.parent / "results" / "m3_ws_threshold")
    )
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--edges", type=int, default=400)
    parser.add_argument("--exact-max-n", type=int, default=300)
    parser.add_argument("--graph-repeats", type=int, default=50)
    parser.add_argument("--k-values", default=",".join(map(str, DEFAULT_K_VALUES)))
    run(parser.parse_args())
