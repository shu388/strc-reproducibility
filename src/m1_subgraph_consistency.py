"""Exact M1 consistency analysis on connected real-network subgraphs.

For each of the six real-world networks, this script selects the same kind of
connected induced subgraph used for the M2 validation and computes exact
leave-one-node STRC together with the three M1 global-connectivity indices:
effective-resistance network efficiency (ER), Kirchhoff index (KI), and
average effective resistance (AER).  It then reports node-level Spearman
correlations across the four rankings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from advanced_baselines import effective_resistance_baselines
from m2_spectral_correlations import (
    NETWORK_FILES,
    connected_induced_sample,
    exact_strc_scores,
    load_network,
)


METHODS = ("STRC", "ER", "KI", "AER")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    data_dir: Path,
    output_dir: Path,
    validation_nodes: int = 80,
    seed: int = 20260815,
    network_files: tuple[str, ...] = NETWORK_FILES,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    wide_rows: list[dict] = []

    for network_index, filename in enumerate(network_files):
        path = data_dir / filename
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        graph = load_network(path)
        subgraph = connected_induced_sample(
            graph, validation_nodes, seed + 1000 + network_index
        )
        print(
            f"[{path.stem}] exact subgraph N={subgraph.number_of_nodes()}, "
            f"M={subgraph.number_of_edges()}"
        )

        scores = {"STRC": exact_strc_scores(subgraph)}
        scores.update(effective_resistance_baselines(subgraph))
        nodes = list(subgraph)
        node_rows = []
        for node in nodes:
            node_rows.append(
                {"node": node, **{method: scores[method][node] for method in METHODS}}
            )
        _write_csv(
            output_dir / f"m1_subgraph_scores_{path.stem}.csv",
            node_rows,
            ["node", *METHODS],
        )

        matrix = np.array(
            [[scores[method][node] for method in METHODS] for node in nodes],
            dtype=float,
        )
        wide = {
            "network": path.stem,
            "N": subgraph.number_of_nodes(),
            "M": subgraph.number_of_edges(),
        }
        for left_index, left in enumerate(METHODS):
            for right in METHODS[left_index + 1 :]:
                right_index = METHODS.index(right)
                rho, pvalue = spearmanr(matrix[:, left_index], matrix[:, right_index])
                wide[f"{left}_vs_{right}_rho"] = float(rho)
                wide[f"{left}_vs_{right}_pvalue"] = float(pvalue)
                rows.append(
                    {
                        "network": path.stem,
                        "N": subgraph.number_of_nodes(),
                        "M": subgraph.number_of_edges(),
                        "metric_a": left,
                        "metric_b": right,
                        "spearman_rho": float(rho),
                        "pvalue": float(pvalue),
                    }
                )
        wide_rows.append(wide)

    _write_csv(
        output_dir / "m1_subgraph_spearman.csv",
        rows,
        ["network", "N", "M", "metric_a", "metric_b", "spearman_rho", "pvalue"],
    )
    wide_fields = ["network", "N", "M"]
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1 :]:
            wide_fields.extend([f"{left}_vs_{right}_rho", f"{left}_vs_{right}_pvalue"])
    _write_csv(output_dir / "m1_subgraph_spearman_wide.csv", wide_rows, wide_fields)
    metadata = {
        "experiment": "Reviewer 5 M1 consistency with global connectivity indices",
        "subgraph_nodes": validation_nodes,
        "seed": seed,
        "methods": list(METHODS),
        "networks": [row["network"] for row in wide_rows],
        "definitions": {
            "STRC": "exact relative reduction in spanning-tree count after node deletion",
            "ER": "exact leave-one-node change in effective-resistance network efficiency",
            "KI": "exact leave-one-node change in Kirchhoff index",
            "AER": "negative average effective resistance from the node to all other nodes",
        },
    }
    (output_dir / "m1_subgraph_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=script_dir.parent / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir.parent / "results" / "m1_subgraph"
    )
    parser.add_argument("--subgraph-nodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--networks", nargs="*", default=list(NETWORK_FILES))
    args = parser.parse_args()
    if args.subgraph_nodes < 3:
        parser.error("--subgraph-nodes must be at least 3")
    run_experiment(
        args.data_dir,
        args.output_dir,
        args.subgraph_nodes,
        args.seed,
        tuple(args.networks),
    )


if __name__ == "__main__":
    main()
