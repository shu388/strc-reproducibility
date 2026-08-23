"""Compute the real-network statistics reported in Table 1.

All inputs are converted to simple undirected graphs, self-loops are removed,
and statistics are evaluated on the largest connected component.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import networkx as nx
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "results" / "dataset_statistics.csv"

DATASETS = (
    ("CA-HepPh", "Collaboration", "CA-HepPh.txt", "https://snap.stanford.edu/data/ca-HepPh.html"),
    ("Email-Enron", "Communication", "Email-Enron.txt", "https://snap.stanford.edu/data/email-Enron.html"),
    ("cit-HepTh", "Citation", "cit-HepTh.txt", "https://snap.stanford.edu/data/cit-HepTh.html"),
    ("bio-dmela", "Biological", "bio-dmela.txt", "https://networkrepository.com/bio-dmela.php"),
    ("Wiki-Vote", "Social", "Wiki-Vote.txt", "https://snap.stanford.edu/data/wiki-Vote.html"),
    ("tech-as-caida", "Infrastructure", "tech-as-caida.txt", "https://networkrepository.com/tech-as-caida.php"),
)


def load_lcc(path: Path) -> nx.Graph:
    graph = nx.Graph()
    with path.open("r", encoding="utf-8-sig") as edge_file:
        for line in edge_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 2 and fields[0] != fields[1]:
                graph.add_edge(fields[0], fields[1])

    if graph.number_of_nodes() == 0:
        raise ValueError(f"No edges were read from {path}")
    if not nx.is_connected(graph):
        nodes = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(nodes).copy()
    return graph


def compute_statistics(
    name: str, category: str, source_url: str, graph: nx.Graph
) -> dict[str, object]:
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=float)
    mean_degree = float(degrees.mean())
    mean_square_degree = float(np.square(degrees).mean())
    return {
        "dataset": name,
        "category": category,
        "source_url": source_url,
        "N": graph.number_of_nodes(),
        "M": graph.number_of_edges(),
        "mean_degree": mean_degree,
        "clustering": nx.average_clustering(graph),
        "beta_star": mean_degree / mean_square_degree,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = []
    for name, category, filename, source_url in DATASETS:
        graph = load_lcc(args.data_dir / filename)
        row = compute_statistics(name, category, source_url, graph)
        rows.append(row)
        print(
            f"{name:14s} {category:14s} N={row['N']:6d} M={row['M']:7d} "
            f"mean_degree={row['mean_degree']:.2f} "
            f"C={row['clustering']:.3f} beta*={row['beta_star']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
