"""Weighted graph construction (paper §3.2.2).

A weighted graph ``G = (V, E, S)`` is built from the pairwise similarity matrix.
Two users are connected iff their aggregated similarity score exceeds a
threshold ``tau`` (default 0.70); the edge weight is the aggregated similarity
score (Eq. 5).
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

DEFAULT_THRESHOLD: float = 0.70


def build_graph(
    similarity_matrix: np.ndarray,
    node_ids: Sequence[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> nx.Graph:
    """Build an undirected weighted graph from a similarity matrix.

    Parameters
    ----------
    similarity_matrix:
        Symmetric ``N x N`` matrix of aggregated similarity scores.
    node_ids:
        Optional node labels (defaults to integer indices ``0..N-1``).
    threshold:
        ``tau``: an edge is added only when ``score > tau`` (Eq. 5).

    Returns
    -------
    networkx.Graph
        Nodes carry no attributes; edges carry a ``weight`` attribute equal to
        the aggregated similarity score.
    """
    matrix = np.asarray(similarity_matrix, dtype=np.float64)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("similarity_matrix must be square")
    if node_ids is None:
        node_ids = list(range(n))
    if len(node_ids) != n:
        raise ValueError("node_ids length must match matrix dimension")

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(matrix[i, j])
            if score > threshold:
                graph.add_edge(node_ids[i], node_ids[j], weight=score)
    return graph


__all__ = ["build_graph", "DEFAULT_THRESHOLD"]
