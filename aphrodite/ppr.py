"""Personalized PageRank for user matchmaking (paper §3.3).

Implements the iterative update (Eq. 6)::

    r' = (1 - d) * M * r + d * v

where ``M`` is the column-stochastic transition matrix derived from the weighted
graph, ``v`` is the personalization vector centred on the target user, and ``d``
is the damping/restart weight (the paper uses ``d = 0.85`` as the weight on the
teleport-to-target term). The walk is iterated to convergence and users are then
ranked by descending score.

Dangling nodes (no outgoing edges) teleport according to ``v``, the standard
PageRank correction, which keeps ``r`` a proper probability distribution.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

DEFAULT_DAMPING: float = 0.85


def transition_matrix(graph: nx.Graph, nodelist: Sequence[Hashable]) -> np.ndarray:
    """Column-stochastic transition matrix ``M`` for the weighted graph.

    ``M[i, j]`` is the probability of moving to node ``i`` from node ``j``,
    proportional to the edge weight. Dangling columns are left zero here and are
    corrected against the personalization vector inside :func:`personalized_pagerank`.
    """
    n = len(nodelist)
    index = {node: k for k, node in enumerate(nodelist)}
    weights = np.zeros((n, n), dtype=np.float64)
    for u, v, data in graph.edges(data=True):
        if u not in index or v not in index:
            continue
        w = float(data.get("weight", 1.0))
        i, j = index[u], index[v]
        weights[i, j] = w
        weights[j, i] = w  # undirected
    col_sums = weights.sum(axis=0)
    m = np.zeros_like(weights)
    nonzero = col_sums > 0
    m[:, nonzero] = weights[:, nonzero] / col_sums[nonzero]
    return m


def personalized_pagerank(
    graph: nx.Graph,
    target: Hashable,
    damping: float = DEFAULT_DAMPING,
    personalization: Mapping[Hashable, float] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[Hashable, float]:
    """Compute Personalized PageRank scores for every node (Eq. 6).

    Parameters
    ----------
    graph:
        Weighted undirected graph.
    target:
        Target user; the personalization vector is centred here (Step 1).
    damping:
        ``d`` — weight on the teleport-to-personalization term (paper: 0.85).
    personalization:
        Optional custom personalization vector ``v`` (node -> mass). If omitted,
        all mass is placed on ``target``. It is normalised to sum to 1.
    max_iter, tol:
        Power-iteration controls; stops when the L1 change is below ``tol``.

    Returns
    -------
    dict
        Mapping of node -> PageRank score (sums to 1).
    """
    nodelist = list(graph.nodes())
    if target not in graph:
        raise KeyError(f"target {target!r} is not a node in the graph")
    n = len(nodelist)
    if n == 0:
        return {}
    index = {node: k for k, node in enumerate(nodelist)}

    # Personalization vector v (Step 1).
    v = np.zeros(n, dtype=np.float64)
    if personalization is None:
        v[index[target]] = 1.0
    else:
        for node, mass in personalization.items():
            if node in index:
                v[index[node]] = float(mass)
        total = v.sum()
        if total <= 0:
            v[index[target]] = 1.0
        else:
            v = v / total

    m = transition_matrix(graph, nodelist)
    col_sums = m.sum(axis=0)
    dangling = col_sums == 0.0  # nodes with no edges teleport via v

    r = v.copy()
    for _ in range(max_iter):
        # Redistribute dangling mass according to v, then apply Eq. 6.
        dangling_mass = r[dangling].sum()
        walk = m @ r + dangling_mass * v
        r_next = (1.0 - damping) * walk + damping * v
        s = r_next.sum()
        if s > 0:
            r_next = r_next / s
        if np.abs(r_next - r).sum() < tol:
            r = r_next
            break
        r = r_next

    return {node: float(r[index[node]]) for node in nodelist}


def rank_matches(
    scores: Mapping[Hashable, float],
    target: Hashable,
    k: int | None = None,
    exclude: Iterable[Hashable] = (),
) -> list[Hashable]:
    """Rank candidate users by descending PPR score (paper §3.3.2).

    The target itself is always excluded. Ties are broken deterministically by
    node id for reproducibility. Returns the top-``k`` node ids (or all if
    ``k`` is None).
    """
    excluded = set(exclude) | {target}
    candidates = [(node, s) for node, s in scores.items() if node not in excluded]
    candidates.sort(key=lambda item: (-item[1], str(item[0])))
    ranked = [node for node, _ in candidates]
    if k is not None:
        ranked = ranked[:k]
    return ranked


__all__ = [
    "personalized_pagerank",
    "transition_matrix",
    "rank_matches",
    "DEFAULT_DAMPING",
]
