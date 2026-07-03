"""Threshold (``tau``) calibration for the PPR matchmaker (paper §3.2.2).

The edge threshold ``tau`` decides which user pairs become graph edges: an edge
is added only when the aggregated similarity ``score > tau`` (Eq. 5). The paper's
default (``tau = 0.70``) is calibrated for tightly clustered data. On a corpus
with a *different* similarity distribution the same value can badly misfire:

* **Too high** for a *diverse* corpus (many unrelated users) → almost no edges,
  a disconnected graph, and Personalized PageRank with nothing to propagate over.
* **Too low** for a *homogeneous* corpus (everybody shares vocabulary) → a
  near-complete graph where every user looks similar to every other, so the
  ranking loses all discriminative power.

This module measures the actual similarity distribution and the resulting graph
health at candidate thresholds, then recommends a ``tau`` in one of two modes:

* :func:`suggest_threshold` — **unsupervised** (no ground truth). Picks the most
  *discriminative* threshold (highest ``tau``) that still keeps the graph
  connected and within a healthy edge-density band.
* :func:`tune_threshold_supervised` — **supervised** (ground truth available).
  Sweeps ``tau``, runs the full PPR matchmaker at each value, and picks the
  ``tau`` that maximises MAP@k (paper §4.2). A single embedding pass is reused
  across all candidates, so only the (cheap) graph build + PPR is repeated.

Diagnostics reported per candidate threshold (:func:`graph_health`):

* ``density`` — fraction of possible undirected edges that exist.
* ``mean_degree`` — average number of neighbours per user.
* ``isolated_fraction`` — share of users with no edges at all (dangling nodes
  PPR can never reach from anyone else).
* ``largest_component_fraction`` — size of the biggest connected component,
  relative to N; a proxy for how globally reachable the graph is.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from .embeddings import EmbeddingBackend
from .graph import DEFAULT_THRESHOLD, build_graph
from .metrics import mean_average_precision_at_k
from .ppr import DEFAULT_DAMPING
from .preprocessing import Preprocessor
from .profiles import ATTRIBUTES, UserProfile

# A threshold above the maximum possible cosine similarity, used to fit the
# embedding pipeline without paying for a (possibly huge) edge list we discard.
_NO_EDGE_THRESHOLD: float = 2.0

# Default candidate thresholds to sweep. Covers the "diverse" regime (~0.30) up
# to well past the paper default (0.70).
DEFAULT_TAU_GRID: tuple[float, ...] = tuple(
    round(0.30 + 0.05 * i, 2) for i in range(11)  # 0.30, 0.35, ..., 0.80
)

DEFAULT_PERCENTILES: tuple[float, ...] = (1, 5, 25, 50, 75, 90, 95, 99)


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the off-diagonal upper-triangle values (each pair once)."""
    m = np.asarray(matrix, dtype=np.float64)
    n = m.shape[0]
    if m.shape != (n, n):
        raise ValueError("similarity_matrix must be square")
    iu = np.triu_indices(n, k=1)
    return m[iu]


def similarity_distribution(
    matrix: np.ndarray, percentiles: Sequence[float] = DEFAULT_PERCENTILES
) -> dict[str, float]:
    """Summarise the pairwise-similarity distribution of a corpus.

    Returns ``count``/``min``/``max``/``mean``/``std`` plus a ``p{q}`` entry for
    each requested percentile ``q`` (e.g. ``"p50"``). An empty (``N < 2``) matrix
    yields zeros so callers need not special-case it.
    """
    values = _upper_triangle(matrix)
    summary: dict[str, float] = {"count": float(values.size)}
    if values.size == 0:
        summary.update({"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0})
        for q in percentiles:
            summary[f"p{int(q)}"] = 0.0
        return summary
    summary["min"] = float(values.min())
    summary["max"] = float(values.max())
    summary["mean"] = float(values.mean())
    summary["std"] = float(values.std())
    for q, val in zip(percentiles, np.percentile(values, percentiles)):
        summary[f"p{int(q)}"] = float(val)
    return summary


def graph_health(
    matrix: np.ndarray,
    node_ids: Sequence[str] | None,
    tau: float,
) -> dict[str, float]:
    """Build the graph at ``tau`` and report connectivity diagnostics.

    Returns a dict with ``tau``, ``edges``, ``possible_edges``, ``density``,
    ``mean_degree``, ``isolated``, ``isolated_fraction``, ``n_components`` and
    ``largest_component_fraction``.
    """
    graph = build_graph(matrix, node_ids=node_ids, threshold=tau)
    n = graph.number_of_nodes()
    edges = graph.number_of_edges()
    possible = n * (n - 1) // 2
    density = edges / possible if possible else 0.0
    mean_degree = (2.0 * edges / n) if n else 0.0
    isolated = sum(1 for _, deg in graph.degree() if deg == 0)
    isolated_fraction = isolated / n if n else 0.0
    if n:
        components = list(nx.connected_components(graph))
        n_components = len(components)
        largest = max((len(c) for c in components), default=0)
        largest_fraction = largest / n
    else:
        n_components = 0
        largest_fraction = 0.0
    return {
        "tau": float(tau),
        "edges": int(edges),
        "possible_edges": int(possible),
        "density": float(density),
        "mean_degree": float(mean_degree),
        "isolated": int(isolated),
        "isolated_fraction": float(isolated_fraction),
        "n_components": int(n_components),
        "largest_component_fraction": float(largest_fraction),
    }


def scan_thresholds(
    matrix: np.ndarray,
    node_ids: Sequence[str] | None = None,
    taus: Sequence[float] = DEFAULT_TAU_GRID,
) -> list[dict[str, float]]:
    """Return :func:`graph_health` for every candidate ``tau`` (ascending)."""
    return [graph_health(matrix, node_ids, tau) for tau in sorted(taus)]


def suggest_threshold(
    matrix: np.ndarray,
    node_ids: Sequence[str] | None = None,
    taus: Sequence[float] = DEFAULT_TAU_GRID,
    min_density: float = 0.05,
    max_density: float = 0.25,
    max_isolated_fraction: float = 0.10,
) -> dict[str, object]:
    """Recommend an edge threshold ``tau`` with no ground truth.

    Strategy — prefer the **most discriminative** graph that is still healthy:

    1. A ``tau`` is *healthy* when its ``density`` lies in
       ``[min_density, max_density]`` and ``isolated_fraction`` stays under
       ``max_isolated_fraction``. Among healthy candidates return the **largest**
       ``tau`` — the highest threshold that still yields a usable, connected graph.
    2. If none is healthy but some are not over-dense, the corpus is too diverse
       to reach the band: return the **most-connected** such graph (fewest
       isolated users) and flag that a threshold-free reciprocal method may suit
       the data better.
    3. If every candidate is over-dense (a very homogeneous corpus), return the
       largest ``tau`` (the most aggressive pruning available on the grid).

    Returns ``{recommended_tau, reason, distribution, table}`` where ``table`` is
    the full :func:`scan_thresholds` output for transparency.
    """
    table = scan_thresholds(matrix, node_ids, taus)
    distribution = similarity_distribution(matrix)

    def _result(tau: float, reason: str) -> dict[str, object]:
        return {
            "recommended_tau": float(tau),
            "reason": reason,
            "distribution": distribution,
            "table": table,
        }

    if not table:
        return _result(DEFAULT_THRESHOLD, "empty corpus; kept paper default tau=0.70")

    healthy = [
        r
        for r in table
        if min_density <= r["density"] <= max_density
        and r["isolated_fraction"] <= max_isolated_fraction
    ]
    if healthy:
        best = max(healthy, key=lambda r: r["tau"])
        return _result(
            best["tau"],
            f"highest tau keeping density in [{min_density:g}, {max_density:g}] "
            f"with <={max_isolated_fraction:g} isolated "
            f"(density={best['density']:.3f}, isolated={best['isolated_fraction']:.3f})",
        )

    # No threshold hits the healthy band. Two degenerate regimes remain.
    not_over_dense = [r for r in table if r["density"] <= max_density]
    if not_over_dense:
        # Corpus is too sparse/diverse to reach the band at these thresholds:
        # take the most-connected graph available (fewest isolated users).
        best = min(
            not_over_dense,
            key=lambda r: (r["isolated_fraction"], -r["density"], r["tau"]),
        )
        if best["density"] >= min_density:
            reason = (
                "no tau met every criterion; picked the most-connected graph "
                f"within the density cap (tau={best['tau']:.2f}, "
                f"density={best['density']:.3f}, "
                f"isolated={best['isolated_fraction']:.3f})"
            )
        else:
            reason = (
                f"corpus too diverse for PPR: even tau={best['tau']:.2f} gives "
                f"density {best['density']:.3f} (< {min_density:g}) with "
                f"{best['isolated_fraction']:.0%} isolated users — lower tau further "
                "or prefer a threshold-free reciprocal method (recon/multi_interest)"
            )
        return _result(best["tau"], reason)

    # Every threshold is over-dense: most aggressive pruning available.
    best = max(table, key=lambda r: r["tau"])
    return _result(
        best["tau"],
        "corpus very homogeneous (all thresholds over-dense); picked the highest "
        f"tau on the grid (density={best['density']:.3f})",
    )


def compute_similarity_matrix(
    profiles: Sequence[UserProfile],
    backend: EmbeddingBackend | None = None,
    preprocessor: Preprocessor | None = None,
    attributes: Sequence[str] = ATTRIBUTES,
    skip_empty: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Embed ``profiles`` once and return ``(similarity_matrix, node_ids)``.

    Thin wrapper over :class:`~aphrodite.matchmaker.MatchmakingAlgorithm` that
    runs stages 1-2 (embedding + pairwise similarity) without paying for a graph
    build we would only discard. Use the returned matrix with
    :func:`suggest_threshold` / :func:`scan_thresholds`.
    """
    from .matchmaker import MatchmakingAlgorithm

    algo = MatchmakingAlgorithm(
        backend=backend,
        preprocessor=preprocessor,
        threshold=_NO_EDGE_THRESHOLD,
        attributes=attributes,
        skip_empty=skip_empty,
    )
    algo.fit(profiles)
    assert algo.similarity_matrix_ is not None  # populated by fit()
    return algo.similarity_matrix_, list(algo.node_ids_)


def _has_gender(profiles: Sequence[UserProfile]) -> bool:
    return any(getattr(p, "gender", "") for p in profiles)


def tune_threshold_supervised(
    profiles: Sequence[UserProfile],
    ground_truth: Mapping[str, set[str]],
    taus: Sequence[float] = DEFAULT_TAU_GRID,
    backend: EmbeddingBackend | None = None,
    preprocessor: Preprocessor | None = None,
    k: int = 10,
    damping: float = DEFAULT_DAMPING,
    attributes: Sequence[str] = ATTRIBUTES,
    skip_empty: bool = False,
    apply_gender_filter: bool | None = None,
) -> dict[str, object]:
    """Pick the ``tau`` that maximises MAP@k against known ground truth.

    Embeds the corpus **once**, then for each candidate ``tau`` rebuilds only the
    graph and re-runs Personalized PageRank — the expensive embedding step is not
    repeated. ``apply_gender_filter`` defaults to ``True`` iff any profile carries
    a gender (otherwise the filter is a no-op).

    Returns ``{recommended_tau, best_map, k, metric, table}`` where ``table`` is a
    list of ``{tau, map, health...}`` rows, one per candidate.
    """
    from .matchmaker import MatchmakingAlgorithm

    if apply_gender_filter is None:
        apply_gender_filter = _has_gender(profiles)

    algo = MatchmakingAlgorithm(
        backend=backend,
        preprocessor=preprocessor,
        threshold=_NO_EDGE_THRESHOLD,
        damping=damping,
        attributes=attributes,
        skip_empty=skip_empty,
        apply_gender_filter=apply_gender_filter,
    )
    algo.fit(profiles)  # single embedding pass
    matrix = algo.similarity_matrix_
    node_ids = list(algo.node_ids_)
    relevant_sets = [set(ground_truth.get(uid, set())) for uid in node_ids]

    table: list[dict[str, float]] = []
    for tau in sorted(taus):
        algo.threshold = tau
        algo.graph_ = build_graph(matrix, node_ids=node_ids, threshold=tau)
        ranked_lists = [algo.recommend(uid, k=k) for uid in node_ids]
        map_k = mean_average_precision_at_k(ranked_lists, relevant_sets, k)
        health = graph_health(matrix, node_ids, tau)
        health["map"] = float(map_k)
        table.append(health)

    if not table:
        return {
            "recommended_tau": float(DEFAULT_THRESHOLD),
            "best_map": 0.0,
            "k": k,
            "metric": f"MAP@{k}",
            "table": table,
        }

    # Prefer higher MAP; break ties toward the higher (more discriminative) tau.
    best = max(table, key=lambda r: (r["map"], r["tau"]))
    return {
        "recommended_tau": float(best["tau"]),
        "best_map": float(best["map"]),
        "k": k,
        "metric": f"MAP@{k}",
        "table": table,
    }


__all__ = [
    "DEFAULT_TAU_GRID",
    "DEFAULT_PERCENTILES",
    "similarity_distribution",
    "graph_health",
    "scan_thresholds",
    "suggest_threshold",
    "compute_similarity_matrix",
    "tune_threshold_supervised",
]
