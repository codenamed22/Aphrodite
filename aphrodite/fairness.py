"""Fairness / congestion rerankers — FairRec and Nash Social Welfare (Tier-2 ``exp-fair``).

Two post-processing rerankers that redistribute *producer exposure* over a
relevance matrix. In our symmetric matchmaking setting every user is at once a
**customer** (receives a top-``k`` list) and a **producer/item** (may appear in
other users' lists). Popular users hog exposure — the *congestion* problem. These
rerankers are congestion **ablations** to compare against the TU/IPFP market
reranker in :mod:`aphrodite.matching`.

* **FairRec** (Patro, Biswas, Ganguly, Gummadi & Chakraborty, *"FairRec:
  Two-Sided Fairness for Personalized Recommendations in Two-Sided Platforms"*,
  WWW 2020). A two-phase greedy that first guarantees every producer a minimum
  exposure (a Maximin-Share style floor ``l = floor(alpha * k)``) while keeping
  the customer allocation Envy-Free-up-to-one-item (EF1) via a rotating
  round-robin, then greedily fills the remaining slots by relevance.

* **Nash Social Welfare (NSW)** reranking. Maximises the *product* of producer
  exposures — equivalently the sum of ``log`` exposure — via a greedy
  round-robin whose per-item marginal gain ``g(j) = log(e_j + 2) - log(e_j + 1)``
  is large for under-exposed items. This naturally spreads exposure to the long
  tail, the defining behaviour of a congestion baseline.

Design: a **pure core** (:func:`fairrec`, :func:`nsw_rerank`) operating on plain
``numpy`` arrays plus a feasibility ``mask`` — fully unit-testable with no
embeddings — and **thin wrappers** (:func:`relevance_matrix`,
:func:`fairrec_recommend_all`, :func:`nsw_recommend_all`, :class:`FairRecReranker`,
:class:`NSWReranker`) that extract a relevance matrix from a fitted
:class:`~aphrodite.reciprocal.ReciprocalRecommender` and return an id-keyed
recommendation dict for benchmark parity with the TU/IPFP reranker.
"""

from __future__ import annotations

import math
from typing import Hashable

import numpy as np

from .ppr import rank_matches
from .reciprocal import ReciprocalRecommender


# -- pure-core helpers ---------------------------------------------------------
def _validate_core_inputs(R: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Coerce ``R``/``mask`` to arrays, sanity-check shapes, force a False diagonal."""
    R = np.asarray(R, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be a square 2-D matrix, got shape {R.shape!r}")
    if mask.shape != R.shape:
        raise ValueError(
            f"mask shape {mask.shape!r} does not match R shape {R.shape!r}"
        )
    mask = mask.copy()
    np.fill_diagonal(mask, False)  # a user is never recommended to itself
    return R, mask, R.shape[0]


def _best_feasible(
    R_row: np.ndarray,
    mask_row: np.ndarray,
    chosen: set[int],
    exposure: np.ndarray | None = None,
    floor: int | None = None,
) -> int:
    """Index of the highest-relevance feasible, not-yet-chosen item (``-1`` if none).

    Iterates candidates in ascending index order and keeps a strict ``>`` on
    relevance, so ties resolve to the *smallest* index (stable, deterministic).
    When ``exposure``/``floor`` are given, only items whose current exposure is
    below ``floor`` qualify.
    """
    best_j = -1
    best_r = -np.inf
    n = R_row.shape[0]
    for j in range(n):
        if not mask_row[j] or j in chosen:
            continue
        if floor is not None and exposure is not None and exposure[j] >= floor:
            continue
        rj = R_row[j]
        if rj > best_r:
            best_r = rj
            best_j = j
    return best_j


def _norm_relevance(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max scale the *feasible* entries of ``R`` to ``[0, 1]`` (computed once).

    Infeasible entries are left at ``0.0`` — they are never selected because the
    ``mask`` gates every choice. A degenerate range (all feasible values equal)
    maps every feasible entry to ``0.0``.
    """
    R = np.asarray(R, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(R, dtype=np.float64)
    vals = R[mask]
    if vals.size == 0:
        return out
    rmin = float(vals.min())
    rmax = float(vals.max())
    if rmax > rmin:
        out[mask] = (R[mask] - rmin) / (rmax - rmin)
    return out


# -- pure core: FairRec --------------------------------------------------------
def fairrec(
    R: np.ndarray,
    mask: np.ndarray,
    k: int,
    alpha: float = 0.5,
) -> list[list[int]]:
    """FairRec two-phase greedy allocation (Patro et al., WWW 2020).

    Parameters
    ----------
    R:
        ``(N, N)`` relevance matrix; ``R[i, j]`` is the relevance of item ``j``
        to customer ``i``. Diagonal / infeasible entries are ignored via ``mask``.
    mask:
        ``(N, N)`` boolean feasibility matrix (``True`` = ``i`` may be recommended
        ``j``). The diagonal is forced ``False``.
    k:
        Number of items to allocate to each customer.
    alpha:
        Fraction of ``k`` used for the exposure floor. Because ``#customers ==
        #producers == N`` here, the guaranteed minimum exposure is
        ``l = floor(alpha * k)``.

    Returns
    -------
    list[list[int]]
        For each customer ``i`` the ordered list of allocated item indices —
        Phase-1 (fairness) picks first, then Phase-2 (relevance) picks.

    Notes
    -----
    * **Phase 1** guarantees the exposure floor with EF1 via a *rotating*
      round-robin: in round ``r`` customers are visited in order
      ``[(r + t) % N for t in range(N)]`` so none is systematically favoured.
      Each visited customer takes their highest-relevance feasible item whose
      exposure is still below ``l``; if none qualifies a relevance-only fallback
      keeps them progressing.
    * **Phase 2** fills every remaining slot greedily by relevance.
    * An infeasible (``mask`` ``False``) pair is never selected and no item is
      duplicated for a customer. Robust when ``k`` exceeds a customer's feasible
      candidate count (they simply receive all feasible items, in relevance
      order).
    """
    R, mask, n = _validate_core_inputs(R, mask)
    k = int(k)
    if n == 0 or k <= 0:
        return [[] for _ in range(n)]

    floor = max(0, math.floor(float(alpha) * k))
    exposure = np.zeros(n, dtype=np.int64)
    allocation: list[list[int]] = [[] for _ in range(n)]
    chosen: list[set[int]] = [set() for _ in range(n)]

    # -- Phase 1: guarantee minimum exposure (round-robin, rotating order) ----
    for r in range(floor):
        order = [(r + t) % n for t in range(n)]
        for i in order:
            if len(allocation[i]) >= k:
                continue
            j = _best_feasible(R[i], mask[i], chosen[i], exposure, floor)
            if j < 0:  # nothing left under the floor -> relevance-only fallback
                j = _best_feasible(R[i], mask[i], chosen[i])
            if j < 0:  # no feasible item remains for this customer
                continue
            allocation[i].append(j)
            chosen[i].add(j)
            exposure[j] += 1

    # -- Phase 2: fill the rest greedily by relevance -------------------------
    for i in range(n):
        while len(allocation[i]) < k:
            j = _best_feasible(R[i], mask[i], chosen[i])
            if j < 0:
                break
            allocation[i].append(j)
            chosen[i].add(j)
            exposure[j] += 1

    return allocation


# -- pure core: Nash Social Welfare -------------------------------------------
def nsw_rerank(
    R: np.ndarray,
    mask: np.ndarray,
    k: int,
    relevance_weight: float = 1.0,
) -> list[list[int]]:
    """Producer-exposure Nash-Social-Welfare reranking via greedy round-robin.

    Maximises (greedily) the product of producer exposures — i.e. the sum of
    ``log`` exposure — so under-exposed items receive a fairness bonus that
    spreads attention to the long tail (the congestion-ablation baseline).

    The marginal fairness gain of assigning item ``j`` is
    ``g(j) = log(e_j + 2) - log(e_j + 1)`` (large when ``j`` is under-exposed).
    A candidate ``(i, j)`` is scored ``relevance_weight * R̂[i, j] + g(j)`` where
    ``R̂`` is :func:`_norm_relevance` (feasible entries min-max scaled to
    ``[0, 1]`` once up front) so the two terms are comparable. Customers are
    processed round-robin for ``k`` rounds using the same rotating order as
    :func:`fairrec`; each picks the feasible, not-already-chosen item maximising
    the combined score (ties broken by raw ``R`` then smallest index).

    Returns, per customer ``i``, the ordered list of allocated item indices.
    Infeasible pairs are never selected, items are never duplicated per customer,
    and exhausted candidates are handled gracefully.
    """
    R, mask, n = _validate_core_inputs(R, mask)
    k = int(k)
    if n == 0 or k <= 0:
        return [[] for _ in range(n)]

    rw = float(relevance_weight)
    Rn = _norm_relevance(R, mask)
    exposure = np.zeros(n, dtype=np.int64)
    allocation: list[list[int]] = [[] for _ in range(n)]
    chosen: list[set[int]] = [set() for _ in range(n)]

    for r in range(k):
        order = [(r + t) % n for t in range(n)]
        for i in order:
            best_j = -1
            best_key: tuple[float, float] | None = None
            for j in range(n):
                if not mask[i, j] or j in chosen[i]:
                    continue
                gain = math.log(exposure[j] + 2.0) - math.log(exposure[j] + 1.0)
                score = rw * Rn[i, j] + gain
                key = (score, float(R[i, j]))
                if best_key is None or key > best_key:
                    best_key = key
                    best_j = j
            if best_j < 0:  # no feasible candidate remains
                continue
            allocation[i].append(best_j)
            chosen[i].add(best_j)
            exposure[best_j] += 1

    return allocation


# -- wrappers over a fitted recommender ---------------------------------------
def relevance_matrix(
    recommender: ReciprocalRecommender,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Materialise the relevance matrix ``R``, id list and feasibility ``mask``.

    ``ids = list(recommender.node_ids_)``. For ``i != j`` a pair is feasible when
    ``ids[j]`` is *not* in ``recommender._incompatible_ids(ids[i])``;
    ``mask[i, j]`` records feasibility (diagonal ``False``) and ``R[i, j]`` is the
    reciprocal :meth:`~aphrodite.reciprocal.ReciprocalRecommender.score_pair`
    for feasible pairs (``0.0`` otherwise — the ``mask`` gates selection). Each
    customer's incompatible set is computed once.

    Returns ``(R, ids, mask)``.
    """
    ids = list(recommender.node_ids_)
    n = len(ids)
    R = np.zeros((n, n), dtype=np.float64)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        incompatible = recommender._incompatible_ids(ids[i])
        a_id = ids[i]
        for j in range(n):
            if i == j:
                continue
            if ids[j] in incompatible:
                continue
            mask[i, j] = True
            R[i, j] = float(recommender.score_pair(a_id, ids[j]))
    return R, ids, mask


def _idx_lists_to_id_dict(
    idx_lists: list[list[int]], ids: list
) -> dict[str, list]:
    return {ids[i]: [ids[j] for j in idx_lists[i]] for i in range(len(ids))}


def fairrec_recommend_all(
    recommender: ReciprocalRecommender,
    k: int = 10,
    alpha: float = 0.5,
) -> dict[str, list]:
    """FairRec top-``k`` recommendations for every user of a fitted recommender.

    Builds ``(R, ids, mask)`` via :func:`relevance_matrix`, runs :func:`fairrec`
    and maps the index allocation back to user ids. Returns ``{user_id: [ids]}``
    with lists of length ``<= k`` that exclude self and gender-incompatible users.
    """
    R, ids, mask = relevance_matrix(recommender)
    idx_lists = fairrec(R, mask, k, alpha=alpha)
    return _idx_lists_to_id_dict(idx_lists, ids)


def nsw_recommend_all(
    recommender: ReciprocalRecommender,
    k: int = 10,
    relevance_weight: float = 1.0,
) -> dict[str, list]:
    """NSW top-``k`` recommendations for every user of a fitted recommender.

    Analogous to :func:`fairrec_recommend_all` but uses :func:`nsw_rerank`.
    """
    R, ids, mask = relevance_matrix(recommender)
    idx_lists = nsw_rerank(R, mask, k, relevance_weight=relevance_weight)
    return _idx_lists_to_id_dict(idx_lists, ids)


class FairRecReranker:
    """Thin FairRec wrapper around a fitted reciprocal recommender.

    Reranks a fitted :class:`~aphrodite.reciprocal.ReciprocalRecommender` with the
    FairRec two-sided fairness allocation (WWW 2020) so every producer receives a
    guaranteed minimum exposure — a congestion ablation vs. the TU/IPFP reranker.

    Parameters
    ----------
    recommender:
        A **fitted** ``ReciprocalRecommender``.
    alpha:
        Exposure-floor fraction (floor ``l = floor(alpha * k)``).
    """

    def __init__(
        self, recommender: ReciprocalRecommender, alpha: float = 0.5
    ) -> None:
        self.recommender = recommender
        self.alpha = float(alpha)
        self.node_ids_ = recommender.node_ids_

    def recommend_all(self, k: int = 10) -> dict[str, list]:
        """FairRec top-``k`` recommendations for every user."""
        return fairrec_recommend_all(self.recommender, k=k, alpha=self.alpha)

    def recommend(self, target_id: str, k: int = 10) -> list:
        """FairRec top-``k`` recommendations for a single user.

        FairRec allocates jointly over the whole market, so this computes the
        full solution and returns ``target_id``'s row.
        """
        if target_id not in self.recommender.profiles_by_id_:
            raise KeyError(f"Unknown target user_id: {target_id!r}")
        return self.recommend_all(k=k)[target_id]


class NSWReranker:
    """Thin Nash-Social-Welfare wrapper around a fitted reciprocal recommender.

    Reranks a fitted :class:`~aphrodite.reciprocal.ReciprocalRecommender` to
    maximise the product of producer exposures, spreading attention to the long
    tail — a congestion ablation vs. the TU/IPFP reranker.

    Parameters
    ----------
    recommender:
        A **fitted** ``ReciprocalRecommender``.
    relevance_weight:
        Weight on the (normalised) relevance term relative to the fairness bonus.
    """

    def __init__(
        self, recommender: ReciprocalRecommender, relevance_weight: float = 1.0
    ) -> None:
        self.recommender = recommender
        self.relevance_weight = float(relevance_weight)
        self.node_ids_ = recommender.node_ids_

    def recommend_all(self, k: int = 10) -> dict[str, list]:
        """NSW top-``k`` recommendations for every user."""
        return nsw_recommend_all(
            self.recommender, k=k, relevance_weight=self.relevance_weight
        )

    def recommend(self, target_id: str, k: int = 10) -> list:
        """NSW top-``k`` recommendations for a single user.

        The allocation is a joint property of the whole market, so this computes
        the full solution and returns ``target_id``'s row.
        """
        if target_id not in self.recommender.profiles_by_id_:
            raise KeyError(f"Unknown target user_id: {target_id!r}")
        return self.recommend_all(k=k)[target_id]


__all__ = [
    "fairrec",
    "nsw_rerank",
    "relevance_matrix",
    "fairrec_recommend_all",
    "nsw_recommend_all",
    "FairRecReranker",
    "NSWReranker",
]
