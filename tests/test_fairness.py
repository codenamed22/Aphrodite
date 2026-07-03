"""Tests for the FairRec + Nash-Social-Welfare fairness/congestion rerankers.

The pure core (:func:`fairrec`, :func:`nsw_rerank`) is exercised on synthetic
relevance matrices; the wrappers are checked end-to-end against a real fitted
:class:`~aphrodite.reciprocal.ReciprocalRecommender`. No network access.
"""

import numpy as np

from aphrodite.datasets import generate_dataset
from aphrodite.fairness import (
    FairRecReranker,
    NSWReranker,
    fairrec,
    fairrec_recommend_all,
    nsw_recommend_all,
    nsw_rerank,
    relevance_matrix,
)
from aphrodite.metrics import coverage_at_k, exposure_counts, gini_exposure
from aphrodite.reciprocal import ReciprocalRecommender


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skewed_R(n: int = 8, seed: int = 0) -> np.ndarray:
    """Relevance matrix where EVERY customer most-prefers item 0.

    Item 0 is dominant (relevance 100). The remaining items carry a small,
    fixed pseudo-random component plus a mild decreasing gradient so that a
    plain top-k baseline concentrates exposure on a few low-index items and
    leaves several items with zero exposure.
    """
    rng = np.random.RandomState(seed)
    R = rng.uniform(0.0, 1.0, size=(n, n)) * 0.1
    gradient = np.array([0.0] + [float(n - j) for j in range(1, n)])
    R = R + gradient[None, :]
    R[:, 0] = 100.0
    np.fill_diagonal(R, -1.0)
    return R


def _full_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def _ids(n: int) -> list:
    return [f"u{i}" for i in range(n)]


def _to_id_dict(idx_lists, ids) -> dict:
    return {ids[i]: [ids[j] for j in idx_lists[i]] for i in range(len(ids))}


def _baseline_topk(R: np.ndarray, mask: np.ndarray, k: int) -> list:
    """Pure top-k by relevance (ignores congestion) — the unfair control."""
    n = R.shape[0]
    out = []
    for i in range(n):
        feasible = [j for j in range(n) if mask[i, j]]
        feasible.sort(key=lambda j: (-R[i, j], j))
        out.append(feasible[:k])
    return out


# ---------------------------------------------------------------------------
# FairRec — exposure floor (Maximin-Share style guarantee)
# ---------------------------------------------------------------------------
def test_fairrec_exposure_floor():
    n, k = 8, 4
    R = _skewed_R(n, seed=0)
    mask = _full_mask(n)
    ids = _ids(n)
    floor = 4 // 2  # l = floor(alpha * k) = floor(0.5 * 4) = 2

    idx = fairrec(R, mask, k, alpha=0.5)
    recs = _to_id_dict(idx, ids)
    exposure = exposure_counts(recs, k, ids)

    # Full off-diagonal mask makes the floor achievable for every item.
    assert min(exposure.values()) >= floor

    # Every customer receives exactly k DISTINCT feasible items.
    for i in range(n):
        assert len(idx[i]) == k
        assert len(set(idx[i])) == k
        assert all(mask[i, j] for j in idx[i])


def test_fairrec_beats_baseline_gini():
    n, k = 8, 4
    R = _skewed_R(n, seed=0)
    mask = _full_mask(n)
    ids = _ids(n)

    fair = _to_id_dict(fairrec(R, mask, k, alpha=0.5), ids)
    base = _to_id_dict(_baseline_topk(R, mask, k), ids)

    # The unfair baseline leaves several items unexposed.
    base_exposure = exposure_counts(base, k, ids)
    assert sum(1 for v in base_exposure.values() if v == 0) >= 2

    assert gini_exposure(fair, k, ids) < gini_exposure(base, k, ids)


# ---------------------------------------------------------------------------
# NSW — spreads exposure to the long tail
# ---------------------------------------------------------------------------
def test_nsw_spreads_exposure():
    n, k = 8, 4
    R = _skewed_R(n, seed=0)
    mask = _full_mask(n)
    ids = _ids(n)

    idx = nsw_rerank(R, mask, k)
    nsw = _to_id_dict(idx, ids)
    base = _to_id_dict(_baseline_topk(R, mask, k), ids)

    assert gini_exposure(nsw, k, ids) < gini_exposure(base, k, ids)
    assert coverage_at_k(nsw, k, ids) >= coverage_at_k(base, k, ids)

    for i in range(n):
        assert len(idx[i]) == k
        assert len(set(idx[i])) == k
        assert all(mask[i, j] for j in idx[i])


# ---------------------------------------------------------------------------
# Mask compliance — infeasible pairs are NEVER selected
# ---------------------------------------------------------------------------
def test_mask_respected_both_rerankers():
    n, k = 8, 4
    R = _skewed_R(n, seed=2)
    # Two gender-like groups (even / odd); cross-group pairs are infeasible.
    mask = _full_mask(n)
    for i in range(n):
        for j in range(n):
            if (i % 2) != (j % 2):
                mask[i, j] = False

    for idx in (fairrec(R, mask, k, alpha=0.5), nsw_rerank(R, mask, k)):
        for i in range(n):
            for j in idx[i]:
                assert mask[i, j]
            assert len(set(idx[i])) == len(idx[i])  # no duplicates


# ---------------------------------------------------------------------------
# Determinism — identical inputs give identical outputs
# ---------------------------------------------------------------------------
def test_determinism():
    n, k = 8, 4
    R = _skewed_R(n, seed=5)
    mask = _full_mask(n)
    assert fairrec(R, mask, k, alpha=0.5) == fairrec(R, mask, k, alpha=0.5)
    assert nsw_rerank(R, mask, k) == nsw_rerank(R, mask, k)


# ---------------------------------------------------------------------------
# Edge cases — k > feasible-candidate count, and N == 2
# ---------------------------------------------------------------------------
def test_k_exceeds_feasible_candidates():
    n = 6
    R = _skewed_R(n, seed=1)
    # Split into two size-3 groups -> each customer has only 2 feasible items.
    mask = _full_mask(n)
    for i in range(n):
        for j in range(n):
            if (i // 3) != (j // 3):
                mask[i, j] = False

    k = 5  # larger than the 2 feasible candidates per customer
    for idx in (fairrec(R, mask, k, alpha=0.5), nsw_rerank(R, mask, k)):
        for i in range(n):
            feasible = [j for j in range(n) if mask[i, j]]
            assert len(idx[i]) == len(feasible)  # all feasible, no more
            assert len(set(idx[i])) == len(idx[i])  # no duplicates
            assert all(mask[i, j] for j in idx[i])


def test_n_two_does_not_crash():
    R = np.array([[0.0, 1.0], [1.0, 0.0]])
    mask = _full_mask(2)
    assert fairrec(R, mask, 3, alpha=0.5) == [[1], [0]]
    assert nsw_rerank(R, mask, 3) == [[1], [0]]


# ---------------------------------------------------------------------------
# End-to-end — real fitted recommender
# ---------------------------------------------------------------------------
def test_end_to_end_recommenders():
    profiles, _ = generate_dataset(n_users=24, seed=3, with_gender=True)
    rec = ReciprocalRecommender(method="recon").fit(profiles)

    R, ids, mask = relevance_matrix(rec)
    assert R.shape == (24, 24)
    assert not mask.diagonal().any()  # no self-recommendations feasible

    fair = fairrec_recommend_all(rec, k=5, alpha=0.6)
    nsw = nsw_recommend_all(rec, k=5)
    base = rec.recommend_all(k=5)

    for uid in rec.node_ids_:
        incompatible = rec._incompatible_ids(uid)
        for recs in (fair, nsw):
            lst = recs[uid]
            assert len(lst) <= 5
            assert len(set(lst)) == len(lst)  # no duplicates
            assert uid not in lst  # never self
            assert all(other not in incompatible for other in lst)

    # Fairness rerankers must not be *worse* than the plain recommender.
    slack = 1e-9
    assert gini_exposure(fair, 5) <= gini_exposure(base, 5) + slack
    assert gini_exposure(nsw, 5) <= gini_exposure(base, 5) + slack

    # Deterministic across repeated calls.
    assert fairrec_recommend_all(rec, k=5, alpha=0.6) == fair
    assert nsw_recommend_all(rec, k=5) == nsw


def test_reranker_classes_parity():
    profiles, _ = generate_dataset(n_users=16, seed=7, with_gender=True)
    rec = ReciprocalRecommender(method="recon").fit(profiles)

    fr = FairRecReranker(rec, alpha=0.5)
    nsw = NSWReranker(rec, relevance_weight=1.0)

    assert fr.node_ids_ == rec.node_ids_
    assert nsw.node_ids_ == rec.node_ids_
    assert fr.recommend_all(k=4) == fairrec_recommend_all(rec, k=4, alpha=0.5)
    assert nsw.recommend_all(k=4) == nsw_recommend_all(rec, k=4)

    uid = rec.node_ids_[0]
    assert fr.recommend(uid, k=4) == fr.recommend_all(k=4)[uid]
    assert nsw.recommend(uid, k=4) == nsw.recommend_all(k=4)[uid]
