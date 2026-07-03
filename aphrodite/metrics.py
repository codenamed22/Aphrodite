"""Evaluation metrics for user matchmaking (paper §4.2).

Implements Precision@k, Recall@k, F1@k (Eq. 7-9) and Mean Average Precision@k
(Eq. 10-12). All functions take a ranked list of predicted user ids and the set
of ground-truth relevant user ids.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Sequence


def _relevant_set(relevant: Iterable[Hashable]) -> set[Hashable]:
    return set(relevant)


def precision_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Precision@k (Eq. 7): fraction of the top-k that are relevant."""
    if k <= 0:
        return 0.0
    rel = _relevant_set(relevant)
    topk = ranked[:k]
    hits = sum(1 for u in topk if u in rel)
    return hits / k


def recall_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """Recall@k (Eq. 8): fraction of relevant users found in the top-k."""
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for u in topk if u in rel)
    return hits / len(rel)


def f1_at_k(ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """F1@k (Eq. 9): harmonic mean of precision and recall at k."""
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def average_precision_at_k(
    ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Average Precision@k for a single user (Eq. 11-12).

    Sum over ranks 1..k of ``P@i * rel_i``, divided by the total number of
    relevant users (paper-faithful denominator).
    """
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    score = 0.0
    hits = 0
    for i, user in enumerate(ranked[:k], start=1):
        if user in rel:  # rel_k indicator (Eq. 12)
            hits += 1
            score += hits / i  # precision at this rank
    return score / len(rel)


def mean_average_precision_at_k(
    ranked_lists: Sequence[Sequence[Hashable]],
    relevant_sets: Sequence[Iterable[Hashable]],
    k: int,
) -> float:
    """MAP@k (Eq. 10): mean of per-user AP@k across all users."""
    if not ranked_lists:
        return 0.0
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets must have equal length")
    aps = [
        average_precision_at_k(ranked, relevant, k)
        for ranked, relevant in zip(ranked_lists, relevant_sets)
    ]
    return sum(aps) / len(aps)


def evaluate_at_ks(
    ranked_lists: Sequence[Sequence[Hashable]],
    relevant_sets: Sequence[Iterable[Hashable]],
    ks: Sequence[int] = (5, 10, 15, 20),
) -> dict[str, float]:
    """Compute mean P/R/F1 and MAP at each k across all evaluated users.

    Returns a flat dict with keys like ``P@5``, ``R@10``, ``F1@15``, ``MAP@20``.
    Users with no relevant matches are skipped for the mean P/R/F1 (recall is
    undefined for them), matching typical recommender evaluation.
    """
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked_lists and relevant_sets must have equal length")

    results: dict[str, float] = {}
    pairs = [
        (ranked, set(rel))
        for ranked, rel in zip(ranked_lists, relevant_sets)
        if len(set(rel)) > 0
    ]
    for k in ks:
        if pairs:
            p = sum(precision_at_k(r, rel, k) for r, rel in pairs) / len(pairs)
            rec = sum(recall_at_k(r, rel, k) for r, rel in pairs) / len(pairs)
            f1 = sum(f1_at_k(r, rel, k) for r, rel in pairs) / len(pairs)
        else:
            p = rec = f1 = 0.0
        map_k = mean_average_precision_at_k(
            [r for r, _ in pairs], [rel for _, rel in pairs], k
        )
        results[f"P@{k}"] = p
        results[f"R@{k}"] = rec
        results[f"F1@{k}"] = f1
        results[f"MAP@{k}"] = map_k
    return results


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "average_precision_at_k",
    "mean_average_precision_at_k",
    "evaluate_at_ks",
]
