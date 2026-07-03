"""Evaluation metrics for user matchmaking (paper §4.2).

Implements Precision@k, Recall@k, F1@k (Eq. 7-9) and Mean Average Precision@k
(Eq. 10-12). All functions take a ranked list of predicted user ids and the set
of ground-truth relevant user ids.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Mapping, Sequence


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


def reciprocity_rate(
    recommendations: Mapping[Hashable, Sequence[Hashable]], k: int
) -> float:
    """Fraction of top-``k`` recommendations that are mutual.

    For every user ``a`` and each ``b`` in ``a``'s top-``k`` list, the pair
    counts as reciprocal when ``a`` also appears in ``b``'s top-``k`` list. This
    measures how bilaterally consistent a recommender is — a property a dating
    app cares about (a suggested match is only useful if it is offered to *both*
    people). Returns a value in ``[0, 1]``; ``0.0`` when there are no
    recommendations.
    """
    if k <= 0:
        return 0.0
    topk = {a: set(list(recs)[:k]) for a, recs in recommendations.items()}
    total = 0
    mutual = 0
    for a, recs in topk.items():
        for b in recs:
            total += 1
            if a in topk.get(b, set()):
                mutual += 1
    if total == 0:
        return 0.0
    return mutual / total


def _universe(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    all_users: Iterable[Hashable] | None,
) -> set[Hashable]:
    """Return the set of users to evaluate over.

    ``all_users`` when provided, otherwise the keys of ``recommendations``.
    """
    if all_users is not None:
        return set(all_users)
    return set(recommendations.keys())


def exposure_counts(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    k: int,
    all_users: Iterable[Hashable] | None = None,
) -> dict[Hashable, int]:
    """Number of *other* users whose top-``k`` list recommends each user.

    The universe of users is ``all_users`` if given, else the keys of
    ``recommendations``. For every user ``u`` in the universe the returned
    ``exposure`` counts how many users list ``u`` within their top-``k``
    (``list[:k]``) recommendations. A user is never credited for recommending
    itself, and only ids that belong to the universe are counted. Users that no
    one recommends receive ``0``.
    """
    universe = _universe(recommendations, all_users)
    counts: dict[Hashable, int] = {u: 0 for u in universe}
    if k <= 0:
        return counts
    for a, recs in recommendations.items():
        for b in list(recs)[:k]:
            if b != a and b in counts:
                counts[b] += 1
    return counts


def gini_exposure(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    k: int,
    all_users: Iterable[Hashable] | None = None,
) -> float:
    """Gini coefficient of the exposure distribution.

    With exposures sorted ascending as ``e`` (``N = len(e)``, ``S = sum(e)``)::

        G = sum_{i=1..N} (2*i - N - 1) * e[i-1] / (N * S)

    Returns ``0.0`` when ``S == 0`` or ``N == 0`` and is clamped to ``[0, 1]``.
    ``0`` means everyone is recommended equally often; values approaching ``1``
    mean a few users hog all of the exposure.
    """
    e = sorted(exposure_counts(recommendations, k, all_users).values())
    n = len(e)
    s = sum(e)
    if s == 0 or n == 0:
        return 0.0
    g = sum((2 * i - n - 1) * e[i - 1] for i in range(1, n + 1)) / (n * s)
    return max(0.0, min(1.0, g))


def coverage_at_k(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    k: int,
    all_users: Iterable[Hashable] | None = None,
) -> float:
    """Fraction of the universe that appears in at least one top-``k`` list.

    ``|{u in universe : exposure(u) >= 1}| / N`` where ``N`` is the size of the
    universe. Returns ``0.0`` when ``N == 0``.
    """
    counts = exposure_counts(recommendations, k, all_users)
    n = len(counts)
    if n == 0:
        return 0.0
    covered = sum(1 for v in counts.values() if v > 0)
    return covered / n


def long_tail_coverage(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    k: int,
    all_users: Iterable[Hashable] | None = None,
    tail_fraction: float = 0.5,
) -> float:
    """Fraction of the least-recommended users that still get any exposure.

    Users are ranked by exposure ascending; the bottom
    ``floor(tail_fraction * N)`` of them form the "tail". The metric returns the
    fraction of tail users with ``exposure > 0``. Returns ``0.0`` when the tail
    is empty. This detects whether the naturally-less-recommended users receive
    any exposure at all.
    """
    e = sorted(exposure_counts(recommendations, k, all_users).values())
    n = len(e)
    tail_size = math.floor(tail_fraction * n)
    if tail_size <= 0:
        return 0.0
    tail = e[:tail_size]
    return sum(1 for v in tail if v > 0) / len(tail)


def total_mutual_matches(
    recommendations: Mapping[Hashable, Sequence[Hashable]], k: int
) -> int:
    """Raw count of unordered mutual pairs in the top-``k`` recommendations.

    Counts each unordered pair ``{a, b}`` once where ``b`` is in ``a``'s top-``k``
    list *and* ``a`` is in ``b``'s top-``k`` list. This is the absolute count that
    underlies :func:`reciprocity_rate`.
    """
    if k <= 0:
        return 0
    topk = {a: set(list(recs)[:k]) for a, recs in recommendations.items()}
    seen: set[frozenset[Hashable]] = set()
    for a, recs in topk.items():
        for b in recs:
            if a != b and a in topk.get(b, set()):
                seen.add(frozenset((a, b)))
    return len(seen)


def exposure_entropy(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    k: int,
    all_users: Iterable[Hashable] | None = None,
) -> float:
    """Normalized Shannon entropy of the exposure distribution.

    With ``p_i = e_i / S`` over the universe (``S = sum(e)``, ``N`` = universe
    size)::

        H = -sum_{p_i > 0} p_i * ln(p_i) / ln(N)

    Returns ``0.0`` when ``S == 0`` or ``N <= 1``. The result lies in ``[0, 1]``;
    ``1.0`` indicates perfectly uniform exposure.
    """
    e = list(exposure_counts(recommendations, k, all_users).values())
    n = len(e)
    s = sum(e)
    if s == 0 or n <= 1:
        return 0.0
    h = 0.0
    for c in e:
        if c > 0:
            p = c / s
            h -= p * math.log(p)
    return max(0.0, min(1.0, h / math.log(n)))


def bilateral_recall_at_k(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    ground_truth: Mapping[Hashable, set],
    k: int,
) -> dict[str, float]:
    """CRRS-style coverage vs. stability recall (Yang et al., KDD 2024).

    Adapted to a symmetric setting. Build the set ``P`` of unordered ground-truth
    match pairs: for each user ``u`` and each ``v`` in ``ground_truth[u]`` with
    ``u != v`` add ``frozenset({u, v})``; let ``M = |P|``. When ``M == 0`` return
    ``{"coverage_recall": 0.0, "stability_recall": 0.0}``.

    For each pair ``{u, v}`` define ``rec_uv = v in recommendations[u][:k]`` and
    ``rec_vu = u in recommendations[v][:k]``. A pair is a coverage hit when
    ``rec_uv or rec_vu`` and a stability hit when ``rec_uv and rec_vu``::

        coverage_recall  = coverage_hits  / M
        stability_recall = stability_hits / M

    Always ``stability_recall <= coverage_recall``; stability rewards *mutual*
    recommendation of true matches.
    """
    pairs: set[frozenset[Hashable]] = set()
    for u, matches in ground_truth.items():
        for v in matches:
            if u != v:
                pairs.add(frozenset((u, v)))
    m = len(pairs)
    if m == 0:
        return {"coverage_recall": 0.0, "stability_recall": 0.0}
    coverage_hits = 0
    stability_hits = 0
    for pair in pairs:
        u, v = tuple(pair)
        rec_uv = v in list(recommendations.get(u, []))[:k]
        rec_vu = u in list(recommendations.get(v, []))[:k]
        if rec_uv or rec_vu:
            coverage_hits += 1
        if rec_uv and rec_vu:
            stability_hits += 1
    return {
        "coverage_recall": coverage_hits / m,
        "stability_recall": stability_hits / m,
    }


def evaluate_congestion(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    k: int,
    all_users: Iterable[Hashable] | None = None,
) -> dict[str, float]:
    """Bundle the exposure-congestion metrics into a single flat dict.

    Returns ``gini_exposure``, ``coverage``, ``long_tail_coverage``,
    ``total_mutual_matches`` (as a float) and ``exposure_entropy`` using the
    functions above.
    """
    return {
        "gini_exposure": gini_exposure(recommendations, k, all_users),
        "coverage": coverage_at_k(recommendations, k, all_users),
        "long_tail_coverage": long_tail_coverage(recommendations, k, all_users),
        "total_mutual_matches": float(total_mutual_matches(recommendations, k)),
        "exposure_entropy": exposure_entropy(recommendations, k, all_users),
    }


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "average_precision_at_k",
    "mean_average_precision_at_k",
    "evaluate_at_ks",
    "reciprocity_rate",
    "exposure_counts",
    "gini_exposure",
    "coverage_at_k",
    "long_tail_coverage",
    "total_mutual_matches",
    "exposure_entropy",
    "bilateral_recall_at_k",
    "evaluate_congestion",
]
