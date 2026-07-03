"""Phase 3 — congestion-aware TU-matching reranker (Choo–Siow / IPFP).

This module reranks the reciprocal recommender's directional scores through the
lens of a **transferable-utility (TU) matching market** at equilibrium, following
Tomita, Yamamoto & Kohjima, *"Congestion-aware Reciprocal Recommendation for
Online Dating"* (arXiv:2306.09060; reference implementation
``CyberAgentAILab/tu-matching-recommendation``).

The idea in one paragraph
-------------------------
Ordinary reciprocal rerankers score a pair ``(i, j)`` in isolation, so a small
number of very desirable users are recommended to *everyone*. That creates
**congestion**: those users are swamped, most requests go unanswered, and the
market as a whole produces few realised matches. A matching-market model fixes
this by making the recommendation of ``j`` to ``i`` depend on *how much everyone
else also wants ``j``*.

We treat every user as an agent in a symmetric (non-bipartite) Choo–Siow
matching market. From the directional scores ``s(i→j)`` we form a symmetric
surplus ``Phi = (S + Sᵀ) / (2β)`` and the Gibbs kernel ``K = exp(Φ)`` (the
"attractiveness" of each pairing, temperature ``β``). The Iterative Proportional
Fitting Procedure (IPFP / Sinkhorn) then solves for the equilibrium
**match masses** ``μ`` under unit mass per agent::

    aᵢ  (stay-single mass of user i)      solves the fixed point
    aᵢ = sqrt(1 + sᵢ²) − sᵢ ,   sᵢ = ½ (K a)ᵢ
    μ  = K ⊙ (a aᵀ)

The vector ``a`` is the market's anti-congestion mechanism: a high-demand user
has a large ``sᵢ`` and therefore a *small* stay-single mass ``aᵢ``, and because
``μ[i, j] = K[i, j]·aᵢ·aⱼ`` that small ``aᵢ`` **discounts the popular user for
everyone**. Ranking each user's candidates by ``μ[i, :]`` (descending) spreads
attention away from the congested few and toward feasible, likely-reciprocated
matches — a threshold-free reranking that needs no interaction logs.

Public API
----------
* :func:`tu_match_scores` — pure IPFP solver ``S → μ``.
* :func:`directional_score_matrix` — build ``S`` from a fitted recommender.
* :func:`gender_mask` — boolean feasibility mask (gender compatibility).
* :func:`tu_recommend_all` — end-to-end recommendations for every user.
* :class:`TUMatchRecommender` — thin wrapper around a fitted
  :class:`~aphrodite.reciprocal.ReciprocalRecommender`.
"""

from __future__ import annotations

from typing import Hashable

import numpy as np

from .ppr import rank_matches
from .reciprocal import ReciprocalRecommender

_OVERFLOW_GUARD: float = 700.0


def _ipfp_masses(K: np.ndarray, n_iter: int = 50, tol: float = 1e-9) -> np.ndarray:
    """Solve the symmetric IPFP fixed point for the stay-single masses ``a``.

    Given the Gibbs kernel ``K`` (symmetric, zero diagonal, feasibility already
    applied), iterates the unit-mass fixed point::

        sᵢ  = ½ (K a)ᵢ
        aᵢ' = sqrt(1 + sᵢ²) − sᵢ

    until the max change drops below ``tol`` or ``n_iter`` iterations elapse.
    A high-demand user ``i`` (large ``sᵢ``) converges to a *small* ``aᵢ`` — the
    congestion discount that is then applied to ``μ`` for every partner.

    Returns the length-``N`` vector ``a`` (all entries in ``(0, 1]``).
    """
    n = K.shape[0]
    a = np.ones(n, dtype=np.float64)
    if n == 0:
        return a
    for _ in range(n_iter):
        s_vec = 0.5 * (K @ a)
        a_new = np.sqrt(1.0 + s_vec**2) - s_vec
        if np.max(np.abs(a_new - a)) < tol:
            a = a_new
            break
        a = a_new
    return a


def tu_match_scores(
    directional: np.ndarray,
    beta: float = 1.0,
    n_iter: int = 50,
    mask: np.ndarray | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """Congestion-aware equilibrium match masses ``μ`` from directional scores.

    Implements the symmetric (non-bipartite) Choo–Siow / IPFP model::

        Φ = (S + Sᵀ) / (2β)         # symmetric surplus, temperature β
        Φ = clip(Φ, −700, 700)      # overflow guard for exp
        K = exp(Φ) ;  diag(K) = 0   # no self-match
        K[~mask] = 0                # infeasible pairs (if mask given)
        a = IPFP(K)                 # stay-single masses (anti-congestion)
        μ = K ⊙ (a aᵀ)              # equilibrium match masses

    Parameters
    ----------
    directional:
        Square ``(N, N)`` matrix of directional scores ``S[i, j] = s(i → j)``.
    beta:
        Temperature ``β``; larger values flatten the surplus (less peaked ``μ``).
    n_iter, tol:
        IPFP iteration budget and convergence tolerance.
    mask:
        Optional ``(N, N)`` boolean matrix; ``True`` marks an *allowed* pairing.
        Disallowed pairs (and the diagonal) receive ``μ = 0``.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(N, N)`` matrix ``μ`` with zero diagonal (and zeros on
        masked-out pairs). Rank user ``i``'s candidates by ``μ[i, :]`` desc.
    """
    S = np.asarray(directional, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(
            f"directional must be a square 2-D matrix, got shape {S.shape!r}"
        )
    n = S.shape[0]

    phi = (S + S.T) / (2.0 * beta)
    phi = np.clip(phi, -_OVERFLOW_GUARD, _OVERFLOW_GUARD)
    k = np.exp(phi)
    np.fill_diagonal(k, 0.0)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (n, n):
            raise ValueError(
                f"mask shape {mask_arr.shape!r} does not match directional {(n, n)!r}"
            )
        k[~mask_arr] = 0.0

    a = _ipfp_masses(k, n_iter=n_iter, tol=tol)
    mu = k * np.outer(a, a)
    return mu


def directional_score_matrix(
    recommender: ReciprocalRecommender,
) -> tuple[np.ndarray, list]:
    """Materialise the dense directional score matrix ``S`` from a recommender.

    Builds ``S[i, j] = recommender._directional(ids[i], ids[j])`` for ``i != j``
    (the diagonal is left at ``0.0``), where ``ids = recommender.node_ids_``.

    Returns ``(S, ids)`` — the ``(N, N)`` matrix and the ordered id list.
    """
    ids = list(recommender.node_ids_)
    n = len(ids)
    s = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a_id = ids[i]
        for j in range(n):
            if i == j:
                continue
            s[i, j] = float(recommender._directional(a_id, ids[j]))
    return s, ids


def gender_mask(recommender: ReciprocalRecommender, ids: list) -> np.ndarray:
    """Boolean feasibility mask; ``mask[i, j]`` is ``True`` when ``i`` may match ``j``.

    A pairing is allowed when ``i != j`` and ``ids[j]`` is not in
    ``recommender._incompatible_ids(ids[i])``. If the recommender has its gender
    filter disabled, every off-diagonal pair is allowed. Each user's
    incompatible set is computed once.
    """
    n = len(ids)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    if not recommender.apply_gender_filter:
        return mask
    index = {uid: j for j, uid in enumerate(ids)}
    for i, uid in enumerate(ids):
        for bad in recommender._incompatible_ids(uid):
            j = index.get(bad)
            if j is not None:
                mask[i, j] = False
    return mask


def tu_recommend_all(
    recommender: ReciprocalRecommender,
    k: int = 10,
    beta: float = 1.0,
    n_iter: int = 50,
) -> dict[str, list]:
    """Congestion-aware top-``k`` recommendations for every user.

    Computes the equilibrium match masses ``μ`` over all users and ranks each
    user's candidates by ``μ[i, :]`` descending, excluding self and
    gender-incompatible users (via :func:`~aphrodite.ppr.rank_matches`).

    Returns ``{user_id: [ranked ids]}`` with lists of length ``<= k``.
    """
    s, ids = directional_score_matrix(recommender)
    mask = gender_mask(recommender, ids)
    mu = tu_match_scores(s, beta=beta, n_iter=n_iter, mask=mask)

    result: dict[str, list] = {}
    for i, uid in enumerate(ids):
        scores: dict[Hashable, float] = {
            ids[j]: float(mu[i, j]) for j in range(len(ids)) if j != i
        }
        result[uid] = rank_matches(
            scores,
            target=uid,
            k=k,
            exclude=recommender._incompatible_ids(uid),
        )
    return result


class TUMatchRecommender:
    """Thin congestion-aware wrapper around a fitted reciprocal recommender.

    Reranks a fitted :class:`~aphrodite.reciprocal.ReciprocalRecommender` through
    the symmetric Choo–Siow / IPFP matching-market model (arXiv:2306.09060),
    replacing pairwise reciprocal scores with equilibrium match masses ``μ`` so
    that high-demand users are automatically discounted (anti-congestion).

    Parameters
    ----------
    recommender:
        A **fitted** ``ReciprocalRecommender`` (its ``_directional`` scores and
        gender compatibility drive the market).
    beta:
        Market temperature ``β`` (see :func:`tu_match_scores`).
    n_iter:
        IPFP iteration budget.
    """

    def __init__(
        self,
        recommender: ReciprocalRecommender,
        beta: float = 1.0,
        n_iter: int = 50,
    ) -> None:
        self.recommender = recommender
        self.beta = float(beta)
        self.n_iter = int(n_iter)
        self.node_ids_ = recommender.node_ids_

    def recommend_all(self, k: int = 10) -> dict[str, list]:
        """Top-``k`` congestion-aware recommendations for every user."""
        return tu_recommend_all(
            self.recommender, k=k, beta=self.beta, n_iter=self.n_iter
        )

    def recommend(self, target_id: str, k: int = 10) -> list:
        """Top-``k`` congestion-aware recommendations for a single user.

        The equilibrium is a global property of the whole market, so this
        computes the full solution and returns ``target_id``'s ranked row.
        """
        if target_id not in self.recommender.profiles_by_id_:
            raise KeyError(f"Unknown target user_id: {target_id!r}")
        return self.recommend_all(k=k)[target_id]


__all__ = [
    "tu_match_scores",
    "directional_score_matrix",
    "gender_mask",
    "tu_recommend_all",
    "TUMatchRecommender",
]
