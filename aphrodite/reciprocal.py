"""Phase 2 — reciprocal user-to-user matchmaking (content-based).

The Phase-1 algorithm (Personalized PageRank over a profile-similarity graph)
treats compatibility as *symmetric*: ``sim(A, B) == sim(B, A)``. Real dating is
*reciprocal* — how much A likes B is generally not how much B likes A, and a
good match requires **mutual** interest. This module implements two content-only
reciprocal recommenders that need no interaction logs, so they run on exactly the
same profile data as Phase 1 and can be compared head-to-head.

Two methods are provided (select via ``method``):

* ``"recon"`` — a content-based take on RECON (Pizzato et al., *RECON: a
  reciprocal recommender for online dating*, RecSys 2010). A **directional**
  score ``s(A→B) = Σ_a w_A[a]·cos(emb_A[a], emb_B[a])`` weights each attribute by
  how much A emphasises it (a log-free proxy for A's preferences, since we have
  no like/pass history). The two directions are combined with the **harmonic
  mean**, which collapses toward the *smaller* side and therefore penalises
  lopsided matches — the defining reciprocal property that a plain symmetric
  average lacks.

* ``"multi_interest"`` — keeps the multi-interest idea of MINER (Li et al., ACL
  Findings 2022) but makes it reciprocal and content-based. Each user's term
  tokens are clustered into ``n_interests`` *interest facets*; the directional
  score measures how well the other user *covers each of A's facets*
  (mean over A's facets of the best-matching facet of B), blended with biography
  similarity. Directions are again combined with the harmonic mean.

Both methods reuse the Phase-1 embedding backend, preprocessing, gender filter
and ranking helper, so the only conceptual change is the scoring function.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

from .embeddings import EmbeddingBackend, LightweightBackend
from .ppr import rank_matches
from .preprocessing import Preprocessor
from .profiles import (
    ATTRIBUTES,
    CONTEXT_ATTRIBUTE,
    TERM_ATTRIBUTES,
    UserProfile,
    validate_profiles,
)
from .similarity import cosine_similarity

DEFAULT_N_INTERESTS: int = 3
DEFAULT_BIO_WEIGHT: float = 0.5

METHODS: tuple[str, ...] = ("recon", "multi_interest")


def harmonic_mean(a: float, b: float) -> float:
    """Reciprocal-recommendation score combiner ``2ab / (a + b)``.

    Negative directional scores (possible with real embeddings) are clamped to
    ``0``. Returns ``0.0`` when either side is non-positive, which correctly
    marks a pair as a non-match if *either* user is uninterested — the whole
    point of a reciprocal score.
    """
    a = max(0.0, float(a))
    b = max(0.0, float(b))
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


# -- score normalization -------------------------------------------------------
SCORE_NORMALIZERS: frozenset[str] = frozenset({"clip01", "unit", "none"})


def _normalize_score(x: float, method: str) -> float:
    """Map a raw directional score to a normalized scalar.

    * ``"clip01"`` — clamp to ``[0, 1]`` (``max(0, min(1, x))``); for inputs
      ``<= 1`` this equals the ``max(0, x)`` clamp used by :func:`harmonic_mean`.
    * ``"unit"`` — affine map of ``[-1, 1]`` onto ``[0, 1]`` via
      ``(clip(x, -1, 1) + 1) / 2``, preserving the ordering of negatives.
    * ``"none"`` — identity, just ``float(x)``.
    """
    x = float(x)
    if method == "clip01":
        return max(0.0, min(1.0, x))
    if method == "unit":
        return (max(-1.0, min(1.0, x)) + 1.0) / 2.0
    if method == "none":
        return x
    raise ValueError(
        f"score_normalizer must be one of {sorted(SCORE_NORMALIZERS)!r}, got {method!r}"
    )


# -- reciprocal aggregation operators ------------------------------------------
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def agg_harmonic(x: float, y: float) -> float:
    """Harmonic mean ``2xy / (x + y)``; ``0`` if either side is ``<= 0``."""
    x, y = _clamp01(x), _clamp01(y)
    if x <= 0.0 or y <= 0.0:
        return 0.0
    return 2.0 * x * y / (x + y)


def agg_geometric(x: float, y: float) -> float:
    """Geometric mean ``sqrt(x * y)`` — reciprocal, harsher than harmonic."""
    x, y = _clamp01(x), _clamp01(y)
    return float(np.sqrt(x * y))


def agg_product(x: float, y: float) -> float:
    """Product ``x * y`` — the harshest reciprocal operator."""
    x, y = _clamp01(x), _clamp01(y)
    return x * y


def agg_min(x: float, y: float) -> float:
    """Minimum — a pair is only as good as its least-interested side."""
    x, y = _clamp01(x), _clamp01(y)
    return min(x, y)


def agg_arithmetic(x: float, y: float) -> float:
    """Arithmetic mean ``(x + y) / 2`` — a NON-reciprocal control baseline."""
    x, y = _clamp01(x), _clamp01(y)
    return (x + y) / 2.0


def agg_uninorm(x: float, y: float) -> float:
    """Cross-ratio uninorm ``xy / (xy + (1-x)(1-y))``; ``0`` if denom is ``0``."""
    x, y = _clamp01(x), _clamp01(y)
    num = x * y
    denom = num + (1.0 - x) * (1.0 - y)
    if denom == 0.0:
        return 0.0
    return num / denom


AGGREGATORS: dict[str, callable] = {
    "harmonic": agg_harmonic,
    "geometric": agg_geometric,
    "product": agg_product,
    "min": agg_min,
    "arithmetic": agg_arithmetic,
    "uninorm": agg_uninorm,
}

AGGREGATIONS: tuple[str, ...] = tuple(AGGREGATORS)


def _farthest_first_kmeans(
    vectors: np.ndarray, k: int, iters: int = 10
) -> np.ndarray:
    """Deterministic k-means returning up to ``k`` unit-norm centroids.

    Seeds with farthest-first traversal (no RNG) so results are reproducible
    across runs and processes, then runs a few Lloyd iterations. ``vectors`` is
    an ``(n, d)`` array of token embeddings. The number of centroids returned is
    ``min(k, n_unique_directions)``; empty clusters are dropped.
    """
    n = vectors.shape[0]
    if n == 0:
        return np.empty((0, vectors.shape[1]), dtype=np.float64)
    k = max(1, min(k, n))

    # Farthest-first seeding: start from vector 0, repeatedly add the point
    # farthest (smallest cosine) from the current seed set.
    seed_idx = [0]
    while len(seed_idx) < k:
        best_i, best_d = None, -np.inf
        for i in range(n):
            if i in seed_idx:
                continue
            # distance to seed set = 1 - max cosine to any seed
            max_cos = max(
                float(np.dot(vectors[i], vectors[s])) for s in seed_idx
            )
            d = 1.0 - max_cos
            if d > best_d:
                best_d, best_i = d, i
        if best_i is None:
            break
        seed_idx.append(best_i)

    centroids = vectors[seed_idx].copy()
    for _ in range(iters):
        # Assign each point to the nearest centroid (max cosine).
        sims = vectors @ centroids.T  # (n, k)
        assign = np.argmax(sims, axis=1)
        new_centroids = []
        for c in range(centroids.shape[0]):
            members = vectors[assign == c]
            if len(members) == 0:
                continue
            mean = members.mean(axis=0)
            norm = np.linalg.norm(mean)
            new_centroids.append(mean / norm if norm > 0 else mean)
        if not new_centroids:
            break
        candidate = np.vstack(new_centroids)
        if candidate.shape == centroids.shape and np.allclose(candidate, centroids):
            centroids = candidate
            break
        centroids = candidate
    return centroids


class ReciprocalRecommender:
    """Content-based reciprocal matchmaker (Phase 2).

    Parameters
    ----------
    backend:
        Embedding backend (defaults to the offline :class:`LightweightBackend`).
        The same backends used for Phase 1 apply here.
    preprocessor:
        Text preprocessing pipeline (defaults to the offline configuration).
    method:
        ``"recon"`` (directional weighted similarity) or ``"multi_interest"``
        (interest-facet coverage). Both combine the two directions with the
        configured ``aggregation`` operator (harmonic mean by default).
    aggregation:
        Reciprocal score-combination operator (see :data:`AGGREGATORS`): one of
        ``"harmonic"``, ``"geometric"``, ``"product"``, ``"min"``, ``"uninorm"``
        (all reciprocal), or ``"arithmetic"`` (a non-reciprocal control
        baseline). Applied to the two *normalized* directional scores.
    score_normalizer:
        How raw directional scores (roughly ``[-1, 1]``) are mapped before
        aggregation: ``"clip01"`` (default, clamp to ``[0, 1]``), ``"unit"``
        (affine ``[-1, 1] -> [0, 1]``) or ``"none"`` (identity). The defaults
        (``aggregation="harmonic"``, ``score_normalizer="clip01"``) reproduce the
        original ``harmonic_mean`` behaviour exactly.
    attributes:
        Attribute names to score over (defaults to the paper's four).
    apply_gender_filter:
        If True (default), the final ranked list is restricted to users mutually
        gender-compatible with the target (see
        :meth:`UserProfile.is_compatible_with`). Scoring itself is
        gender-agnostic, mirroring Phase 1.
    n_interests:
        ``K`` — number of interest facets per user for ``"multi_interest"``.
    bio_weight:
        For ``"multi_interest"``, the blend weight of biography similarity vs.
        interest-facet coverage in the directional score (``0..1``).
    """

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        preprocessor: Preprocessor | None = None,
        method: str = "recon",
        attributes: Sequence[str] = ATTRIBUTES,
        apply_gender_filter: bool = True,
        n_interests: int = DEFAULT_N_INTERESTS,
        bio_weight: float = DEFAULT_BIO_WEIGHT,
        aggregation: str = "harmonic",
        score_normalizer: str = "clip01",
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if aggregation not in AGGREGATORS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
            )
        if score_normalizer not in SCORE_NORMALIZERS:
            raise ValueError(
                f"score_normalizer must be one of {sorted(SCORE_NORMALIZERS)!r}, "
                f"got {score_normalizer!r}"
            )
        self.backend = backend if backend is not None else LightweightBackend()
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor()
        self.method = method
        self.attributes = tuple(attributes)
        self.apply_gender_filter = apply_gender_filter
        self.n_interests = int(n_interests)
        self.bio_weight = float(bio_weight)
        self.aggregation = aggregation
        self.score_normalizer = score_normalizer

        self.profiles_: list[UserProfile] = []
        self.profiles_by_id_: dict[str, UserProfile] = {}
        self.node_ids_: list[str] = []
        self.embeddings_: dict[str, dict[str, np.ndarray]] = {}
        self.weights_: dict[str, dict[str, float]] = {}
        self.facets_: dict[str, np.ndarray] = {}

    # -- embedding (shared with Phase 1 semantics) ---------------------------
    def _embed_profile(self, profile: UserProfile) -> dict[str, np.ndarray]:
        emb: dict[str, np.ndarray] = {}
        for attr in self.attributes:
            text = profile.attribute(attr)
            if attr == CONTEXT_ATTRIBUTE:
                emb[attr] = self.backend.embed_context(text)
            else:
                tokens = self.preprocessor.process(text)
                emb[attr] = self.backend.embed_terms(tokens)
        return emb

    def _attribute_weights(self, profile: UserProfile) -> dict[str, float]:
        """Per-user attribute importance weights (sum to 1).

        Content-based proxy for how much the user emphasises each attribute:
        proportional to the number of meaningful tokens they provided for it.
        Term attributes are tokenised with the preprocessor; the biography uses
        a whitespace word count. Falls back to uniform weights if the profile is
        empty. These weights make the two directional scores differ, which is
        what gives the harmonic mean something to penalise.
        """
        raw: dict[str, float] = {}
        for attr in self.attributes:
            text = profile.attribute(attr)
            if attr == CONTEXT_ATTRIBUTE:
                raw[attr] = float(len(text.split()))
            else:
                raw[attr] = float(len(self.preprocessor.process(text)))
        total = sum(raw.values())
        if total <= 0:
            return {attr: 1.0 / len(self.attributes) for attr in self.attributes}
        return {attr: raw[attr] / total for attr in self.attributes}

    def _interest_facets(self, profile: UserProfile) -> np.ndarray:
        """Extract up to ``n_interests`` unit-norm interest-facet vectors.

        Facets are cluster centroids over the embeddings of the user's *term*
        tokens (interests + hobbies + occupation). Returns an ``(m, term_dim)``
        array with ``m <= n_interests``; an empty array if the user has no term
        tokens.
        """
        tokens: list[str] = []
        for attr in self.attributes:
            if attr in TERM_ATTRIBUTES:
                tokens.extend(self.preprocessor.process(profile.attribute(attr)))
        if not tokens:
            return np.empty((0, self.backend.term_dim), dtype=np.float64)
        vecs = []
        for tok in tokens:
            v = self.backend.embed_terms([tok])
            norm = np.linalg.norm(v)
            if norm > 0:
                vecs.append(v / norm)
        if not vecs:
            return np.empty((0, self.backend.term_dim), dtype=np.float64)
        return _farthest_first_kmeans(np.vstack(vecs), self.n_interests)

    # -- fit -----------------------------------------------------------------
    def fit(self, profiles: Sequence[UserProfile]) -> "ReciprocalRecommender":
        """Embed every profile and precompute per-user weights / facets."""
        validate_profiles(profiles)
        self.profiles_ = list(profiles)
        self.profiles_by_id_ = {p.user_id: p for p in self.profiles_}
        self.node_ids_ = [p.user_id for p in self.profiles_]
        self.embeddings_ = {p.user_id: self._embed_profile(p) for p in self.profiles_}
        if self.method == "recon":
            self.weights_ = {
                p.user_id: self._attribute_weights(p) for p in self.profiles_
            }
        else:  # multi_interest
            self.facets_ = {
                p.user_id: self._interest_facets(p) for p in self.profiles_
            }
        return self

    def _check_fitted(self) -> None:
        if not self.node_ids_:
            raise RuntimeError("Call fit(profiles) before requesting matches.")

    # -- directional scoring -------------------------------------------------
    def _directional_recon(self, a_id: str, b_id: str) -> float:
        """``s(A→B) = Σ_a w_A[a]·cos(emb_A[a], emb_B[a])`` (weights from A)."""
        emb_a = self.embeddings_[a_id]
        emb_b = self.embeddings_[b_id]
        w_a = self.weights_[a_id]
        return sum(
            w_a[attr] * cosine_similarity(emb_a[attr], emb_b[attr])
            for attr in self.attributes
        )

    def _facet_coverage(self, facets_a: np.ndarray, facets_b: np.ndarray) -> float:
        """Mean over A's facets of the best cosine match to any of B's facets."""
        if facets_a.shape[0] == 0 or facets_b.shape[0] == 0:
            return 0.0
        sims = facets_a @ facets_b.T  # (|A|, |B|), rows already unit-norm
        best_per_a = sims.max(axis=1)
        return float(best_per_a.mean())

    def _directional_multi_interest(self, a_id: str, b_id: str) -> float:
        """Blend A-centric interest coverage with biography similarity."""
        coverage = self._facet_coverage(self.facets_[a_id], self.facets_[b_id])
        bio_sim = cosine_similarity(
            self.embeddings_[a_id][CONTEXT_ATTRIBUTE],
            self.embeddings_[b_id][CONTEXT_ATTRIBUTE],
        )
        return (1.0 - self.bio_weight) * coverage + self.bio_weight * bio_sim

    def _directional(self, a_id: str, b_id: str) -> float:
        if self.method == "recon":
            return self._directional_recon(a_id, b_id)
        return self._directional_multi_interest(a_id, b_id)

    # -- reciprocal scoring + recommendation ---------------------------------
    def score_pair(self, a_id: str, b_id: str) -> float:
        """Reciprocal score of a pair via the configured aggregation operator.

        Both directional scores are normalized with ``self.score_normalizer`` and
        then combined with ``AGGREGATORS[self.aggregation]``. Symmetric by
        construction (``score_pair(a, b) == score_pair(b, a)``): match quality is
        a property of the pair. With the defaults (``aggregation="harmonic"``,
        ``score_normalizer="clip01"``) this is identical to
        ``harmonic_mean(s(A->B), s(B->A))``.
        """
        self._check_fitted()
        if a_id not in self.profiles_by_id_:
            raise KeyError(f"Unknown user_id: {a_id!r}")
        if b_id not in self.profiles_by_id_:
            raise KeyError(f"Unknown user_id: {b_id!r}")
        da = self._directional(a_id, b_id)
        db = self._directional(b_id, a_id)
        na = _normalize_score(da, self.score_normalizer)
        nb = _normalize_score(db, self.score_normalizer)
        return AGGREGATORS[self.aggregation](na, nb)

    def score(self, target_id: str) -> dict[Hashable, float]:
        """Reciprocal score of ``target_id`` against every other user."""
        self._check_fitted()
        if target_id not in self.profiles_by_id_:
            raise KeyError(f"Unknown target user_id: {target_id!r}")
        return {
            uid: self.score_pair(target_id, uid)
            for uid in self.node_ids_
            if uid != target_id
        }

    def _incompatible_ids(self, target_id: str) -> set[Hashable]:
        """User ids NOT mutually gender-compatible with the target (or empty)."""
        if not self.apply_gender_filter:
            return set()
        target = self.profiles_by_id_.get(target_id)
        if target is None:
            return set()
        return {
            uid
            for uid, profile in self.profiles_by_id_.items()
            if uid != target_id and not target.is_compatible_with(profile)
        }

    def recommend(self, target_id: str, k: int | None = 10) -> list[Hashable]:
        """Top-``k`` mutually-compatible matches ranked by reciprocal score."""
        scores = self.score(target_id)
        return rank_matches(
            scores, target=target_id, k=k, exclude=self._incompatible_ids(target_id)
        )

    def recommend_all(self, k: int | None = 10) -> dict[str, list[Hashable]]:
        """Convenience: recommendations for every user."""
        return {uid: self.recommend(uid, k=k) for uid in self.node_ids_}


__all__ = [
    "ReciprocalRecommender",
    "harmonic_mean",
    "_normalize_score",
    "SCORE_NORMALIZERS",
    "agg_harmonic",
    "agg_geometric",
    "agg_product",
    "agg_min",
    "agg_arithmetic",
    "agg_uninorm",
    "AGGREGATORS",
    "AGGREGATIONS",
    "METHODS",
    "DEFAULT_N_INTERESTS",
    "DEFAULT_BIO_WEIGHT",
]
