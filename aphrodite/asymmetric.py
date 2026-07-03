"""Phase 2 (Tier-2) — DPGNN/ConFit-inspired *asymmetric* reciprocal recommender.

The Phase-1 PPR matcher and the RECON-style :class:`ReciprocalRecommender` both
score compatibility with a **symmetric** attribute cosine: the raw affinity of
``A`` for ``B`` uses the *same* representation as ``B`` for ``A`` (only the
per-attribute weighting differs in RECON). Modern learned reciprocal
recommenders instead give every user **two** representations and let the two
directions diverge structurally:

* **DPGNN** — Dual-Perspective Graph Neural Network for reciprocal recommendation
  (Yang et al., *arXiv:2208.08612*). Each user is embedded from *two*
  perspectives — as a *sender* (who they are / their active preference) and as a
  *receiver* (how they are perceived) — and the directional preference of ``A``
  for ``B`` is the inner product of ``A``'s sender embedding with ``B``'s
  receiver embedding. The two perspectives are trained on directional
  like/pass edges, which we do not have.
* **ConFit** — Contrastive fine-tuning for (job) reciprocal recommendation,
  which similarly keeps a "self" view and a "preference/target" view per entity.

**Content-only adaptation.** We have no swipe/like logs, so we cannot *learn*
the two perspectives. Instead we *construct* them from profile text, giving each
user a **self vector** (who they are) and a **preference vector** (who they
want), and score direction ``A -> B`` as ``cos(pref_A, self_B)``. Because
``pref_A`` and ``self_B`` are generally different vectors, this is inherently
**asymmetric**: ``s(A->B) != s(B->A)`` in general, unlike the symmetric
attribute cosine used by RECON.

**Single embedding space.** ``term_dim`` and ``context_dim`` of a backend may
differ (e.g. word2vec=300 vs BERT=384), yet the self and preference vectors must
live in the *same* space to be comparable. We therefore build **every** vector —
including biography tokens — with :meth:`EmbeddingBackend.embed_terms` (term
space, dimensionality ``term_dim``), never ``embed_context``.

**Homophily knob.** The preference vector blends the user's own description with
their stated preference signal (biography, or an optional hypothetical-partner
text) via a homophily weight ``lambda in [0, 1]``::

    self_vec(U) = unit(desc(U))
    pref_vec(U) = unit(lambda * desc(U) + (1 - lambda) * bio(U))

At ``lambda = 1`` the preference vector collapses onto the self vector, so
``pref_vec == self_vec`` and scoring becomes **symmetric** (a pure homophily
matcher). At ``lambda = 0`` the preference is driven entirely by the stated
preference signal. Intermediate values interpolate "similar to me" and "matches
what I describe wanting", and are what make the two directions genuinely differ.

The optional ``pref_text_fn`` hook lets a caller substitute a
locally-generated hypothetical-partner biography for the user's own biography as
the preference signal; it is purely optional and never required (no network).

The class mirrors :class:`~aphrodite.reciprocal.ReciprocalRecommender`'s public
surface (``fit``/``score_pair``/``score``/``recommend``/``recommend_all`` plus
the ``_directional`` / ``_incompatible_ids`` / ``node_ids_`` / ``apply_gender_filter``
duck-type contract) so it drops straight into the same benchmark and into the
Phase-3 :class:`~aphrodite.matching.TUMatchRecommender` congestion reranker.
"""

from __future__ import annotations

from typing import Callable, Hashable, Sequence

import numpy as np

from .embeddings import EmbeddingBackend, LightweightBackend
from .ppr import rank_matches
from .preprocessing import Preprocessor
from .profiles import (
    CONTEXT_ATTRIBUTE,
    TERM_ATTRIBUTES,
    UserProfile,
    validate_profiles,
)
from .reciprocal import (
    AGGREGATIONS,
    AGGREGATORS,
    SCORE_NORMALIZERS,
    _normalize_score,
)
from .similarity import cosine_similarity


class AsymmetricRecommender:
    """Content-based asymmetric reciprocal matchmaker (self/preference vectors).

    Each user gets a **self** vector (who they are, from the term attributes) and
    a **preference** vector (who they want, a homophily blend of their
    description and their preference signal). The directional score
    ``s(A -> B) = cos(pref_A, self_B)`` is asymmetric by construction; the two
    directions are combined into a single reciprocal score with the configured
    aggregation operator.

    Parameters
    ----------
    backend:
        Embedding backend (defaults to the offline :class:`LightweightBackend`).
        Only :meth:`EmbeddingBackend.embed_terms` is used, so the self and
        preference vectors always share the ``term_dim`` space.
    preprocessor:
        Text preprocessing pipeline (defaults to the offline configuration).
    homophily:
        ``lambda in [0, 1]`` — weight of the user's own description in the
        preference vector. ``1.0`` makes ``pref_vec == self_vec`` (symmetric
        homophily matching); ``0.0`` uses only the preference signal.
    aggregation:
        Reciprocal score-combination operator (see
        :data:`~aphrodite.reciprocal.AGGREGATORS`), one of :data:`AGGREGATIONS`.
    score_normalizer:
        How raw directional cosines are mapped before aggregation, one of
        :data:`~aphrodite.reciprocal.SCORE_NORMALIZERS`. Defaults to ``"unit"``
        because a cosine can be negative and ``"unit"`` maps ``[-1, 1] -> [0, 1]``
        (order-preserving), whereas ``"clip01"`` would floor all negatives to 0.
    apply_gender_filter:
        If True (default), the final ranked list is restricted to users mutually
        gender-compatible with the target. Scoring itself is gender-agnostic.
    pref_text_fn:
        Optional ``callable(profile) -> str`` supplying the preference-signal
        text (e.g. a locally-generated hypothetical-partner biography). If given,
        it replaces the user's own biography as the preference signal. Purely
        optional and never required.
    """

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        preprocessor: Preprocessor | None = None,
        homophily: float = 0.5,
        aggregation: str = "harmonic",
        score_normalizer: str = "unit",
        apply_gender_filter: bool = True,
        pref_text_fn: Callable[[UserProfile], str] | None = None,
    ) -> None:
        homophily = float(homophily)
        if not 0.0 <= homophily <= 1.0:
            raise ValueError(
                f"homophily must be in [0, 1], got {homophily!r}"
            )
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
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor()
        )
        self.homophily = homophily
        self.aggregation = aggregation
        self.score_normalizer = score_normalizer
        self.apply_gender_filter = apply_gender_filter
        self.pref_text_fn = pref_text_fn

        self.node_ids_: list[str] = []
        self.profiles_: list[UserProfile] = []
        self.profiles_by_id_: dict[str, UserProfile] = {}
        self.self_vecs_: dict[str, np.ndarray] = {}
        self.pref_vecs_: dict[str, np.ndarray] = {}

    # -- vector construction (all in term space via embed_terms) -------------
    def _unit(self, vec: np.ndarray) -> np.ndarray:
        """Return ``vec / ||vec||``; ``vec`` unchanged if its norm is zero."""
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    def _mean_token_vec(self, text: str) -> np.ndarray:
        """Mean of the per-token *unit* term vectors of ``text``.

        Each token is embedded on its own via ``embed_terms([tok])`` and
        unit-normalized (zero vectors dropped); the returned vector is the mean
        of those unit vectors, or a ``term_dim`` zero vector when the text has no
        (non-zero) tokens.
        """
        tokens = self.preprocessor.process(text)
        vecs: list[np.ndarray] = []
        for tok in tokens:
            v = np.asarray(self.backend.embed_terms([tok]), dtype=np.float64)
            norm = np.linalg.norm(v)
            if norm > 0.0:
                vecs.append(v / norm)
        if not vecs:
            return np.zeros(self.backend.term_dim, dtype=np.float64)
        return np.mean(vecs, axis=0)

    def _desc(self, profile: UserProfile) -> np.ndarray:
        """Description vector from the concatenated TERM_ATTRIBUTES text."""
        text = " ".join(profile.attribute(attr) for attr in TERM_ATTRIBUTES)
        return self._mean_token_vec(text)

    def _pref_signal_text(self, profile: UserProfile) -> str:
        """Preference-signal text: ``pref_text_fn`` output or the biography."""
        if self.pref_text_fn is not None:
            return self.pref_text_fn(profile)
        return profile.attribute(CONTEXT_ATTRIBUTE)

    def _bio(self, profile: UserProfile) -> np.ndarray:
        """Preference-signal vector (biography or hypothetical-partner text)."""
        return self._mean_token_vec(self._pref_signal_text(profile))

    def _self_vec(self, profile: UserProfile) -> np.ndarray:
        """Who ``U`` is: unit-normalized description vector."""
        return self._unit(self._desc(profile))

    def _pref_vec(self, profile: UserProfile) -> np.ndarray:
        """Who ``U`` wants: unit homophily blend of description and preference.

        With ``homophily == 1.0`` this equals :meth:`_self_vec` exactly, making
        the directional score symmetric.
        """
        desc = self._desc(profile)
        bio = self._bio(profile)
        blended = self.homophily * desc + (1.0 - self.homophily) * bio
        return self._unit(blended)

    # -- fit -----------------------------------------------------------------
    def fit(self, profiles: Sequence[UserProfile]) -> "AsymmetricRecommender":
        """Validate profiles and precompute per-user self/preference vectors."""
        validate_profiles(profiles)
        if not profiles:
            raise ValueError("Cannot fit on an empty profile collection.")
        self.profiles_ = list(profiles)
        self.profiles_by_id_ = {p.user_id: p for p in self.profiles_}
        self.node_ids_ = [p.user_id for p in self.profiles_]
        self.self_vecs_ = {p.user_id: self._self_vec(p) for p in self.profiles_}
        self.pref_vecs_ = {p.user_id: self._pref_vec(p) for p in self.profiles_}
        return self

    def _check_fitted(self) -> None:
        if not self.node_ids_:
            raise RuntimeError("Call fit(profiles) before requesting matches.")

    # -- directional scoring (ASYMMETRIC) ------------------------------------
    def _directional(self, a_id: str, b_id: str) -> float:
        """``s(A -> B) = cos(pref_A, self_B)`` — asymmetric by construction."""
        return cosine_similarity(self.pref_vecs_[a_id], self.self_vecs_[b_id])

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

    # -- reciprocal scoring + recommendation ---------------------------------
    def score_pair(self, a_id: str, b_id: str) -> float:
        """Reciprocal score of a pair via the configured aggregation operator.

        Both directional scores are normalized with ``self.score_normalizer`` and
        then combined with ``AGGREGATORS[self.aggregation]``. Symmetric in
        ``(a, b)`` by construction: match quality is a property of the pair even
        though the underlying directional scores are not.
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

    def recommend(self, target_id: str, k: int | None = 10) -> list[Hashable]:
        """Top-``k`` mutually-compatible matches ranked by reciprocal score."""
        scores = self.score(target_id)
        return rank_matches(
            scores, target=target_id, k=k, exclude=self._incompatible_ids(target_id)
        )

    def recommend_all(self, k: int | None = 10) -> dict[str, list[Hashable]]:
        """Convenience: recommendations for every user."""
        return {uid: self.recommend(uid, k=k) for uid in self.node_ids_}


__all__ = ["AsymmetricRecommender"]
