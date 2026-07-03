"""Algorithm 1 orchestrator: the end-to-end user matchmaking pipeline.

Ties together the three stages of the paper (Thaiprayoon & Unger, NLPIR 2023):

1. Text representation — preprocess + embed each profile attribute (§3.1).
2. Graph representation — pairwise similarity matrix + weighted graph (§3.2).
3. User matchmaking — Personalized PageRank ranking (§3.3).
"""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import networkx as nx
import numpy as np

from .embeddings import EmbeddingBackend, LightweightBackend
from .graph import DEFAULT_THRESHOLD, build_graph
from .ppr import DEFAULT_DAMPING, personalized_pagerank, rank_matches
from .preprocessing import Preprocessor
from .profiles import (
    ATTRIBUTES,
    CONTEXT_ATTRIBUTE,
    TERM_ATTRIBUTES,
    UserProfile,
    validate_profiles,
)
from .similarity import pairwise_similarity_matrix


class MatchmakingAlgorithm:
    """Personalized-PageRank user matchmaking (Algorithm 1).

    Parameters
    ----------
    backend:
        Embedding backend. Defaults to the offline :class:`LightweightBackend`;
        pass :class:`~aphrodite.embeddings.Word2VecBertBackend` for the
        paper-faithful Word2Vec + BERT embeddings.
    preprocessor:
        Text preprocessing pipeline (defaults to the offline configuration).
    threshold:
        ``tau`` similarity threshold for connecting two users (default 0.70).
    damping:
        ``d`` damping/restart weight for Personalized PageRank (default 0.85).
    skip_empty:
        If True, empty attribute pairs are excluded from the aggregated score.
    apply_gender_filter:
        If True (default), recommendations are restricted to users who are
        mutually gender-compatible with the target (see
        :meth:`UserProfile.is_compatible_with`). The similarity graph itself is
        built gender-agnostically so that high-order relationships still
        propagate through same-gender intermediaries; only the final ranked
        list is filtered. Set to False to ignore gender entirely (paper-faithful
        behaviour).
    """

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        preprocessor: Preprocessor | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        damping: float = DEFAULT_DAMPING,
        attributes: Sequence[str] = ATTRIBUTES,
        skip_empty: bool = False,
        apply_gender_filter: bool = True,
    ) -> None:
        self.backend = backend if backend is not None else LightweightBackend()
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor()
        self.threshold = threshold
        self.damping = damping
        self.attributes = tuple(attributes)
        self.skip_empty = skip_empty
        self.apply_gender_filter = apply_gender_filter

        self.profiles_: list[UserProfile] = []
        self.profiles_by_id_: dict[str, UserProfile] = {}
        self.node_ids_: list[str] = []
        self.embeddings_: list[dict[str, np.ndarray]] = []
        self.similarity_matrix_: np.ndarray | None = None
        self.graph_: nx.Graph | None = None

    # -- stage 1: text representation ----------------------------------------
    def embed_profile(self, profile: UserProfile) -> dict[str, np.ndarray]:
        """Embed every attribute of a profile (Eq. 1-2)."""
        emb: dict[str, np.ndarray] = {}
        for attr in self.attributes:
            text = profile.attribute(attr)
            if attr in TERM_ATTRIBUTES:
                tokens = self.preprocessor.process(text)
                emb[attr] = self.backend.embed_terms(tokens)
            elif attr == CONTEXT_ATTRIBUTE:
                emb[attr] = self.backend.embed_context(text)
            else:  # a term-style attribute not in the default context set
                tokens = self.preprocessor.process(text)
                emb[attr] = self.backend.embed_terms(tokens)
        return emb

    # -- fit: build matrix + graph -------------------------------------------
    def fit(self, profiles: Sequence[UserProfile]) -> "MatchmakingAlgorithm":
        """Run stages 1-2: embed profiles, build similarity matrix and graph."""
        validate_profiles(profiles)
        self.profiles_ = list(profiles)
        self.profiles_by_id_ = {p.user_id: p for p in self.profiles_}
        self.node_ids_ = [p.user_id for p in self.profiles_]
        self.embeddings_ = [self.embed_profile(p) for p in self.profiles_]
        self.similarity_matrix_ = pairwise_similarity_matrix(
            self.embeddings_, attributes=self.attributes, skip_empty=self.skip_empty
        )
        self.graph_ = build_graph(
            self.similarity_matrix_, node_ids=self.node_ids_, threshold=self.threshold
        )
        return self

    # -- stage 3: matchmaking -------------------------------------------------
    def _check_fitted(self) -> nx.Graph:
        if self.graph_ is None:
            raise RuntimeError("Call fit(profiles) before requesting matches.")
        return self.graph_

    def score(self, target_id: str) -> dict[Hashable, float]:
        """Return Personalized PageRank scores for all users w.r.t. a target."""
        graph = self._check_fitted()
        if target_id not in graph:
            raise KeyError(f"Unknown target user_id: {target_id!r}")
        return personalized_pagerank(graph, target_id, damping=self.damping)

    def _incompatible_ids(self, target_id: str) -> set[Hashable]:
        """Return the set of user ids that are NOT gender-compatible with target.

        Returns an empty set when gender filtering is disabled. Compatibility is
        mutual (see :meth:`UserProfile.is_compatible_with`).
        """
        if not self.apply_gender_filter:
            return set()
        target = self.profiles_by_id_.get(target_id)
        if target is None:
            return set()
        incompatible: set[Hashable] = set()
        for uid, profile in self.profiles_by_id_.items():
            if uid == target_id:
                continue
            if not target.is_compatible_with(profile):
                incompatible.add(uid)
        return incompatible

    def recommend(self, target_id: str, k: int | None = 10) -> list[Hashable]:
        """Return the top-``k`` matching user ids for ``target_id`` (§3.3.2).

        When gender filtering is enabled, only users mutually compatible with the
        target's gender preferences are returned; PPR scores are computed on the
        full graph first so high-order relationships are preserved.
        """
        scores = self.score(target_id)
        return rank_matches(
            scores, target=target_id, k=k, exclude=self._incompatible_ids(target_id)
        )

    def recommend_all(self, k: int | None = 10) -> dict[str, list[Hashable]]:
        """Convenience: recommendations for every user."""
        return {uid: self.recommend(uid, k=k) for uid in self.node_ids_}


__all__ = ["MatchmakingAlgorithm"]
