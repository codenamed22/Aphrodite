"""Aphrodite — user matchmaking service.

Phase 1 implements Algorithm 1 from Thaiprayoon & Unger, "Enhancing a User
Matchmaking Algorithm using Personalized PageRank" (NLPIR 2023).
"""

from .profiles import UserProfile, ATTRIBUTES, TERM_ATTRIBUTES, CONTEXT_ATTRIBUTE
from .preprocessing import Preprocessor
from .embeddings import EmbeddingBackend, LightweightBackend
from .similarity import (
    cosine_similarity,
    aggregated_similarity,
    pairwise_similarity_matrix,
)
from .graph import build_graph, DEFAULT_THRESHOLD
from .ppr import personalized_pagerank, rank_matches, DEFAULT_DAMPING
from .matchmaker import MatchmakingAlgorithm
from .metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision_at_k,
    mean_average_precision_at_k,
    evaluate_at_ks,
)

__version__ = "0.1.0"

__all__ = [
    "UserProfile",
    "ATTRIBUTES",
    "TERM_ATTRIBUTES",
    "CONTEXT_ATTRIBUTE",
    "Preprocessor",
    "EmbeddingBackend",
    "LightweightBackend",
    "cosine_similarity",
    "aggregated_similarity",
    "pairwise_similarity_matrix",
    "build_graph",
    "DEFAULT_THRESHOLD",
    "personalized_pagerank",
    "rank_matches",
    "DEFAULT_DAMPING",
    "MatchmakingAlgorithm",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "average_precision_at_k",
    "mean_average_precision_at_k",
    "evaluate_at_ks",
    "__version__",
]
