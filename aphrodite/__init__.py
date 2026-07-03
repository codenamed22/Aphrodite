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
from .reciprocal import (
    ReciprocalRecommender,
    harmonic_mean,
    METHODS,
    DEFAULT_N_INTERESTS,
    DEFAULT_BIO_WEIGHT,
)
from .metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision_at_k,
    mean_average_precision_at_k,
    evaluate_at_ks,
    reciprocity_rate,
)
from .calibration import (
    similarity_distribution,
    graph_health,
    scan_thresholds,
    suggest_threshold,
    compute_similarity_matrix,
    tune_threshold_supervised,
    DEFAULT_TAU_GRID,
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
    "ReciprocalRecommender",
    "harmonic_mean",
    "METHODS",
    "DEFAULT_N_INTERESTS",
    "DEFAULT_BIO_WEIGHT",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "average_precision_at_k",
    "mean_average_precision_at_k",
    "evaluate_at_ks",
    "reciprocity_rate",
    "similarity_distribution",
    "graph_health",
    "scan_thresholds",
    "suggest_threshold",
    "compute_similarity_matrix",
    "tune_threshold_supervised",
    "DEFAULT_TAU_GRID",
    "__version__",
]
