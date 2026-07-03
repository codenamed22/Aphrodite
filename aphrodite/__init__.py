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
    AGGREGATORS,
    AGGREGATIONS,
    SCORE_NORMALIZERS,
)
from .matching import (
    tu_match_scores,
    directional_score_matrix,
    gender_mask,
    tu_recommend_all,
    TUMatchRecommender,
)
from .asymmetric import AsymmetricRecommender
from .fairness import (
    fairrec,
    nsw_rerank,
    relevance_matrix,
    fairrec_recommend_all,
    nsw_recommend_all,
    FairRecReranker,
    NSWReranker,
)
from .metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision_at_k,
    mean_average_precision_at_k,
    evaluate_at_ks,
    reciprocity_rate,
    exposure_counts,
    gini_exposure,
    coverage_at_k,
    long_tail_coverage,
    total_mutual_matches,
    exposure_entropy,
    bilateral_recall_at_k,
    evaluate_congestion,
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
    "AGGREGATORS",
    "AGGREGATIONS",
    "SCORE_NORMALIZERS",
    "tu_match_scores",
    "directional_score_matrix",
    "gender_mask",
    "tu_recommend_all",
    "TUMatchRecommender",
    "AsymmetricRecommender",
    "fairrec",
    "nsw_rerank",
    "relevance_matrix",
    "fairrec_recommend_all",
    "nsw_recommend_all",
    "FairRecReranker",
    "NSWReranker",
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
    "similarity_distribution",
    "graph_health",
    "scan_thresholds",
    "suggest_threshold",
    "compute_similarity_matrix",
    "tune_threshold_supervised",
    "DEFAULT_TAU_GRID",
    "__version__",
]
