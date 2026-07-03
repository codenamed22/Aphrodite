"""Similarity computations (paper §3.2.1).

* Cosine similarity between two attribute vectors (Eq. 3).
* Aggregated similarity between two profiles = simple average of per-attribute
  cosine scores (Eq. 4).
* Construction of the ``N x N`` pairwise similarity matrix.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .profiles import ATTRIBUTES

# A profile embedding maps each attribute name to its vector.
ProfileEmbedding = Mapping[str, np.ndarray]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two vectors (Eq. 3).

    Returns 0.0 when either vector has zero norm (degenerate / empty attribute),
    which keeps the aggregated score well-defined.
    """
    a = np.asarray(vec1, dtype=np.float64)
    b = np.asarray(vec2, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def aggregated_similarity(
    emb_a: ProfileEmbedding,
    emb_b: ProfileEmbedding,
    attributes: Sequence[str] = ATTRIBUTES,
    skip_empty: bool = False,
) -> float:
    """Aggregated similarity between two profiles (Eq. 4).

    Simple average of the per-attribute cosine similarities. By default divides
    by the total number of attributes ``q`` (paper-faithful). If ``skip_empty``
    is True, attribute pairs where either side is a zero vector are excluded
    from both the sum and the count.
    """
    scores: list[float] = []
    counted = 0
    for attr in attributes:
        va = emb_a[attr]
        vb = emb_b[attr]
        if skip_empty and (np.linalg.norm(va) == 0.0 or np.linalg.norm(vb) == 0.0):
            continue
        scores.append(cosine_similarity(va, vb))
        counted += 1
    denom = counted if skip_empty else len(attributes)
    if denom == 0:
        return 0.0
    return float(sum(scores) / denom)


def pairwise_similarity_matrix(
    embeddings: Sequence[ProfileEmbedding],
    attributes: Sequence[str] = ATTRIBUTES,
    skip_empty: bool = False,
) -> np.ndarray:
    """Build the symmetric ``N x N`` aggregated similarity matrix.

    The diagonal is set to 1.0 (a user is maximally similar to itself); it is
    not used for edge construction since the graph has no self-loops.
    """
    n = len(embeddings)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            score = aggregated_similarity(
                embeddings[i], embeddings[j], attributes, skip_empty=skip_empty
            )
            matrix[i, j] = score
            matrix[j, i] = score
    return matrix


__all__ = [
    "cosine_similarity",
    "aggregated_similarity",
    "pairwise_similarity_matrix",
    "ProfileEmbedding",
]
