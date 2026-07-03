"""Embedding backends for the matchmaking algorithm.

Two attribute families are embedded differently (paper §3.1.2):

* **Term-based** attributes (interests, hobbies, occupation) → mean of word
  vectors, OOV words ignored (Eq. 1).
* **Context-based** attribute (biography) → a single sentence embedding (Eq. 2).

Because cosine similarity is only ever computed between *matching* attributes of
two users (interests↔interests, biography↔biography), the term and context
spaces are independent and may have different dimensionalities.
"""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np


class EmbeddingBackend(abc.ABC):
    """Abstract embedding backend.

    Implementations provide term-based word embeddings (averaged over tokens)
    and a context-based sentence embedding.
    """

    #: Dimensionality of term-based (word) vectors.
    term_dim: int
    #: Dimensionality of context-based (sentence) vectors.
    context_dim: int

    @abc.abstractmethod
    def embed_terms(self, tokens: Sequence[str]) -> np.ndarray:
        """Return the mean word vector for ``tokens`` (Eq. 1).

        Out-of-vocabulary tokens are ignored. If no token has a vector, a zero
        vector of shape ``(term_dim,)`` is returned.
        """

    @abc.abstractmethod
    def embed_context(self, text: str) -> np.ndarray:
        """Return a single context vector for ``text`` (Eq. 2), shape ``(context_dim,)``."""

    def _empty_term_vector(self) -> np.ndarray:
        return np.zeros(self.term_dim, dtype=np.float64)

    def _empty_context_vector(self) -> np.ndarray:
        return np.zeros(self.context_dim, dtype=np.float64)


__all__ = ["EmbeddingBackend"]
