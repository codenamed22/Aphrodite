"""Lightweight, fully-offline embedding backend.

This backend produces deterministic pseudo-embeddings by hashing tokens to fixed
random unit vectors. Identical tokens map to identical vectors and distinct
tokens map to (near) orthogonal vectors, so the cosine similarity between two
averaged term vectors behaves like a smooth bag-of-words overlap. This makes it
ideal for fast, reproducible unit tests and quick iteration without downloading
multi-gigabyte pretrained models.

For paper-faithful semantics use :class:`aphrodite.embeddings.Word2VecBertBackend`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from .base import EmbeddingBackend

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _token_vector(token: str, dim: int, salt: str) -> np.ndarray:
    """Deterministically map a token to a fixed unit vector of size ``dim``.

    Uses a stable hash (blake2b) as the seed so vectors are identical across
    processes and runs (Python's builtin ``hash`` is salted per-process).
    """
    digest = hashlib.blake2b((salt + "\x00" + token).encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


class LightweightBackend(EmbeddingBackend):
    """Deterministic hashing-based embeddings (offline, no downloads)."""

    def __init__(self, term_dim: int = 64, context_dim: int = 64) -> None:
        self.term_dim = int(term_dim)
        self.context_dim = int(context_dim)

    def embed_terms(self, tokens: Sequence[str]) -> np.ndarray:
        if not tokens:
            return self._empty_term_vector()
        vecs = [_token_vector(t, self.term_dim, salt="term") for t in tokens if t]
        if not vecs:
            return self._empty_term_vector()
        return np.mean(vecs, axis=0)

    def embed_context(self, text: str) -> np.ndarray:
        tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
        if not tokens:
            return self._empty_context_vector()
        vecs = [_token_vector(t, self.context_dim, salt="context") for t in tokens]
        return np.mean(vecs, axis=0)


__all__ = ["LightweightBackend"]
