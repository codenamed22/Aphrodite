"""Embedding backends for Aphrodite matchmaking."""

from .base import EmbeddingBackend
from .lightweight import LightweightBackend

__all__ = ["EmbeddingBackend", "LightweightBackend", "Word2VecBertBackend"]


def __getattr__(name: str):
    # Lazily expose the heavy backend so importing the package does not require
    # gensim / sentence-transformers to be installed.
    if name == "Word2VecBertBackend":
        from .word2vec_bert import Word2VecBertBackend

        return Word2VecBertBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
