"""Paper-faithful embedding backend: Word2Vec (term) + BERT (context).

* Term-based attributes are embedded as the mean of pretrained Word2Vec word
  vectors (default: GoogleNews 300d), with OOV tokens ignored (Eq. 1).
* The biography is embedded with a BERT sentence encoder via
  ``sentence-transformers`` (Eq. 2).

Heavy dependencies (``gensim``, ``sentence-transformers``) are imported lazily so
the rest of the package works without them. Install with ``pip install
aphrodite[full]``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import EmbeddingBackend


class Word2VecBertBackend(EmbeddingBackend):
    """Word2Vec + BERT embeddings matching the paper.

    Parameters
    ----------
    word2vec_name:
        gensim-downloader name of the word vectors (default GoogleNews 300d).
    bert_model:
        sentence-transformers model name for the biography encoder.
    word2vec:
        Optional preloaded gensim KeyedVectors (skips download).
    sentence_encoder:
        Optional preloaded SentenceTransformer instance.
    """

    def __init__(
        self,
        word2vec_name: str = "word2vec-google-news-300",
        bert_model: str = "all-MiniLM-L6-v2",
        word2vec: Optional[object] = None,
        sentence_encoder: Optional[object] = None,
    ) -> None:
        self._word2vec_name = word2vec_name
        self._bert_model = bert_model
        self._kv = word2vec
        self._encoder = sentence_encoder
        self._term_dim: Optional[int] = None
        self._context_dim: Optional[int] = None

    # -- lazy loaders ---------------------------------------------------------
    def _kv_model(self):
        if self._kv is None:
            import gensim.downloader as api  # noqa: WPS433 (lazy import)

            self._kv = api.load(self._word2vec_name)
        return self._kv

    def _encoder_model(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            self._encoder = SentenceTransformer(self._bert_model)
        return self._encoder

    # -- dimensions -----------------------------------------------------------
    @property
    def term_dim(self) -> int:  # type: ignore[override]
        if self._term_dim is None:
            self._term_dim = int(self._kv_model().vector_size)
        return self._term_dim

    @term_dim.setter
    def term_dim(self, value: int) -> None:
        self._term_dim = int(value)

    @property
    def context_dim(self) -> int:  # type: ignore[override]
        if self._context_dim is None:
            self._context_dim = int(self._encoder_model().get_sentence_embedding_dimension())
        return self._context_dim

    @context_dim.setter
    def context_dim(self, value: int) -> None:
        self._context_dim = int(value)

    # -- embedding ------------------------------------------------------------
    def embed_terms(self, tokens: Sequence[str]) -> np.ndarray:
        kv = self._kv_model()
        vecs = [kv[t] for t in tokens if t in kv]  # OOV ignored (Eq. 1)
        if not vecs:
            return self._empty_term_vector()
        return np.mean(np.asarray(vecs, dtype=np.float64), axis=0)

    def embed_context(self, text: str) -> np.ndarray:
        if not text:
            return self._empty_context_vector()
        enc = self._encoder_model()
        vec = enc.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(vec, dtype=np.float64)


__all__ = ["Word2VecBertBackend"]
