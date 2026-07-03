import numpy as np

from aphrodite.embeddings import LightweightBackend


def test_deterministic_across_instances():
    a = LightweightBackend(term_dim=32)
    b = LightweightBackend(term_dim=32)
    va = a.embed_terms(["music", "guitar"])
    vb = b.embed_terms(["music", "guitar"])
    np.testing.assert_allclose(va, vb)


def test_same_token_same_vector():
    be = LightweightBackend(term_dim=16)
    v1 = be.embed_terms(["jazz"])
    v2 = be.embed_terms(["jazz"])
    np.testing.assert_allclose(v1, v2)


def test_distinct_tokens_low_similarity():
    be = LightweightBackend(term_dim=256)
    v1 = be.embed_terms(["jazz"])
    v2 = be.embed_terms(["football"])
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    assert abs(cos) < 0.3


def test_shared_tokens_increase_similarity():
    be = LightweightBackend(term_dim=256)
    shared = be.embed_terms(["a", "b", "c"])
    overlap = be.embed_terms(["a", "b", "d"])
    disjoint = be.embed_terms(["x", "y", "z"])

    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    assert cos(shared, overlap) > cos(shared, disjoint)


def test_empty_returns_zero_vector():
    be = LightweightBackend(term_dim=8, context_dim=8)
    np.testing.assert_allclose(be.embed_terms([]), np.zeros(8))
    np.testing.assert_allclose(be.embed_context(""), np.zeros(8))


def test_dimensions():
    be = LightweightBackend(term_dim=10, context_dim=20)
    assert be.embed_terms(["x"]).shape == (10,)
    assert be.embed_context("hello world").shape == (20,)


class _FakeKV:
    vector_size = 3
    _d = {"music": np.array([1.0, 0.0, 0.0]), "jazz": np.array([0.0, 1.0, 0.0])}

    def __contains__(self, k):
        return k in self._d

    def __getitem__(self, k):
        return self._d[k]


class _FakeEncoder:
    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, text, **kw):
        return np.array([0.1, 0.2, 0.3, 0.4])


def test_word2vec_bert_backend_logic():
    from aphrodite.embeddings import Word2VecBertBackend

    be = Word2VecBertBackend(word2vec=_FakeKV(), sentence_encoder=_FakeEncoder())
    # OOV token 'zzz' is ignored; mean of the two known vectors (Eq. 1).
    np.testing.assert_allclose(be.embed_terms(["music", "jazz", "zzz"]), [0.5, 0.5, 0.0])
    assert be.term_dim == 3
    # all-OOV -> zero vector
    np.testing.assert_allclose(be.embed_terms(["zzz"]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(be.embed_context("hello"), [0.1, 0.2, 0.3, 0.4])
    assert be.context_dim == 4
