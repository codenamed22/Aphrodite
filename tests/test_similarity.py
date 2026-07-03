import numpy as np

from aphrodite.similarity import (
    aggregated_similarity,
    cosine_similarity,
    pairwise_similarity_matrix,
)


def test_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == 1.0


def test_cosine_orthogonal():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_cosine_zero_vector():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_aggregated_average():
    emb_a = {"interests": np.array([1.0, 0.0]), "hobbies": np.array([1.0, 0.0])}
    emb_b = {"interests": np.array([1.0, 0.0]), "hobbies": np.array([0.0, 1.0])}
    # cos(interests)=1, cos(hobbies)=0 -> avg 0.5
    score = aggregated_similarity(emb_a, emb_b, attributes=("interests", "hobbies"))
    assert score == 0.5


def test_pairwise_matrix_symmetry_and_diagonal():
    embs = [
        {"a": np.array([1.0, 0.0])},
        {"a": np.array([0.0, 1.0])},
        {"a": np.array([1.0, 1.0])},
    ]
    m = pairwise_similarity_matrix(embs, attributes=("a",))
    assert m.shape == (3, 3)
    np.testing.assert_allclose(np.diag(m), np.ones(3))
    np.testing.assert_allclose(m, m.T)
    # user0 vs user2 = cos((1,0),(1,1)) = 1/sqrt2
    assert abs(m[0, 2] - 1 / np.sqrt(2)) < 1e-9
