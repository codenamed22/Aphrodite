import numpy as np
import pytest

from aphrodite import ReciprocalRecommender
from aphrodite.datasets import generate_dataset
from aphrodite.matching import (
    TUMatchRecommender,
    _ipfp_masses,
    directional_score_matrix,
    gender_mask,
    tu_match_scores,
    tu_recommend_all,
)


# -- helpers -------------------------------------------------------------------
def _block_matrix(n_per: int = 4, within: float = 0.9, across: float = 0.1):
    """Symmetric directional scores: two clusters, high within / low across."""
    n = 2 * n_per
    s = np.full((n, n), across, dtype=np.float64)
    s[:n_per, :n_per] = within
    s[n_per:, n_per:] = within
    np.fill_diagonal(s, 0.0)
    return s, n_per


def _kernel_from_scores(s: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Replicate tu_match_scores' Gibbs kernel construction (no mask)."""
    phi = np.clip((s + s.T) / (2.0 * beta), -700, 700)
    k = np.exp(phi)
    np.fill_diagonal(k, 0.0)
    return k


# -- pure IPFP solver ----------------------------------------------------------
def test_tu_match_scores_symmetric_zero_diagonal_and_block_structure():
    s, n_per = _block_matrix(n_per=4, within=0.9, across=0.1)
    mu = tu_match_scores(s, beta=1.0, n_iter=100)

    # symmetric, zero diagonal
    assert np.allclose(mu, mu.T)
    assert np.allclose(np.diag(mu), 0.0)

    n = mu.shape[0]
    within_mask = np.zeros((n, n), dtype=bool)
    within_mask[:n_per, :n_per] = True
    within_mask[n_per:, n_per:] = True
    np.fill_diagonal(within_mask, False)
    across_mask = ~within_mask
    np.fill_diagonal(across_mask, False)

    # every within-cluster mass strictly exceeds every across-cluster mass
    assert mu[within_mask].min() > mu[across_mask].max()


def test_tu_match_scores_rejects_non_square():
    with pytest.raises(ValueError):
        tu_match_scores(np.zeros((3, 4)))


def test_tu_match_scores_n2_no_crash():
    s = np.array([[0.0, 0.5], [0.3, 0.0]], dtype=np.float64)
    mu = tu_match_scores(s, n_iter=10)
    assert mu.shape == (2, 2)
    assert np.allclose(np.diag(mu), 0.0)
    assert np.allclose(mu, mu.T)


# -- congestion / anti-popularity property ------------------------------------
def test_popular_user_gets_small_stay_single_mass():
    n = 6
    s = np.full((n, n), 0.1, dtype=np.float64)
    s[:, 0] = 2.0   # everyone strongly prefers user 0
    s[0, :] = 0.2   # user 0 is only mildly interested in everyone
    np.fill_diagonal(s, 0.0)

    k = _kernel_from_scores(s)
    a = _ipfp_masses(k, n_iter=500)

    # high demand => small stay-single mass => discounted for everyone
    assert a[0] < a[1:].mean()


def test_ipfp_masses_in_unit_interval():
    s, _ = _block_matrix(n_per=3)
    k = _kernel_from_scores(s)
    a = _ipfp_masses(k, n_iter=200)
    assert np.all(a > 0.0)
    assert np.all(a <= 1.0 + 1e-9)


# -- gender mask respected -----------------------------------------------------
def test_mask_zeroes_disallowed_pairs():
    n = 4
    rng = np.random.default_rng(0)
    s = rng.random((n, n))
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    mask[0, 1] = mask[1, 0] = False

    mu = tu_match_scores(s, mask=mask, n_iter=50)
    assert np.all(mu[~mask] == 0.0)
    assert mu[0, 1] == 0.0 and mu[1, 0] == 0.0


def test_gender_mask_respected_end_to_end():
    profiles, _ = generate_dataset(n_users=20, seed=4, with_gender=True)
    rec = ReciprocalRecommender(method="recon").fit(profiles)
    s, ids = directional_score_matrix(rec)
    mask = gender_mask(rec, ids)

    # mask is symmetric-feasibility with a False diagonal
    assert np.array_equal(np.diag(mask), np.zeros(len(ids), dtype=bool))

    mu = tu_match_scores(s, mask=mask, n_iter=50)
    assert np.all(mu[~mask] == 0.0)

    recs = tu_recommend_all(rec, k=5)
    for uid in ids:
        incompatible = rec._incompatible_ids(uid)
        assert uid not in recs[uid]
        assert not (set(recs[uid]) & incompatible)


def test_gender_mask_all_true_when_filter_disabled():
    profiles, _ = generate_dataset(n_users=12, seed=1, with_gender=True)
    rec = ReciprocalRecommender(apply_gender_filter=False).fit(profiles)
    _, ids = directional_score_matrix(rec)
    mask = gender_mask(rec, ids)
    off_diag = ~np.eye(len(ids), dtype=bool)
    assert np.all(mask[off_diag])


# -- directional matrix builder ------------------------------------------------
def test_directional_score_matrix_shape_and_diagonal():
    profiles, _ = generate_dataset(n_users=10, seed=2)
    rec = ReciprocalRecommender(method="recon").fit(profiles)
    s, ids = directional_score_matrix(rec)
    assert ids == list(rec.node_ids_)
    assert s.shape == (len(ids), len(ids))
    assert np.allclose(np.diag(s), 0.0)
    # off-diagonal entries reflect the recommender's directional scores
    assert s[0, 1] == pytest.approx(rec._directional(ids[0], ids[1]))


# -- end-to-end wrapper --------------------------------------------------------
def test_end_to_end_recommend_all_deterministic_and_valid():
    profiles, _ = generate_dataset(n_users=24, seed=3, with_gender=True)
    rec = ReciprocalRecommender(method="recon").fit(profiles)
    tu = TUMatchRecommender(rec)

    recs = tu.recommend_all(k=5)
    assert set(recs) == set(rec.node_ids_)
    for uid in rec.node_ids_:
        lst = recs[uid]
        assert len(lst) <= 5
        assert uid not in lst
        assert not (set(lst) & rec._incompatible_ids(uid))

    # deterministic across independent fits/runs
    rec2 = ReciprocalRecommender(method="recon").fit(profiles)
    recs2 = TUMatchRecommender(rec2).recommend_all(k=5)
    assert recs == recs2


def test_recommend_matches_recommend_all_row():
    profiles, _ = generate_dataset(n_users=16, seed=5, with_gender=True)
    rec = ReciprocalRecommender(method="recon").fit(profiles)
    tu = TUMatchRecommender(rec)
    all_recs = tu.recommend_all(k=4)
    target = rec.node_ids_[0]
    assert tu.recommend(target, k=4) == all_recs[target]


def test_recommend_unknown_target_raises():
    profiles, _ = generate_dataset(n_users=8, seed=6)
    rec = ReciprocalRecommender(method="recon").fit(profiles)
    tu = TUMatchRecommender(rec)
    with pytest.raises(KeyError):
        tu.recommend("does-not-exist")
