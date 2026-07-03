import math

import numpy as np
import pytest

from aphrodite import ReciprocalRecommender, UserProfile, reciprocity_rate
from aphrodite.datasets import generate_dataset
from aphrodite.metrics import evaluate_at_ks
from aphrodite.reciprocal import harmonic_mean, _farthest_first_kmeans


# -- harmonic mean --------------------------------------------------------------
def test_harmonic_mean_basic():
    assert harmonic_mean(1.0, 1.0) == pytest.approx(1.0)
    assert harmonic_mean(0.5, 0.5) == pytest.approx(0.5)
    # harmonic mean collapses toward the smaller side (reciprocal penalty)
    assert harmonic_mean(0.9, 0.1) == pytest.approx(2 * 0.9 * 0.1 / 1.0)
    assert harmonic_mean(0.9, 0.1) < (0.9 + 0.1) / 2  # below the arithmetic mean


def test_harmonic_mean_zero_and_negative():
    assert harmonic_mean(0.0, 0.8) == 0.0      # one side uninterested -> no match
    assert harmonic_mean(-0.5, 0.8) == 0.0     # negatives clamped to zero


# -- construction / validation --------------------------------------------------
def test_invalid_method_raises():
    with pytest.raises(ValueError):
        ReciprocalRecommender(method="nope")


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        ReciprocalRecommender().recommend("u000")


def test_unknown_target_raises():
    profiles, _ = generate_dataset(n_users=6, seed=3)
    rec = ReciprocalRecommender().fit(profiles)
    with pytest.raises(KeyError):
        rec.recommend("does-not-exist")
    with pytest.raises(KeyError):
        rec.score_pair("u000", "nobody")


# -- attribute weights (recon) --------------------------------------------------
def test_attribute_weights_normalized():
    profiles, _ = generate_dataset(n_users=8, seed=1)
    rec = ReciprocalRecommender(method="recon").fit(profiles)
    for uid, w in rec.weights_.items():
        assert w.keys() == set(rec.attributes)
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(v >= 0.0 for v in w.values())


def test_empty_profile_weights_uniform():
    profiles = [
        UserProfile("a", interests="guitar jazz", hobbies="composing",
                    occupation="musician", biography="music melody"),
        UserProfile("blank"),  # all attributes empty
    ]
    rec = ReciprocalRecommender(method="recon").fit(profiles)
    w = rec.weights_["blank"]
    assert all(v == pytest.approx(1.0 / len(rec.attributes)) for v in w.values())


# -- symmetry of the pair score -------------------------------------------------
@pytest.mark.parametrize("method", ["recon", "multi_interest"])
def test_pair_score_symmetric(method):
    profiles, _ = generate_dataset(n_users=20, seed=5)
    rec = ReciprocalRecommender(method=method).fit(profiles)
    a, b = profiles[0].user_id, profiles[7].user_id
    assert rec.score_pair(a, b) == pytest.approx(rec.score_pair(b, a))


# -- recommendation behaviour ---------------------------------------------------
@pytest.mark.parametrize("method", ["recon", "multi_interest"])
def test_recommend_excludes_self_and_respects_k(method):
    profiles, _ = generate_dataset(n_users=20, seed=2)
    rec = ReciprocalRecommender(method=method).fit(profiles)
    target = profiles[0].user_id
    matches = rec.recommend(target, k=5)
    assert target not in matches
    assert len(matches) <= 5


@pytest.mark.parametrize("method", ["recon", "multi_interest"])
def test_same_cluster_users_matched(method):
    profiles = [
        UserProfile("t1", interests="guitar piano jazz", hobbies="composing",
                    occupation="musician", biography="I love music and melody."),
        UserProfile("t2", interests="guitar jazz piano", hobbies="composing",
                    occupation="musician", biography="Music and rhythm are my life."),
        UserProfile("s1", interests="football running fitness", hobbies="gym",
                    occupation="coach", biography="Sports and training every day."),
        UserProfile("s2", interests="running football fitness", hobbies="gym",
                    occupation="athlete", biography="I enjoy competition and fitness."),
    ]
    rec = ReciprocalRecommender(method=method).fit(profiles)
    assert rec.recommend("t1", k=1) == ["t2"]
    assert rec.recommend("s1", k=1) == ["s2"]


# -- gender filter --------------------------------------------------------------
@pytest.mark.parametrize("method", ["recon", "multi_interest"])
def test_gender_filter_no_violations(method):
    profiles, _ = generate_dataset(n_users=60, seed=42, with_gender=True)
    by_id = {p.user_id: p for p in profiles}
    rec = ReciprocalRecommender(method=method, apply_gender_filter=True).fit(profiles)
    for p in profiles:
        for r in rec.recommend(p.user_id, k=20):
            assert p.is_compatible_with(by_id[r])


def test_gender_filter_can_be_disabled():
    profiles, _ = generate_dataset(n_users=30, seed=42, with_gender=True)
    rec = ReciprocalRecommender(apply_gender_filter=False).fit(profiles)
    assert rec._incompatible_ids(profiles[0].user_id) == set()


# -- multi-interest facets ------------------------------------------------------
def test_facets_bounded_by_n_interests():
    profiles, _ = generate_dataset(n_users=10, seed=1)
    rec = ReciprocalRecommender(method="multi_interest", n_interests=3).fit(profiles)
    for uid, facets in rec.facets_.items():
        assert facets.shape[0] <= 3
        if facets.shape[0] > 0:
            norms = np.linalg.norm(facets, axis=1)
            assert np.allclose(norms, 1.0)  # centroids are unit-norm


def test_facets_empty_for_blank_terms():
    profiles = [
        UserProfile("a", interests="guitar jazz", hobbies="composing",
                    occupation="musician", biography="music"),
        UserProfile("blank", biography="hello world"),  # no term tokens
    ]
    rec = ReciprocalRecommender(method="multi_interest").fit(profiles)
    assert rec.facets_["blank"].shape[0] == 0


def test_farthest_first_kmeans_deterministic():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((12, 8))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    a = _farthest_first_kmeans(v, 3)
    b = _farthest_first_kmeans(v, 3)
    assert a.shape == b.shape
    assert np.allclose(a, b)  # no RNG -> reproducible


# -- reciprocity metric ---------------------------------------------------------
def test_reciprocity_rate_bounds_and_values():
    # fully mutual
    recs = {"a": ["b"], "b": ["a"]}
    assert reciprocity_rate(recs, k=5) == pytest.approx(1.0)
    # one-directional
    recs = {"a": ["b"], "b": ["c"], "c": ["b"]}
    # pairs: a->b (b lists c, not a) no; b->c (c lists b) yes; c->b (b lists c) yes
    assert reciprocity_rate(recs, k=5) == pytest.approx(2 / 3)
    assert reciprocity_rate({}, k=5) == 0.0
    assert reciprocity_rate(recs, k=0) == 0.0


@pytest.mark.parametrize("method", ["recon", "multi_interest"])
def test_reciprocity_rate_in_unit_interval(method):
    profiles, _ = generate_dataset(n_users=40, seed=7, with_gender=True)
    rec = ReciprocalRecommender(method=method).fit(profiles)
    rr = reciprocity_rate(rec.recommend_all(k=10), k=10)
    assert 0.0 <= rr <= 1.0


# -- end-to-end quality ---------------------------------------------------------
@pytest.mark.parametrize("method", ["recon", "multi_interest"])
def test_end_to_end_metrics_reasonable(method):
    profiles, gt = generate_dataset(n_users=60, seed=42)
    rec = ReciprocalRecommender(method=method, apply_gender_filter=False).fit(profiles)
    ranked = [rec.recommend(p.user_id, k=20) for p in profiles]
    rel = [gt[p.user_id] for p in profiles]
    out = evaluate_at_ks(ranked, rel, ks=(5, 10))
    # clustered structure should be recovered well above random
    assert out["P@5"] > 0.4
    assert out["MAP@10"] > 0.2
