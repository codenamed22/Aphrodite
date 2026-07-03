"""Tests for the DPGNN/ConFit-inspired asymmetric reciprocal recommender."""

import numpy as np
import pytest

from aphrodite.asymmetric import AsymmetricRecommender
from aphrodite.datasets import generate_dataset
from aphrodite.matching import TUMatchRecommender
from aphrodite.profiles import UserProfile


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


# -- asymmetry ------------------------------------------------------------------
def test_directional_scores_are_asymmetric():
    profiles, _ = generate_dataset(n_users=20, seed=1, with_gender=True)
    rec = AsymmetricRecommender(homophily=0.4).fit(profiles)
    ids = rec.node_ids_
    found = False
    for a in ids:
        for b in ids:
            if a == b:
                continue
            if abs(rec._directional(a, b) - rec._directional(b, a)) > 1e-6:
                found = True
                break
        if found:
            break
    assert found, "expected at least one genuinely directional (asymmetric) pair"


# -- symmetry at homophily == 1.0 ----------------------------------------------
def test_symmetric_at_full_homophily():
    profiles, _ = generate_dataset(n_users=15, seed=2, with_gender=True)
    rec = AsymmetricRecommender(homophily=1.0).fit(profiles)
    ids = rec.node_ids_

    # pref_vec == self_vec for a few users
    for uid in ids[:5]:
        assert np.allclose(rec.pref_vecs_[uid], rec.self_vecs_[uid])

    # directional score is symmetric for several pairs
    pairs = [(ids[0], ids[1]), (ids[2], ids[5]), (ids[3], ids[7]), (ids[4], ids[9])]
    for a, b in pairs:
        assert rec._directional(a, b) == pytest.approx(
            rec._directional(b, a), abs=1e-9
        )


# -- unit norm of stored vectors -----------------------------------------------
def test_vectors_are_unit_norm_or_zero():
    profiles, _ = generate_dataset(n_users=20, seed=3, with_gender=True)
    rec = AsymmetricRecommender(homophily=0.5).fit(profiles)
    for uid in rec.node_ids_:
        for vec in (rec.self_vecs_[uid], rec.pref_vecs_[uid]):
            n = _norm(vec)
            assert np.isclose(n, 1.0, atol=1e-9) or np.isclose(n, 0.0, atol=1e-12)


# -- score_pair: symmetric and in [0, 1] with unit normalizer ------------------
def test_score_pair_symmetric_and_bounded():
    profiles, _ = generate_dataset(n_users=18, seed=4, with_gender=True)
    rec = AsymmetricRecommender(homophily=0.4, score_normalizer="unit").fit(profiles)
    ids = rec.node_ids_
    for a, b in [(ids[0], ids[3]), (ids[1], ids[6]), (ids[2], ids[9])]:
        v = rec.score_pair(a, b)
        assert rec.score_pair(a, b) == pytest.approx(rec.score_pair(b, a), abs=1e-12)
        assert 0.0 <= v <= 1.0


# -- recommend_all: bounded, excludes self + incompatible, deterministic -------
def test_recommend_all_valid_and_deterministic():
    profiles, _ = generate_dataset(n_users=24, seed=5, with_gender=True)
    rec = AsymmetricRecommender(homophily=0.5).fit(profiles)
    recs1 = rec.recommend_all(k=5)
    recs2 = rec.recommend_all(k=5)
    assert recs1 == recs2  # deterministic across runs

    for uid, matches in recs1.items():
        assert len(matches) <= 5
        assert uid not in matches
        incompatible = rec._incompatible_ids(uid)
        assert not (set(matches) & incompatible)


# -- pluggable into the TU congestion reranker ---------------------------------
def test_plugs_into_tu_match_recommender():
    profiles, _ = generate_dataset(n_users=20, seed=6, with_gender=True)
    rec = AsymmetricRecommender(homophily=0.4).fit(profiles)
    tu = TUMatchRecommender(rec, beta=1.0)
    recs = tu.recommend_all(k=5)
    assert set(recs) == set(rec.node_ids_)
    for uid, matches in recs.items():
        assert len(matches) <= 5
        assert uid not in matches
        assert not (set(matches) & rec._incompatible_ids(uid))


# -- pref_text_fn hook (smoke) -------------------------------------------------
def test_pref_text_fn_hook():
    profiles, _ = generate_dataset(n_users=20, seed=7, with_gender=True)

    base = AsymmetricRecommender(homophily=0.4).fit(profiles)
    hooked = AsymmetricRecommender(
        homophily=0.4, pref_text_fn=lambda p: "music dancing art"
    ).fit(profiles)

    base_recs = base.recommend_all(k=5)
    hooked_recs = hooked.recommend_all(k=5)

    # Always valid output.
    for recs, rec in ((base_recs, base), (hooked_recs, hooked)):
        for uid, matches in recs.items():
            assert len(matches) <= 5
            assert uid not in matches
            assert not (set(matches) & rec._incompatible_ids(uid))

    # The hook feeds a constant preference signal, so it must change at least one
    # user's preference vector, and hence (deterministically) some top-1 pick.
    changed_vec = any(
        not np.allclose(base.pref_vecs_[uid], hooked.pref_vecs_[uid])
        for uid in base.node_ids_
    )
    assert changed_vec
    changed_top1 = any(
        base_recs[uid][:1] != hooked_recs[uid][:1] for uid in base.node_ids_
    )
    assert changed_top1


# -- validation / error handling -----------------------------------------------
def test_invalid_homophily_raises():
    with pytest.raises(ValueError):
        AsymmetricRecommender(homophily=1.5)
    with pytest.raises(ValueError):
        AsymmetricRecommender(homophily=-0.1)


def test_invalid_aggregation_raises():
    with pytest.raises(ValueError):
        AsymmetricRecommender(aggregation="nope")


def test_invalid_score_normalizer_raises():
    with pytest.raises(ValueError):
        AsymmetricRecommender(score_normalizer="nope")


def test_unknown_user_in_score_pair_raises():
    profiles, _ = generate_dataset(n_users=6, seed=8)
    rec = AsymmetricRecommender().fit(profiles)
    with pytest.raises(KeyError):
        rec.score_pair(rec.node_ids_[0], "does-not-exist")
    with pytest.raises(KeyError):
        rec.score_pair("does-not-exist", rec.node_ids_[0])


def test_unknown_target_in_score_raises():
    profiles, _ = generate_dataset(n_users=6, seed=8)
    rec = AsymmetricRecommender().fit(profiles)
    with pytest.raises(KeyError):
        rec.score("does-not-exist")


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        AsymmetricRecommender().recommend("u000")


def test_empty_profile_yields_zero_vectors_and_score():
    empty = UserProfile(user_id="empty", interests="", hobbies="", occupation="",
                        biography="")
    other = UserProfile(user_id="other", interests="", hobbies="", occupation="",
                        biography="")
    rec = AsymmetricRecommender(
        apply_gender_filter=False, score_normalizer="clip01"
    ).fit([empty, other])
    assert _norm(rec.self_vecs_["empty"]) == pytest.approx(0.0)
    assert _norm(rec.pref_vecs_["empty"]) == pytest.approx(0.0)
    # zero vectors -> cosine 0 -> clip01 -> score 0.0 without crashing
    assert rec.score_pair("empty", "other") == pytest.approx(0.0)
    assert rec.recommend("empty", k=5) == ["other"]
