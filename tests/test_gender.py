"""Tests for the dating-app gender identity and preference filtering."""

import pytest

from aphrodite import MatchmakingAlgorithm, UserProfile
from aphrodite.datasets import GENDERS, generate_dataset


# -- UserProfile.is_compatible_with --------------------------------------------

def _p(uid, gender, seeking):
    return UserProfile(uid, gender=gender, seeking=frozenset(seeking))


def test_mutual_straight_compatible():
    man = _p("m", "man", {"woman"})
    woman = _p("w", "woman", {"man"})
    assert man.is_compatible_with(woman)
    assert woman.is_compatible_with(man)  # symmetric


def test_one_sided_preference_not_compatible():
    man = _p("m", "man", {"woman"})
    woman = _p("w", "woman", {"woman"})  # not into men
    assert not man.is_compatible_with(woman)
    assert not woman.is_compatible_with(man)


def test_empty_seeking_is_open_to_everyone():
    open_user = _p("o", "non-binary", set())
    straight_woman = _p("w", "woman", {"man"})
    # open_user accepts her, but she only seeks men -> not mutual
    assert not open_user.is_compatible_with(straight_woman)
    # two open users are compatible
    other_open = _p("o2", "man", set())
    assert open_user.is_compatible_with(other_open)


def test_bisexual_compatible_with_multiple():
    bi = _p("b", "woman", {"man", "woman", "non-binary"})
    man = _p("m", "man", {"woman"})
    nb = _p("n", "non-binary", {"woman"})
    assert bi.is_compatible_with(man)
    assert bi.is_compatible_with(nb)


def test_compatibility_case_insensitive():
    man = _p("m", "Man", {"Woman"})
    woman = _p("w", "woman", {"man"})
    assert man.is_compatible_with(woman)


# -- serialization -------------------------------------------------------------

def test_as_dict_from_dict_roundtrip_with_gender():
    p = UserProfile("u1", name="A", interests="x", gender="woman",
                    seeking=frozenset({"man", "woman"}))
    restored = UserProfile.from_dict(p.as_dict())
    assert restored.gender == "woman"
    assert restored.seeking == frozenset({"man", "woman"})
    assert restored == p


def test_from_dict_accepts_comma_separated_seeking():
    p = UserProfile.from_dict({"user_id": "u1", "gender": "man", "seeking": "woman, non-binary"})
    assert p.seeking == frozenset({"woman", "non-binary"})


def test_from_dict_defaults_empty_gender():
    p = UserProfile.from_dict({"user_id": "u1"})
    assert p.gender == ""
    assert p.seeking == frozenset()


# -- matchmaker filtering ------------------------------------------------------

def test_recommendations_respect_gender_filter():
    profiles, _ = generate_dataset(n_users=60, seed=7, with_gender=True)
    by_id = {p.user_id: p for p in profiles}
    algo = MatchmakingAlgorithm(apply_gender_filter=True).fit(profiles)
    for p in profiles:
        for r in algo.recommend(p.user_id, k=20):
            assert p.is_compatible_with(by_id[r]), (
                f"{p.user_id} matched incompatible {r}"
            )


def test_filter_off_may_return_incompatible():
    # A straight man and a gay man share all interests but are incompatible.
    profiles = [
        UserProfile("m1", interests="guitar piano jazz", hobbies="composing",
                    occupation="musician", biography="I love music.",
                    gender="man", seeking=frozenset({"woman"})),
        UserProfile("m2", interests="guitar piano jazz", hobbies="composing",
                    occupation="musician", biography="I love music.",
                    gender="man", seeking=frozenset({"man"})),
        UserProfile("w1", interests="guitar piano jazz", hobbies="composing",
                    occupation="musician", biography="I love music.",
                    gender="woman", seeking=frozenset({"man"})),
    ]
    algo_off = MatchmakingAlgorithm(threshold=0.5, apply_gender_filter=False).fit(profiles)
    algo_on = MatchmakingAlgorithm(threshold=0.5, apply_gender_filter=True).fit(profiles)

    # m1 is a straight man; m2 (gay man) is incompatible, w1 is compatible.
    off = algo_off.recommend("m1", k=5)
    on = algo_on.recommend("m1", k=5)
    assert "m2" in off          # filter off surfaces the incompatible man
    assert "m2" not in on       # filter on removes him
    assert "w1" in on           # compatible woman remains


def test_default_filter_is_on():
    algo = MatchmakingAlgorithm()
    assert algo.apply_gender_filter is True


# -- dataset gender generation -------------------------------------------------

def test_with_gender_assigns_valid_genders():
    profiles, _ = generate_dataset(n_users=40, seed=3, with_gender=True)
    for p in profiles:
        assert p.gender in GENDERS
        assert p.seeking  # non-empty seeking set assigned


def test_without_gender_leaves_fields_empty():
    profiles, _ = generate_dataset(n_users=20, seed=3, with_gender=False)
    for p in profiles:
        assert p.gender == ""
        assert p.seeking == frozenset()


def test_gender_aware_ground_truth_is_compatible():
    profiles, gt = generate_dataset(n_users=60, seed=5, with_gender=True)
    by_id = {p.user_id: p for p in profiles}
    for uid, matches in gt.items():
        for other in matches:
            assert by_id[uid].is_compatible_with(by_id[other])


def test_gender_ground_truth_subset_of_cluster():
    # Gender-aware GT must be a subset of the plain cluster GT.
    _, gt_plain = generate_dataset(n_users=60, seed=11, with_gender=False)
    _, gt_gender = generate_dataset(n_users=60, seed=11, with_gender=True)
    for uid in gt_plain:
        assert gt_gender[uid] <= gt_plain[uid]
