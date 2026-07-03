import math

from aphrodite.metrics import (
    bilateral_recall_at_k,
    coverage_at_k,
    evaluate_congestion,
    exposure_counts,
    exposure_entropy,
    gini_exposure,
    long_tail_coverage,
    total_mutual_matches,
)


# ---------------------------------------------------------------------------
# "Star" case: B, C, D all recommend only A; A recommends B. k = 1.
# exposure: A=3, B=1, C=0, D=0
# ---------------------------------------------------------------------------
STAR = {
    "A": ["B"],
    "B": ["A"],
    "C": ["A"],
    "D": ["A"],
}


def test_star_exposure_counts():
    counts = exposure_counts(STAR, k=1)
    assert counts == {"A": 3, "B": 1, "C": 0, "D": 0}


def test_star_gini_high():
    # sorted exposures [0, 0, 1, 3] -> G = 10 / (4*4) = 0.625
    g = gini_exposure(STAR, k=1)
    assert abs(g - 0.625) < 1e-9
    assert g > 0.4


def test_star_coverage():
    # only A and B ever appear in a top-1 list -> 2 / 4
    assert coverage_at_k(STAR, k=1) == 0.5


def test_star_entropy_below_one():
    h = exposure_entropy(STAR, k=1)
    # p = [3/4, 1/4] -> H = -(0.75 ln0.75 + 0.25 ln0.25) / ln4
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(4)
    assert abs(h - expected) < 1e-9
    assert h < 1.0


def test_star_long_tail_low():
    # bottom floor(0.5*4)=2 users are C, D (exposure 0) -> 0 have exposure>0
    assert long_tail_coverage(STAR, k=1, tail_fraction=0.5) == 0.0


# ---------------------------------------------------------------------------
# "Uniform" case: a directed cycle so everyone has exposure exactly 1.
# ---------------------------------------------------------------------------
UNIFORM = {
    "A": ["B"],
    "B": ["C"],
    "C": ["D"],
    "D": ["A"],
}


def test_uniform_exposure_equal():
    assert exposure_counts(UNIFORM, k=1) == {"A": 1, "B": 1, "C": 1, "D": 1}


def test_uniform_gini_zero():
    assert gini_exposure(UNIFORM, k=1) == 0.0


def test_uniform_entropy_one():
    assert abs(exposure_entropy(UNIFORM, k=1) - 1.0) < 1e-9


def test_uniform_coverage_full():
    assert coverage_at_k(UNIFORM, k=1) == 1.0


def test_uniform_long_tail_full():
    # every tail user has exposure > 0
    assert long_tail_coverage(UNIFORM, k=1, tail_fraction=0.5) == 1.0


# ---------------------------------------------------------------------------
# total_mutual_matches
# ---------------------------------------------------------------------------
def test_total_mutual_matches_counts_once():
    recs = {
        "a": ["b"],
        "b": ["a"],  # mutual pair {a, b}
        "c": ["d"],
        "d": ["e"],  # one-directional c -> d, not mutual
        "e": [],
    }
    assert total_mutual_matches(recs, k=1) == 1


def test_total_mutual_matches_respects_k():
    recs = {
        "a": ["x", "b"],
        "b": ["y", "a"],
    }
    # b is at index 1 in a's list and a at index 1 in b's list
    assert total_mutual_matches(recs, k=1) == 0
    assert total_mutual_matches(recs, k=2) == 1


# ---------------------------------------------------------------------------
# bilateral_recall_at_k
# ---------------------------------------------------------------------------
def test_bilateral_recall_mutual():
    gt = {"a": {"b"}, "b": {"a"}}
    recs = {"a": ["b"], "b": ["a"]}
    res = bilateral_recall_at_k(recs, gt, k=1)
    assert res["coverage_recall"] == 1.0
    assert res["stability_recall"] == 1.0


def test_bilateral_recall_one_directional():
    gt = {"a": {"b"}, "b": {"a"}}
    recs = {"a": ["b"], "b": []}
    res = bilateral_recall_at_k(recs, gt, k=1)
    assert res["coverage_recall"] == 1.0
    assert res["stability_recall"] == 0.0


def test_bilateral_recall_empty_ground_truth():
    res = bilateral_recall_at_k({"a": ["b"]}, {}, k=1)
    assert res == {"coverage_recall": 0.0, "stability_recall": 0.0}


def test_bilateral_recall_stability_leq_coverage():
    gt = {"a": {"b"}, "b": {"a"}, "c": {"d"}, "d": {"c"}}
    recs = {"a": ["b"], "b": ["a"], "c": ["d"], "d": []}
    res = bilateral_recall_at_k(recs, gt, k=1)
    # pair {a,b} mutual, pair {c,d} coverage-only
    assert res["coverage_recall"] == 1.0
    assert res["stability_recall"] == 0.5
    assert res["stability_recall"] <= res["coverage_recall"]


# ---------------------------------------------------------------------------
# Deterministic Gini value
# ---------------------------------------------------------------------------
def test_gini_known_value():
    # exposures [0, 0, 1, 3]: A recommended by B,C,D (=3), B by A (=1), C,D none.
    recs = {"A": ["B"], "B": ["A"], "C": ["A"], "D": ["A"]}
    counts = exposure_counts(recs, k=1)
    assert sorted(counts.values()) == [0, 0, 1, 3]
    assert abs(gini_exposure(recs, k=1) - 0.625) < 1e-9


def test_gini_all_equal_zero():
    recs = {"A": ["B"], "B": ["C"], "C": ["A"]}
    assert sorted(exposure_counts(recs, k=1).values()) == [1, 1, 1]
    assert gini_exposure(recs, k=1) == 0.0


# ---------------------------------------------------------------------------
# all_users universe handling
# ---------------------------------------------------------------------------
def test_all_users_universe_includes_unrecommended():
    recs = {"A": ["B"]}
    counts = exposure_counts(recs, k=1, all_users={"A", "B", "C"})
    assert counts == {"A": 0, "B": 1, "C": 0}
    assert coverage_at_k(recs, k=1, all_users={"A", "B", "C"}) == 1 / 3


# ---------------------------------------------------------------------------
# Edge cases: empty recommendations
# ---------------------------------------------------------------------------
def test_empty_recommendations():
    empty: dict = {}
    assert exposure_counts(empty, k=5) == {}
    assert gini_exposure(empty, k=5) == 0.0
    assert coverage_at_k(empty, k=5) == 0.0
    assert long_tail_coverage(empty, k=5) == 0.0
    assert total_mutual_matches(empty, k=5) == 0
    assert exposure_entropy(empty, k=5) == 0.0
    assert bilateral_recall_at_k(empty, {}, k=5) == {
        "coverage_recall": 0.0,
        "stability_recall": 0.0,
    }
    cong = evaluate_congestion(empty, k=5)
    assert cong == {
        "gini_exposure": 0.0,
        "coverage": 0.0,
        "long_tail_coverage": 0.0,
        "total_mutual_matches": 0.0,
        "exposure_entropy": 0.0,
    }


# ---------------------------------------------------------------------------
# evaluate_congestion aggregation
# ---------------------------------------------------------------------------
def test_evaluate_congestion_bundles_metrics():
    cong = evaluate_congestion(STAR, k=1)
    assert set(cong) == {
        "gini_exposure",
        "coverage",
        "long_tail_coverage",
        "total_mutual_matches",
        "exposure_entropy",
    }
    assert cong["gini_exposure"] == gini_exposure(STAR, k=1)
    assert cong["coverage"] == coverage_at_k(STAR, k=1)
    assert cong["total_mutual_matches"] == float(total_mutual_matches(STAR, k=1))
    assert isinstance(cong["total_mutual_matches"], float)
