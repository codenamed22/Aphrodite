import math

import pytest

from aphrodite import ReciprocalRecommender
from aphrodite.datasets import generate_dataset
from aphrodite.reciprocal import (
    AGGREGATIONS,
    AGGREGATORS,
    agg_arithmetic,
    agg_geometric,
    agg_harmonic,
    agg_min,
    agg_product,
    agg_uninorm,
    harmonic_mean,
)


# -- operator formulas ----------------------------------------------------------
def test_operator_formulas_at_known_point():
    x, y = 0.2, 0.8
    assert agg_min(x, y) == pytest.approx(0.2)
    assert agg_product(x, y) == pytest.approx(0.16)
    assert agg_geometric(x, y) == pytest.approx(0.4)
    assert agg_harmonic(x, y) == pytest.approx(2 * x * y / (x + y))  # 0.32
    assert agg_harmonic(x, y) == pytest.approx(0.32)
    assert agg_arithmetic(x, y) == pytest.approx(0.5)
    # uninorm: xy / (xy + (1-x)(1-y)) = 0.16 / (0.16 + 0.16) = 0.5
    assert agg_uninorm(x, y) == pytest.approx(0.5)


def test_registry_names_and_mapping():
    assert set(AGGREGATORS) == {
        "harmonic",
        "geometric",
        "product",
        "min",
        "arithmetic",
        "uninorm",
    }
    assert AGGREGATIONS == tuple(AGGREGATORS)
    assert AGGREGATORS["harmonic"] is agg_harmonic
    assert AGGREGATORS["geometric"] is agg_geometric
    assert AGGREGATORS["product"] is agg_product
    assert AGGREGATORS["min"] is agg_min
    assert AGGREGATORS["arithmetic"] is agg_arithmetic
    assert AGGREGATORS["uninorm"] is agg_uninorm


# -- ordering property ----------------------------------------------------------
def test_operator_ordering_property():
    # By the AM-GM-HM inequality (harmonic <= geometric <= arithmetic) plus the
    # product/min relation, the harshness ordering at (0.2, 0.8) is:
    #   product (0.16) <= min (0.2) <= harmonic (0.32) <= geometric (0.4)
    #     <= arithmetic (0.5)
    x, y = 0.2, 0.8
    p = agg_product(x, y)
    m = agg_min(x, y)
    g = agg_geometric(x, y)
    h = agg_harmonic(x, y)
    a = agg_arithmetic(x, y)
    assert p <= m <= h <= g <= a


# -- reciprocity: one side zero -------------------------------------------------
def test_reciprocity_zero_side():
    y = 0.8
    for name in ("harmonic", "geometric", "product", "min", "uninorm"):
        assert AGGREGATORS[name](0.0, y) == pytest.approx(0.0)
        assert AGGREGATORS[name](y, 0.0) == pytest.approx(0.0)
    # arithmetic is the non-reciprocal control: returns y / 2
    assert agg_arithmetic(0.0, y) == pytest.approx(y / 2)
    assert agg_arithmetic(y, 0.0) == pytest.approx(y / 2)


# -- defensive clamping ---------------------------------------------------------
def test_operators_clamp_inputs_to_unit_interval():
    for name, fn in AGGREGATORS.items():
        out = fn(1.5, -0.3)  # clamps to (1.0, 0.0)
        assert 0.0 <= out <= 1.0
        # with one side clamped to 0, reciprocal ops collapse to 0
        if name != "arithmetic":
            assert out == pytest.approx(0.0)


# -- backward-compat regression -------------------------------------------------
def test_defaults_match_harmonic_mean_of_directionals():
    profiles, _ = generate_dataset(n_users=20, seed=1)
    rec = ReciprocalRecommender(method="recon").fit(profiles)
    ids = rec.node_ids_
    pairs = [
        (ids[0], ids[1]),
        (ids[0], ids[7]),
        (ids[3], ids[11]),
        (ids[5], ids[19]),
        (ids[2], ids[14]),
    ]
    for a, b in pairs:
        expected = harmonic_mean(rec._directional(a, b), rec._directional(b, a))
        assert rec.score_pair(a, b) == pytest.approx(expected, abs=1e-12)


def test_defaults_identical_to_explicit_harmonic_clip01():
    profiles, _ = generate_dataset(n_users=20, seed=1)
    default = ReciprocalRecommender(method="recon").fit(profiles)
    explicit = ReciprocalRecommender(
        method="recon", aggregation="harmonic", score_normalizer="clip01"
    ).fit(profiles)
    assert default.recommend_all(k=10) == explicit.recommend_all(k=10)


# -- alternative operator end-to-end --------------------------------------------
def test_geometric_runs_end_to_end():
    profiles, _ = generate_dataset(n_users=20, seed=1)
    rec = ReciprocalRecommender(method="recon", aggregation="geometric").fit(profiles)
    recs = rec.recommend_all(k=10)
    assert isinstance(recs, dict)
    for uid, matches in recs.items():
        assert isinstance(matches, list)
        assert len(matches) <= 10
    for a in rec.node_ids_[:5]:
        for b in rec.node_ids_[:5]:
            if a == b:
                continue
            s = rec.score_pair(a, b)
            assert 0.0 <= s <= 1.0


# -- validation -----------------------------------------------------------------
def test_invalid_aggregation_raises():
    with pytest.raises(ValueError):
        ReciprocalRecommender(aggregation="bogus")


def test_invalid_score_normalizer_raises():
    with pytest.raises(ValueError):
        ReciprocalRecommender(score_normalizer="bogus")
