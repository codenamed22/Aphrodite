import numpy as np
import pytest

from aphrodite.calibration import (
    DEFAULT_TAU_GRID,
    compute_similarity_matrix,
    graph_health,
    scan_thresholds,
    similarity_distribution,
    suggest_threshold,
    tune_threshold_supervised,
)
from aphrodite.datasets import generate_dataset


def _block_matrix(within: float, across: float, sizes: tuple[int, ...]) -> np.ndarray:
    """Symmetric block matrix: high similarity within clusters, low across."""
    n = sum(sizes)
    m = np.full((n, n), across, dtype=np.float64)
    idx = 0
    for s in sizes:
        m[idx:idx + s, idx:idx + s] = within
        idx += s
    np.fill_diagonal(m, 1.0)
    return m


# -- similarity_distribution -------------------------------------------------
def test_distribution_reports_percentiles_and_stats():
    m = _block_matrix(within=0.9, across=0.1, sizes=(3, 3))
    dist = similarity_distribution(m)
    # 6 nodes -> 15 unordered pairs off the diagonal.
    assert dist["count"] == 15
    assert dist["min"] == pytest.approx(0.1)
    assert dist["max"] == pytest.approx(0.9)
    # Percentiles must be non-decreasing.
    ordered = [dist[f"p{q}"] for q in (1, 5, 25, 50, 75, 90, 95, 99)]
    assert ordered == sorted(ordered)


def test_distribution_handles_empty_matrix():
    dist = similarity_distribution(np.ones((1, 1)))
    assert dist["count"] == 0
    assert dist["mean"] == 0.0
    assert dist["p50"] == 0.0


# -- graph_health ------------------------------------------------------------
def test_graph_health_counts_within_cluster_edges():
    m = _block_matrix(within=0.9, across=0.1, sizes=(3, 3))
    # tau between across and within keeps only the two triangles: 2 * C(3,2) = 6.
    health = graph_health(m, node_ids=[f"u{i}" for i in range(6)], tau=0.5)
    assert health["edges"] == 6
    assert health["isolated"] == 0
    assert health["n_components"] == 2
    assert health["largest_component_fraction"] == pytest.approx(0.5)


def test_graph_health_isolates_everyone_above_max_similarity():
    m = _block_matrix(within=0.9, across=0.1, sizes=(3, 3))
    health = graph_health(m, node_ids=None, tau=0.95)
    assert health["edges"] == 0
    assert health["isolated_fraction"] == pytest.approx(1.0)
    assert health["density"] == 0.0


def test_edges_are_monotonic_non_increasing_in_tau():
    rng = np.random.default_rng(0)
    a = rng.random((12, 12))
    m = (a + a.T) / 2.0
    np.fill_diagonal(m, 1.0)
    table = scan_thresholds(m, taus=[0.2, 0.4, 0.6, 0.8])
    edges = [r["edges"] for r in table]
    assert edges == sorted(edges, reverse=True)
    # scan is returned in ascending tau order.
    assert [r["tau"] for r in table] == [0.2, 0.4, 0.6, 0.8]


# -- suggest_threshold -------------------------------------------------------
def test_suggest_returns_grid_tau_and_full_report():
    m = _block_matrix(within=0.9, across=0.1, sizes=(8, 8, 8))
    result = suggest_threshold(m)
    assert result["recommended_tau"] in DEFAULT_TAU_GRID
    assert set(result) == {"recommended_tau", "reason", "distribution", "table"}
    # Recommended tau must separate the within-cluster (0.9) from across (0.1).
    assert 0.1 < result["recommended_tau"] <= 0.9


def test_suggest_flags_too_diverse_corpus():
    # Everything barely similar: no threshold in the grid builds a real graph.
    m = np.full((10, 10), 0.1, dtype=np.float64)
    np.fill_diagonal(m, 1.0)
    result = suggest_threshold(m)
    assert "too diverse" in result["reason"]
    # Falls back to the most permissive (lowest) threshold on the grid.
    assert result["recommended_tau"] == min(DEFAULT_TAU_GRID)


def test_suggest_flags_over_dense_corpus():
    # Everyone almost identical: every threshold yields a saturated graph.
    m = np.full((10, 10), 0.95, dtype=np.float64)
    np.fill_diagonal(m, 1.0)
    result = suggest_threshold(m)
    assert "over-dense" in result["reason"]
    assert result["recommended_tau"] == max(DEFAULT_TAU_GRID)


# -- compute_similarity_matrix -----------------------------------------------
def test_compute_similarity_matrix_shape_and_symmetry():
    profiles, _ = generate_dataset(n_users=12, seed=3)
    matrix, node_ids = compute_similarity_matrix(profiles)
    assert matrix.shape == (12, 12)
    assert node_ids == [p.user_id for p in profiles]
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-6)


# -- tune_threshold_supervised -----------------------------------------------
def test_supervised_tuning_picks_best_map_on_clustered_data():
    profiles, ground_truth = generate_dataset(n_users=36, seed=5)
    result = tune_threshold_supervised(profiles, ground_truth, k=10)
    assert result["recommended_tau"] in DEFAULT_TAU_GRID
    assert result["metric"] == "MAP@10"
    # Every row carries both graph-health and its MAP score.
    assert all("map" in row for row in result["table"])
    # The recommended tau really is the argmax over the table.
    best_map = max(row["map"] for row in result["table"])
    assert result["best_map"] == pytest.approx(best_map)
    # Clustered synthetic data is separable -> a strong optimum is reachable.
    assert result["best_map"] > 0.8


def test_supervised_tuning_prefers_higher_tau_on_ties():
    profiles, ground_truth = generate_dataset(n_users=36, seed=5)
    result = tune_threshold_supervised(profiles, ground_truth, k=10)
    best_map = result["best_map"]
    tied = [row["tau"] for row in result["table"] if row["map"] == pytest.approx(best_map)]
    assert result["recommended_tau"] == max(tied)
