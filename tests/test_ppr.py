import networkx as nx
import numpy as np

from aphrodite.ppr import personalized_pagerank, rank_matches, transition_matrix


def _chain_graph():
    g = nx.Graph()
    g.add_edge("a", "b", weight=0.9)
    g.add_edge("b", "c", weight=0.9)
    g.add_edge("c", "d", weight=0.9)
    return g


def test_transition_matrix_column_stochastic():
    g = _chain_graph()
    nodes = list(g.nodes())
    m = transition_matrix(g, nodes)
    col_sums = m.sum(axis=0)
    # every node here has at least one edge, so columns sum to 1
    np.testing.assert_allclose(col_sums, np.ones(len(nodes)))


def test_scores_sum_to_one():
    g = _chain_graph()
    scores = personalized_pagerank(g, "a")
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_closer_nodes_score_higher():
    g = _chain_graph()
    scores = personalized_pagerank(g, "a")
    assert scores["b"] > scores["c"] > scores["d"]


def test_target_has_high_score():
    g = _chain_graph()
    scores = personalized_pagerank(g, "a")
    assert scores["a"] == max(scores.values())


def test_rank_matches_excludes_target():
    scores = {"a": 0.5, "b": 0.3, "c": 0.2}
    ranked = rank_matches(scores, target="a", k=10)
    assert "a" not in ranked
    assert ranked == ["b", "c"]


def test_rank_matches_top_k():
    scores = {"a": 0.5, "b": 0.3, "c": 0.2, "d": 0.1}
    ranked = rank_matches(scores, target="a", k=2)
    assert ranked == ["b", "c"]


def test_unknown_target_raises():
    g = _chain_graph()
    try:
        personalized_pagerank(g, "zzz")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_isolated_target():
    g = nx.Graph()
    g.add_node("lonely")
    g.add_edge("x", "y", weight=0.9)
    scores = personalized_pagerank(g, "lonely")
    # all restart mass stays on the target
    assert scores["lonely"] == max(scores.values())
