import numpy as np

from aphrodite.graph import build_graph


def test_threshold_creates_edges():
    m = np.array(
        [
            [1.0, 0.8, 0.5],
            [0.8, 1.0, 0.9],
            [0.5, 0.9, 1.0],
        ]
    )
    g = build_graph(m, node_ids=["a", "b", "c"], threshold=0.70)
    assert g.has_edge("a", "b")
    assert g.has_edge("b", "c")
    assert not g.has_edge("a", "c")  # 0.5 <= 0.70


def test_edge_weight_is_score():
    m = np.array([[1.0, 0.85], [0.85, 1.0]])
    g = build_graph(m, node_ids=["x", "y"], threshold=0.70)
    assert g["x"]["y"]["weight"] == 0.85


def test_no_self_loops():
    m = np.array([[1.0, 0.9], [0.9, 1.0]])
    g = build_graph(m, threshold=0.70)
    assert not any(u == v for u, v in g.edges())


def test_all_nodes_present_even_if_isolated():
    m = np.array([[1.0, 0.1], [0.1, 1.0]])
    g = build_graph(m, node_ids=["a", "b"], threshold=0.70)
    assert set(g.nodes()) == {"a", "b"}
    assert g.number_of_edges() == 0
