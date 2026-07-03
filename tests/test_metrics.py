from aphrodite.metrics import (
    average_precision_at_k,
    evaluate_at_ks,
    f1_at_k,
    mean_average_precision_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(ranked, relevant, 2) == 0.5  # a relevant, b not
    assert precision_at_k(ranked, relevant, 4) == 0.5


def test_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c", "e"}
    # top-4 contains a and c -> 2 of 3 relevant
    assert abs(recall_at_k(ranked, relevant, 4) - 2 / 3) < 1e-9


def test_f1_at_k():
    ranked = ["a", "b"]
    relevant = {"a", "c"}
    p = precision_at_k(ranked, relevant, 2)  # 0.5
    r = recall_at_k(ranked, relevant, 2)  # 0.5
    assert abs(f1_at_k(ranked, relevant, 2) - (2 * p * r / (p + r))) < 1e-9


def test_f1_zero_when_no_hits():
    assert f1_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_average_precision_perfect():
    # all relevant items ranked first -> AP = 1.0
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "b"}
    assert abs(average_precision_at_k(ranked, relevant, 4) - 1.0) < 1e-9


def test_average_precision_known_value():
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.8333...
    ranked = ["a", "x", "b", "y"]
    relevant = {"a", "b"}
    assert abs(average_precision_at_k(ranked, relevant, 4) - (1.0 + 2 / 3) / 2) < 1e-9


def test_map_averages_users():
    lists = [["a", "b"], ["c", "d"]]
    rels = [{"a"}, {"d"}]
    ap1 = average_precision_at_k(lists[0], rels[0], 2)  # 1.0
    ap2 = average_precision_at_k(lists[1], rels[1], 2)  # 0.5
    assert abs(mean_average_precision_at_k(lists, rels, 2) - (ap1 + ap2) / 2) < 1e-9


def test_evaluate_at_ks_keys():
    lists = [["a", "b", "c"], ["b", "a", "c"]]
    rels = [{"a"}, {"b"}]
    out = evaluate_at_ks(lists, rels, ks=(1, 2))
    for key in ("P@1", "R@1", "F1@1", "MAP@1", "P@2", "R@2", "F1@2", "MAP@2"):
        assert key in out
    assert out["P@1"] == 1.0  # both users have their relevant item at rank 1
