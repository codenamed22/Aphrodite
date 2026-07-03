import pytest

from aphrodite import MatchmakingAlgorithm, UserProfile
from aphrodite.datasets import generate_dataset
from aphrodite.metrics import evaluate_at_ks


def test_fit_builds_graph_and_matrix():
    profiles, _ = generate_dataset(n_users=12, seed=1)
    algo = MatchmakingAlgorithm().fit(profiles)
    assert algo.similarity_matrix_.shape == (12, 12)
    assert algo.graph_.number_of_nodes() == 12


def test_recommend_excludes_self_and_respects_k():
    profiles, _ = generate_dataset(n_users=20, seed=2)
    algo = MatchmakingAlgorithm().fit(profiles)
    target = profiles[0].user_id
    matches = algo.recommend(target, k=5)
    assert target not in matches
    assert len(matches) <= 5


def test_recommend_before_fit_raises():
    algo = MatchmakingAlgorithm()
    with pytest.raises(RuntimeError):
        algo.recommend("u000")


def test_unknown_target_raises():
    profiles, _ = generate_dataset(n_users=6, seed=3)
    algo = MatchmakingAlgorithm().fit(profiles)
    with pytest.raises(KeyError):
        algo.recommend("does-not-exist")


def test_same_cluster_users_matched():
    # Two clearly separated clusters via shared vocabulary.
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
    algo = MatchmakingAlgorithm(threshold=0.5).fit(profiles)
    assert algo.recommend("t1", k=1) == ["t2"]
    assert algo.recommend("s1", k=1) == ["s2"]


def test_end_to_end_metrics_reasonable():
    profiles, gt = generate_dataset(n_users=60, seed=42)
    algo = MatchmakingAlgorithm().fit(profiles)
    ranked = [algo.recommend(p.user_id, k=20) for p in profiles]
    rel = [gt[p.user_id] for p in profiles]
    out = evaluate_at_ks(ranked, rel, ks=(5, 10))
    # The clustered structure should be recovered well above random.
    assert out["P@5"] > 0.4
    assert out["MAP@10"] > 0.2
