import tempfile
from pathlib import Path

from aphrodite.datasets import generate_dataset, load_dataset, save_dataset


def test_generate_shapes():
    profiles, gt = generate_dataset(n_users=30, seed=7)
    assert len(profiles) == 30
    assert set(gt.keys()) == {p.user_id for p in profiles}


def test_ground_truth_excludes_self():
    profiles, gt = generate_dataset(n_users=18, seed=8)
    for uid, matches in gt.items():
        assert uid not in matches


def test_ground_truth_symmetry():
    profiles, gt = generate_dataset(n_users=18, seed=9)
    for uid, matches in gt.items():
        for other in matches:
            assert uid in gt[other]


def test_reproducible():
    p1, g1 = generate_dataset(n_users=15, seed=123)
    p2, g2 = generate_dataset(n_users=15, seed=123)
    assert [p.as_dict() for p in p1] == [p.as_dict() for p in p2]
    assert g1 == g2


def test_save_and_load_roundtrip():
    profiles, gt = generate_dataset(n_users=10, seed=5)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.json"
        save_dataset(path, profiles, gt)
        loaded_profiles, loaded_gt = load_dataset(path)
    assert [p.as_dict() for p in loaded_profiles] == [p.as_dict() for p in profiles]
    assert loaded_gt == gt
