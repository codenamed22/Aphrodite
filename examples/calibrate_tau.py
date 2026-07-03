"""Calibrate the PPR edge threshold ``tau`` for a dataset (paper §3.2.2).

Measures the pairwise-similarity distribution of a corpus and the resulting
graph health across candidate thresholds, then recommends a ``tau``. When the
dataset ships ground-truth matches, it *also* runs supervised tuning: sweeping
``tau``, running the full PPR matchmaker, and picking the value that maximises
MAP@k.

Usage::

    # Synthetic clustered data (paper-style), lightweight offline backend
    python -m examples.calibrate_tau

    # A real dataset file ({"profiles": [...], "ground_truth": {...}})
    python -m examples.calibrate_tau --data data/test_real.json

    # Paper-faithful Word2Vec + BERT embeddings (downloads models on first run)
    python -m examples.calibrate_tau --data data/test_real.json --backend word2vec_bert

Interpreting the output:

* ``density``  — target a healthy band (~0.05-0.25). Near 0 means ``tau`` is too
  high (empty graph); near 1 means it is too low (everyone connected).
* ``isolated`` — users with no edges; PPR can never reach them from anyone else.
* ``MAP@k``    — only shown with ground truth; the authoritative signal. Pick the
  highest ``tau`` whose MAP is at (or near) the maximum.
"""

from __future__ import annotations

import argparse

from aphrodite.calibration import (
    compute_similarity_matrix,
    suggest_threshold,
    tune_threshold_supervised,
)
from aphrodite.datasets import generate_dataset, load_dataset


def _make_backend(name: str):
    if name == "lightweight":
        return None  # MatchmakingAlgorithm defaults to LightweightBackend
    from aphrodite.embeddings import Word2VecBertBackend

    return Word2VecBertBackend()


def _print_distribution(dist: dict) -> None:
    print("Pairwise similarity distribution "
          f"({int(dist['count'])} pairs):")
    print(f"  min={dist['min']:.3f}  mean={dist['mean']:.3f}  "
          f"max={dist['max']:.3f}  std={dist['std']:.3f}")
    pcts = [k for k in dist if k.startswith("p")]
    pcts.sort(key=lambda s: int(s[1:]))
    cells = "  ".join(f"{k}={dist[k]:.3f}" for k in pcts)
    print(f"  {cells}\n")


def _print_scan(table: list, supervised: bool) -> None:
    if supervised:
        print(f"{'tau':>5}  {'edges':>6}  {'density':>7}  {'mean_deg':>8}  "
              f"{'isolated':>8}  {'lcc':>5}  {'MAP':>6}")
    else:
        print(f"{'tau':>5}  {'edges':>6}  {'density':>7}  {'mean_deg':>8}  "
              f"{'isolated':>8}  {'lcc':>5}")
    for r in table:
        base = (f"{r['tau']:>5.2f}  {r['edges']:>6d}  {r['density']:>7.3f}  "
                f"{r['mean_degree']:>8.2f}  {r['isolated_fraction']:>8.3f}  "
                f"{r['largest_component_fraction']:>5.2f}")
        if supervised:
            base += f"  {r['map']:>6.3f}"
        print(base)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the PPR edge threshold tau for a dataset."
    )
    parser.add_argument("--data", type=str, default=None,
                        help="dataset JSON ({profiles, ground_truth}); if omitted, "
                             "a synthetic clustered dataset is generated")
    parser.add_argument("--backend", choices=("lightweight", "word2vec_bert"),
                        default="lightweight",
                        help="embedding backend (default: offline lightweight)")
    parser.add_argument("--n", type=int, default=60,
                        help="number of synthetic users when --data is omitted")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the synthetic dataset")
    parser.add_argument("--with-gender", action="store_true",
                        help="generate gendered synthetic data (when --data omitted)")
    parser.add_argument("--k", type=int, default=10,
                        help="k for supervised MAP@k tuning")
    parser.add_argument("--min-density", type=float, default=0.05,
                        help="lower bound of the healthy edge-density band")
    parser.add_argument("--max-density", type=float, default=0.25,
                        help="upper bound of the healthy edge-density band")
    parser.add_argument("--max-isolated", type=float, default=0.10,
                        help="max acceptable fraction of isolated (edgeless) users")
    args = parser.parse_args()

    if args.data:
        profiles, ground_truth = load_dataset(args.data)
        source = args.data
    else:
        profiles, ground_truth = generate_dataset(
            n_users=args.n, seed=args.seed, with_gender=args.with_gender
        )
        source = f"synthetic (n={args.n}, seed={args.seed})"

    has_ground_truth = any(len(v) > 0 for v in ground_truth.values())
    backend = _make_backend(args.backend)

    print(f"Dataset: {source}  |  users: {len(profiles)}  |  "
          f"backend: {args.backend}  |  "
          f"ground truth: {'yes' if has_ground_truth else 'no'}\n")

    matrix, node_ids = compute_similarity_matrix(profiles, backend=backend)

    unsup = suggest_threshold(
        matrix, node_ids,
        min_density=args.min_density,
        max_density=args.max_density,
        max_isolated_fraction=args.max_isolated,
    )
    _print_distribution(unsup["distribution"])

    if has_ground_truth:
        sup = tune_threshold_supervised(
            profiles, ground_truth, backend=backend, k=args.k
        )
        _print_scan(sup["table"], supervised=True)
        print(f"Unsupervised recommendation: tau = {unsup['recommended_tau']:.2f}")
        print(f"  ({unsup['reason']})")
        print(f"Supervised recommendation:   tau = {sup['recommended_tau']:.2f}  "
              f"(maximises {sup['metric']} = {sup['best_map']:.3f})")
        print("\n>>> Use the SUPERVISED tau when ground truth is trustworthy; "
              "it directly optimises ranking quality.")
    else:
        _print_scan(unsup["table"], supervised=False)
        print(f"Recommended tau = {unsup['recommended_tau']:.2f}")
        print(f"  ({unsup['reason']})")
        print("\n>>> No ground truth: this is a graph-health heuristic. Confirm the "
              "recommended tau yields a connected, non-saturated graph above.")


if __name__ == "__main__":
    main()
