"""End-to-end demo of Algorithm 1 (Personalized PageRank matchmaking).

Generates a synthetic clustered dataset, runs the matchmaking algorithm, and
reports Precision/Recall/F1/MAP at k in {5, 10, 15, 20}.

Usage::

    python -m examples.run_matchmaking            # offline lightweight backend
    python -m examples.run_matchmaking --n 120    # larger dataset
"""

from __future__ import annotations

import argparse

from aphrodite import MatchmakingAlgorithm
from aphrodite.datasets import generate_dataset
from aphrodite.metrics import evaluate_at_ks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Algorithm 1 matchmaking demo.")
    parser.add_argument("--n", type=int, default=90, help="number of synthetic users")
    parser.add_argument("--threshold", type=float, default=0.70, help="edge threshold tau")
    parser.add_argument("--damping", type=float, default=0.85, help="PPR damping d")
    parser.add_argument("--seed", type=int, default=42, help="dataset RNG seed")
    parser.add_argument("--target", type=str, default=None, help="show matches for one user")
    args = parser.parse_args()

    profiles, ground_truth = generate_dataset(n_users=args.n, seed=args.seed)
    algo = MatchmakingAlgorithm(threshold=args.threshold, damping=args.damping)
    algo.fit(profiles)

    ks = (5, 10, 15, 20)
    ranked_lists = []
    relevant_sets = []
    for p in profiles:
        ranked_lists.append(algo.recommend(p.user_id, k=max(ks)))
        relevant_sets.append(ground_truth[p.user_id])

    metrics = evaluate_at_ks(ranked_lists, relevant_sets, ks=ks)

    print(f"Users: {len(profiles)}  |  graph edges: {algo.graph_.number_of_edges()}  "
          f"|  tau={args.threshold}  d={args.damping}\n")
    header = "        " + "".join(f"@{k:<8}" for k in ks)
    print(header)
    for metric in ("P", "R", "F1", "MAP"):
        row = "".join(f"{metrics[f'{metric}@{k}']:<9.3f}" for k in ks)
        print(f"{metric:<6}  {row}")

    if args.target:
        matches = algo.recommend(args.target, k=10)
        print(f"\nTop matches for {args.target}: {matches}")
        print(f"Ground-truth matches: {sorted(ground_truth[args.target])}")


if __name__ == "__main__":
    main()
