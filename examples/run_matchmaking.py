"""End-to-end matchmaking demo (Phase 1 PPR or Phase 2 reciprocal).

Generates a synthetic clustered dataset, runs the chosen algorithm, and reports
Precision/Recall/F1/MAP at k in {5, 10, 15, 20} plus the reciprocity rate.

Usage::

    python -m examples.run_matchmaking                          # Phase 1 PPR
    python -m examples.run_matchmaking --algorithm recon --with-gender
    python -m examples.run_matchmaking --algorithm multi_interest --n 120
"""

from __future__ import annotations

import argparse

from aphrodite import MatchmakingAlgorithm, ReciprocalRecommender
from aphrodite.datasets import generate_dataset
from aphrodite.metrics import evaluate_at_ks, reciprocity_rate


def _build(algorithm: str, args, apply_gender_filter: bool):
    """Construct and return the requested (unfitted) recommender."""
    if algorithm == "ppr":
        return MatchmakingAlgorithm(
            threshold=args.threshold,
            damping=args.damping,
            apply_gender_filter=apply_gender_filter,
        )
    return ReciprocalRecommender(
        method=algorithm, apply_gender_filter=apply_gender_filter
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a matchmaking demo.")
    parser.add_argument("--n", type=int, default=90, help="number of synthetic users")
    parser.add_argument("--algorithm", choices=("ppr", "recon", "multi_interest"),
                        default="ppr", help="matchmaking algorithm (Phase 1 ppr or Phase 2)")
    parser.add_argument("--threshold", type=float, default=0.70, help="edge threshold tau (ppr)")
    parser.add_argument("--damping", type=float, default=0.85, help="PPR damping d (ppr)")
    parser.add_argument("--seed", type=int, default=42, help="dataset RNG seed")
    parser.add_argument("--target", type=str, default=None, help="show matches for one user")
    parser.add_argument("--with-gender", action="store_true",
                        help="assign genders/preferences and filter matches (dating mode)")
    args = parser.parse_args()

    profiles, ground_truth = generate_dataset(
        n_users=args.n, seed=args.seed, with_gender=args.with_gender
    )
    algo = _build(args.algorithm, args, apply_gender_filter=args.with_gender)
    algo.fit(profiles)
    by_id = {p.user_id: p for p in profiles}

    ks = (5, 10, 15, 20)
    ranked_lists = []
    relevant_sets = []
    recs = {}
    for p in profiles:
        r = algo.recommend(p.user_id, k=max(ks))
        recs[p.user_id] = r
        ranked_lists.append(r)
        relevant_sets.append(ground_truth[p.user_id])

    metrics = evaluate_at_ks(ranked_lists, relevant_sets, ks=ks)

    mode = "dating (gender-filtered)" if args.with_gender else "paper-faithful"
    edges = getattr(getattr(algo, "graph_", None), "number_of_edges", lambda: None)()
    edge_info = f"graph edges: {edges}  |  " if edges is not None else ""
    print(f"Algorithm: {args.algorithm}  |  users: {len(profiles)}  |  {edge_info}"
          f"reciprocity@10: {reciprocity_rate(recs, 10):.3f}  |  mode: {mode}\n")
    header = "        " + "".join(f"@{k:<8}" for k in ks)
    print(header)
    for metric in ("P", "R", "F1", "MAP"):
        row = "".join(f"{metrics[f'{metric}@{k}']:<9.3f}" for k in ks)
        print(f"{metric:<6}  {row}")

    if args.target:
        t = by_id[args.target]
        matches = algo.recommend(args.target, k=10)
        if args.with_gender:
            print(f"\nTarget {args.target}: {t.gender}, seeking {sorted(t.seeking)}")
            print("Top matches:")
            for r in matches:
                m = by_id[r]
                print(f"  {r}  ({m.gender}, seeking {sorted(m.seeking)})")
        else:
            print(f"\nTop matches for {args.target}: {matches}")
        print(f"Ground-truth matches: {sorted(ground_truth[args.target])}")


if __name__ == "__main__":
    main()
