"""Head-to-head comparison: Phase-1 PPR vs Phase-2 reciprocal recommenders.

Runs the three algorithms on one shared, gender-aware synthetic dataset and
prints Precision/Recall/F1/MAP at k plus the reciprocity rate, so the tradeoffs
are directly comparable:

* ``ppr``            — Phase 1, Personalized PageRank over a similarity graph.
* ``recon``          — Phase 2a, RECON-style directional + harmonic-mean.
* ``multi_interest`` — Phase 2b, multi-interest facet coverage + harmonic-mean.

Usage::

    python -m examples.compare_algorithms
    python -m examples.compare_algorithms --n 120 --seed 7
"""

from __future__ import annotations

import argparse

from aphrodite import MatchmakingAlgorithm, ReciprocalRecommender
from aphrodite.datasets import generate_dataset
from aphrodite.metrics import evaluate_at_ks, reciprocity_rate


def _evaluate(recommend_all, profiles, ground_truth, ks):
    recs = {p.user_id: recommend_all[p.user_id] for p in profiles}
    ranked = [recs[p.user_id] for p in profiles]
    relevant = [ground_truth[p.user_id] for p in profiles]
    metrics = evaluate_at_ks(ranked, relevant, ks=ks)
    metrics["Recip@10"] = reciprocity_rate(recs, k=10)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 1 vs Phase 2 matchmakers.")
    parser.add_argument("--n", type=int, default=90, help="number of synthetic users")
    parser.add_argument("--seed", type=int, default=42, help="dataset RNG seed")
    parser.add_argument("--threshold", type=float, default=0.70, help="PPR edge threshold tau")
    parser.add_argument("--damping", type=float, default=0.85, help="PPR damping d")
    parser.add_argument("--interests", type=int, default=3, help="facets per user (multi_interest)")
    args = parser.parse_args()

    ks = (5, 10, 15, 20)
    profiles, ground_truth = generate_dataset(
        n_users=args.n, seed=args.seed, with_gender=True
    )

    ppr = MatchmakingAlgorithm(
        threshold=args.threshold, damping=args.damping, apply_gender_filter=True
    ).fit(profiles)
    recon = ReciprocalRecommender(method="recon", apply_gender_filter=True).fit(profiles)
    multi = ReciprocalRecommender(
        method="multi_interest", n_interests=args.interests, apply_gender_filter=True
    ).fit(profiles)

    algos = {
        "ppr": ppr.recommend_all(k=max(ks)),
        "recon": recon.recommend_all(k=max(ks)),
        "multi_interest": multi.recommend_all(k=max(ks)),
    }
    results = {
        name: _evaluate(recs, profiles, ground_truth, ks) for name, recs in algos.items()
    }

    print(f"Users: {len(profiles)}  |  dating mode (gender-filtered)  |  seed={args.seed}\n")
    cols = [f"P@{ks[0]}", f"R@{ks[1]}", f"F1@{ks[1]}", f"MAP@{ks[1]}", f"MAP@{ks[-1]}", "Recip@10"]
    print(f"{'algorithm':<16}" + "".join(f"{c:>10}" for c in cols))
    for name, m in results.items():
        row = "".join(f"{m[c]:>10.3f}" for c in cols)
        print(f"{name:<16}{row}")

    best_recip = max(results, key=lambda n: results[n]["Recip@10"])
    print(
        f"\nReciprocity: {best_recip} is most bilaterally consistent "
        f"({results[best_recip]['Recip@10']:.3f}) vs ppr ({results['ppr']['Recip@10']:.3f})."
    )


if __name__ == "__main__":
    main()
