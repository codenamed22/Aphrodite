"""Head-to-head benchmark harness for Aphrodite reciprocal matchmakers.

Compares the Phase-1/Phase-2 baselines against the Tier-1 research methods from
the deep-dive (see ``plan.md``) across *quality*, *reciprocity* and *congestion*
metrics, on two synthetic regimes and (optionally) both embedding backends:

* **Algorithms**
    - ``ppr``            Phase-1 Personalized PageRank over a similarity graph.
    - ``recon``          Phase-2a RECON directional + harmonic mean (baseline).
    - ``multi_interest`` Phase-2b multi-interest facet coverage.
    - ``recon+tu``       RECON scores reranked by the congestion-aware TU/IPFP
                         (Choo-Siow) equilibrium, arXiv:2306.09060 (the flagship
                         method under test).

* **Regimes** (the central hypothesis is about corpus *homogeneity*)
    - ``diverse``      all interest themes → clusters separate easily.
    - ``homogeneous``  few themes → everyone looks similar → congestion, which is
                       our real single-college-in-India target setting.

* **Metrics** (k = 10 unless noted)
    - quality:      MAP@10, P@5, R@10, F1@10
    - reciprocity:  Recip@10, Mutual@10 (absolute mutual pairs),
                    bilateral coverage/stability recall of the ground truth.
    - congestion:   Gini@10 (exposure inequality, lower is better),
                    Cov@10 (catalog coverage), Tail@10 (long-tail coverage),
                    Ent@10 (exposure entropy, 1.0 = perfectly spread).

A second table sweeps the pluggable aggregation operators (unit-normalized so
the operator effect is isolated) to see which combiner maximizes reciprocity.

Hypothesis: TU/IPFP raises Mutual@10 and cuts Gini@10 at roughly equal MAP@10,
with the gap widening on the homogeneous corpus.

Usage::

    python -m examples.benchmark
    python -m examples.benchmark --n 120 --seed 7 --beta 0.5
    python -m examples.benchmark --backend both   # also run word2vec+bert
"""

from __future__ import annotations

import argparse

from aphrodite import (
    MatchmakingAlgorithm,
    ReciprocalRecommender,
    TUMatchRecommender,
    AsymmetricRecommender,
    FairRecReranker,
    NSWReranker,
    LightweightBackend,
)
from aphrodite.datasets import generate_dataset, THEMES
from aphrodite.metrics import (
    evaluate_at_ks,
    reciprocity_rate,
    total_mutual_matches,
    gini_exposure,
    coverage_at_k,
    long_tail_coverage,
    exposure_entropy,
    bilateral_recall_at_k,
)
from aphrodite.reciprocal import AGGREGATIONS


REGIMES = {
    # (themes, noise): homogeneous = few overlapping themes -> congestion.
    "diverse": (THEMES, 0.15),
    "homogeneous": (THEMES[:2], 0.30),
}

# Columns rendered in the main comparison table, in order.
MAIN_COLS = [
    "MAP@10",
    "P@5",
    "R@10",
    "F1@10",
    "Recip@10",
    "Mutual@10",
    "CRec@10",
    "SRec@10",
    "Gini@10",
    "Cov@10",
    "Tail@10",
    "Ent@10",
]


def _all_metrics(recs, profiles, ground_truth, ks, all_users):
    """Compute quality + reciprocity + congestion metrics for one algorithm."""
    ranked = [recs[p.user_id] for p in profiles]
    relevant = [ground_truth[p.user_id] for p in profiles]
    m = evaluate_at_ks(ranked, relevant, ks=ks)
    m["Recip@10"] = reciprocity_rate(recs, k=10)
    m["Mutual@10"] = float(total_mutual_matches(recs, k=10))
    m["Gini@10"] = gini_exposure(recs, k=10, all_users=all_users)
    m["Cov@10"] = coverage_at_k(recs, k=10, all_users=all_users)
    m["Tail@10"] = long_tail_coverage(recs, k=10, all_users=all_users)
    m["Ent@10"] = exposure_entropy(recs, k=10, all_users=all_users)
    bilat = bilateral_recall_at_k(recs, ground_truth, k=10)
    m["CRec@10"] = bilat["coverage_recall"]
    m["SRec@10"] = bilat["stability_recall"]
    return m


def _make_backend(kind):
    if kind == "lightweight":
        return LightweightBackend()
    from aphrodite.embeddings import Word2VecBertBackend  # lazy, heavy

    return Word2VecBertBackend()


def _run_regime(name, themes, noise, backend, args, ks):
    profiles, ground_truth = generate_dataset(
        n_users=args.n,
        themes=themes,
        noise=noise,
        seed=args.seed,
        with_gender=True,
    )
    all_users = {p.user_id for p in profiles}
    kmax = max(ks)

    ppr = MatchmakingAlgorithm(
        threshold=args.threshold,
        damping=args.damping,
        backend=backend,
        apply_gender_filter=True,
    ).fit(profiles)
    recon = ReciprocalRecommender(
        method="recon", backend=backend, apply_gender_filter=True
    ).fit(profiles)
    multi = ReciprocalRecommender(
        method="multi_interest",
        n_interests=args.interests,
        backend=backend,
        apply_gender_filter=True,
    ).fit(profiles)
    asym = AsymmetricRecommender(
        backend=backend, homophily=args.homophily, apply_gender_filter=True
    ).fit(profiles)
    tu = TUMatchRecommender(recon, beta=args.beta, n_iter=args.tu_iters)
    fair = FairRecReranker(recon, alpha=args.alpha)
    nsw = NSWReranker(recon)

    algos = {
        "ppr": ppr.recommend_all(k=kmax),
        "recon": recon.recommend_all(k=kmax),
        "multi_interest": multi.recommend_all(k=kmax),
        "asym(self/pref)": asym.recommend_all(k=kmax),
        "recon+tu": tu.recommend_all(k=kmax),
        "recon+fairrec": fair.recommend_all(k=kmax),
        "recon+nsw": nsw.recommend_all(k=kmax),
    }
    results = {
        a: _all_metrics(recs, profiles, ground_truth, ks, all_users)
        for a, recs in algos.items()
    }

    print(
        f"\n### regime={name}  backend={args.backend_label}  "
        f"users={len(profiles)}  themes={len(themes)}  seed={args.seed}"
    )
    header = f"{'algorithm':<16}" + "".join(f"{c:>10}" for c in MAIN_COLS)
    print(header)
    print("-" * len(header))
    for a, m in results.items():
        row = "".join(f"{m[c]:>10.3f}" for c in MAIN_COLS)
        print(f"{a:<16}{row}")

    _hypothesis_line(results)
    _aggregation_sweep(profiles, ground_truth, all_users, backend, ks)
    return results


def _hypothesis_line(results):
    base, tu = results["recon"], results["recon+tu"]
    d_mut = tu["Mutual@10"] - base["Mutual@10"]
    d_gini = tu["Gini@10"] - base["Gini@10"]
    d_map = tu["MAP@10"] - base["MAP@10"]
    verdict = (
        "SUPPORTS" if (d_mut >= 0 and d_gini <= 0) else "does NOT support"
    )
    print(
        f"  -> TU vs RECON: Mutual {d_mut:+.1f}, Gini {d_gini:+.3f}, "
        f"MAP {d_map:+.3f}  [{verdict} the congestion hypothesis]"
    )
    # Congestion-reranker ablation: which mechanism best trades MAP for fairness?
    ablation = {
        "recon+tu": results["recon+tu"],
        "recon+fairrec": results["recon+fairrec"],
        "recon+nsw": results["recon+nsw"],
    }
    best = min(ablation, key=lambda a: ablation[a]["Gini@10"])
    print(
        "  -> congestion rerankers (Gini | MAP | Recip): "
        + "  ".join(
            f"{a.split('+')[1]}={m['Gini@10']:.3f}/{m['MAP@10']:.3f}/{m['Recip@10']:.3f}"
            for a, m in ablation.items()
        )
        + f"  [fairest exposure: {best.split('+')[1]}]"
    )


def _aggregation_sweep(profiles, ground_truth, all_users, backend, ks):
    """Isolate the aggregation operator (all unit-normalized) on RECON."""
    print(f"  aggregation sweep (recon, unit-normalized):")
    cols = ["MAP@10", "Recip@10", "Mutual@10", "Gini@10"]
    print(f"    {'operator':<12}" + "".join(f"{c:>10}" for c in cols))
    for op in AGGREGATIONS:
        rec = ReciprocalRecommender(
            method="recon",
            aggregation=op,
            score_normalizer="unit",
            backend=backend,
            apply_gender_filter=True,
        ).fit(profiles)
        recs = rec.recommend_all(k=max(ks))
        m = _all_metrics(recs, profiles, ground_truth, ks, all_users)
        row = "".join(f"{m[c]:>10.3f}" for c in cols)
        print(f"    {op:<12}{row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aphrodite benchmark harness.")
    parser.add_argument("--n", type=int, default=90, help="number of users")
    parser.add_argument("--seed", type=int, default=42, help="dataset RNG seed")
    parser.add_argument("--threshold", type=float, default=0.70, help="PPR tau")
    parser.add_argument("--damping", type=float, default=0.85, help="PPR damping")
    parser.add_argument("--interests", type=int, default=3, help="facets/user")
    parser.add_argument("--beta", type=float, default=1.0, help="TU temperature")
    parser.add_argument("--tu-iters", type=int, default=50, help="IPFP iterations")
    parser.add_argument(
        "--homophily", type=float, default=0.5, help="asym self/pref blend (1=symmetric)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="FairRec exposure-floor fraction"
    )
    parser.add_argument(
        "--backend",
        choices=["lightweight", "word2vec_bert", "both"],
        default="lightweight",
        help="embedding backend(s) to run",
    )
    args = parser.parse_args()

    ks = (5, 10, 15, 20)
    backend_kinds = (
        ["lightweight", "word2vec_bert"]
        if args.backend == "both"
        else [args.backend]
    )

    print("Aphrodite benchmark — quality vs reciprocity vs congestion")
    print("Gini/lower = fairer exposure; Ent/higher = more spread; Mutual = raw pairs.")

    for kind in backend_kinds:
        try:
            backend = _make_backend(kind)
        except Exception as exc:  # noqa: BLE001 - heavy backend optional
            print(f"\n[skip] backend '{kind}' unavailable: {exc}")
            continue
        args.backend_label = kind
        for name, (themes, noise) in REGIMES.items():
            _run_regime(name, themes, noise, backend, args, ks)


if __name__ == "__main__":
    main()
