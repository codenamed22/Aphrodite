---
name: tau-calibration
description: |
  Calibrate or adjust the PPR matchmaking similarity threshold tau for the
  Aphrodite dating app. Use when connecting a NEW dataset, when the similarity
  graph looks empty or saturated, when PPR returns no/too-many matches, or when
  asked to "tune tau", "calibrate the threshold", "pick tau", "adjust the
  threshold", or "why are there no graph edges". Runs data-driven calibration
  (similarity distribution + graph health, and MAP@k tuning when ground truth
  exists) and recommends a tau to set on MatchmakingAlgorithm / build_graph.
---

# Tau (τ) Calibration

`tau` is the edge threshold in Algorithm 1 (paper §3.2.2): an edge is created
only when aggregated similarity `score > tau` (default **0.70**). It is
**dataset-dependent** — the same value that is perfect for one corpus can make
the graph empty or complete for another. This skill measures the data and picks
a `tau` empirically instead of guessing.

## When to use

- Onboarding a **new profile dataset** (different population, embeddings, or attributes).
- PPR returns **no matches** or **everyone matches everyone** → the graph is empty or saturated.
- `build_graph(...)` / `MatchmakingAlgorithm.fit(...)` produces **0 edges** or a near-complete graph.
- The user asks to **tune / calibrate / adjust tau**, or asks why there are no edges.
- After **switching embedding backend** (`lightweight` ↔ `word2vec_bert`) — the similarity scale shifts.

## Core idea (read before changing tau)

The right `tau` depends on the **similarity distribution** of the corpus:

| Corpus type | Distribution | Symptom at τ=0.70 | Fix |
|-------------|--------------|-------------------|-----|
| **Diverse** (unrelated users, e.g. broad public app) | left-shifted, max often < 0.66 | **0 edges**, PPR has nothing to rank | lower τ toward ~0.35, or use a **threshold-free** reciprocal method (`recon` / `multi_interest`) |
| **Homogeneous** (one college, shared vocabulary — the primary Aphrodite target) | right-shifted, within-community sims 0.85–0.98 | healthy graph (~23% density) | **keep τ ≈ 0.70–0.75** |
| **Tight synthetic clusters** | bimodal | healthy | τ ≈ 0.60–0.75 |

**Do not eyeball a percentile.** Judge the *graph* it produces: edge density
(target ~0.05–0.25), isolated-node fraction (want ~0), and — when ground truth
exists — MAP@k.

## How to run

Always run from the repo root with the project venv active
(`source .venv/bin/activate`). Two entry points:

### 1. CLI (fastest — use this first)

```bash
# A real dataset file: {"profiles": [...], "ground_truth": {...}}
python -m examples.calibrate_tau --data data/test_real.json

# Paper-faithful embeddings (downloads Word2Vec + BERT on first run; slower)
python -m examples.calibrate_tau --data data/test_real.json --backend word2vec_bert

# No file? Calibrate on synthetic clustered data
python -m examples.calibrate_tau --n 60 --with-gender
```

**IMPORTANT: calibrate on the SAME backend you deploy with.** `lightweight` and
`word2vec_bert` live on different similarity scales, so a `tau` tuned on one is
invalid for the other. If unsure which the app uses, calibrate both and report both.

### 2. Python API (for custom data or scripting)

```python
from aphrodite.calibration import (
    compute_similarity_matrix, suggest_threshold, tune_threshold_supervised,
)

# Unsupervised (no ground truth) — graph-health heuristic
matrix, ids = compute_similarity_matrix(profiles)              # backend=... to match deployment
report = suggest_threshold(matrix, ids)
print(report["recommended_tau"], report["reason"])

# Supervised (ground truth available) — authoritative, maximises MAP@k
result = tune_threshold_supervised(profiles, ground_truth, k=10)
print(result["recommended_tau"], result["best_map"])          # embeds ONCE, sweeps tau cheaply
```

## Interpreting the output & choosing tau

1. **If the dataset has ground truth → trust the SUPERVISED tau.** It sweeps the
   grid, runs full PPR at each value, and picks the `tau` maximising MAP@k (ties
   broken toward the higher, more discriminative `tau`). Prefer it over the
   unsupervised heuristic.
2. **No ground truth → use the graph-health heuristic**, then sanity-check the
   scan table yourself:
   - Pick the **highest** `tau` whose **density** is in ~[0.05, 0.25] **and**
     **isolated fraction** ≈ 0. Higher `tau` = more discriminative edges.
   - If even the lowest `tau` leaves most nodes isolated → corpus is **too
     diverse**: lower `tau` further, or switch to `recon` / `multi_interest`
     (both are threshold-free and immune to this problem).
   - If every `tau` is over-dense → corpus is **very homogeneous**: use the
     highest `tau`, and consider extending the grid past 0.80.
3. **Default for the Aphrodite target population** (one Indian college, shared
   vocabulary): keep **τ = 0.70** unless calibration on real data says otherwise.
   Empirically this keeps ~100% of within-community edges, drops ~99% of
   across-community edges, at a healthy ~23% density — it does **not** produce a
   near-complete graph.

## Applying the chosen tau

Set it wherever the matchmaker is constructed — do **not** edit
`DEFAULT_THRESHOLD` in `aphrodite/graph.py` unless the user wants a new global
default:

```python
from aphrodite import MatchmakingAlgorithm
algo = MatchmakingAlgorithm(threshold=0.70)   # <- calibrated tau
algo.fit(profiles)
```

CLI demo: `python -m examples.run_matchmaking --algorithm ppr --threshold <tau>`.

## Validate after changing tau

1. Re-run the calibrator and confirm the graph is connected and not saturated
   (`isolated ≈ 0`, `density` in band).
2. If ground truth exists, confirm MAP@k did not drop:
   `python -m examples.run_matchmaking --algorithm ppr --threshold <tau> --with-gender`.
3. Run the test suite: `python -m pytest -q` (calibration tests live in
   `tests/test_calibration.py`).

## Reference

- Tool: `aphrodite/calibration.py` — `similarity_distribution`, `graph_health`,
  `scan_thresholds`, `suggest_threshold`, `compute_similarity_matrix`,
  `tune_threshold_supervised`.
- CLI: `examples/calibrate_tau.py`.
- Threshold is applied in `aphrodite/graph.py::build_graph` and consumed by
  `aphrodite/matchmaker.py::MatchmakingAlgorithm(threshold=...)`.
- Threshold-free alternatives: `aphrodite/reciprocal.py`
  (`ReciprocalRecommender(method="recon"|"multi_interest")`).
