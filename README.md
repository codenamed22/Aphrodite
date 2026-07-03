# Aphrodite 💘

A dating-app **matchmaking service** built on peer-reviewed research.

The project is delivered in phases:

| Phase | Algorithm | Paper | Status |
|-------|-----------|-------|--------|
| **1** | Personalized-PageRank matchmaking | *Enhancing a User Matchmaking Algorithm using Personalized PageRank* — Thaiprayoon & Unger, **NLPIR 2023** | ✅ implemented |
| **2** | Reciprocal recommendation (RECON + multi-interest) | *RECON: a reciprocal recommender for online dating* — Pizzato et al., **RecSys 2010**; multi-interest idea from *MINER*, Li et al., **ACL 2022 Findings** | ✅ implemented |

---

## Phase 1 — Algorithm 1 (Personalized PageRank)

Faithful implementation of the paper's three-stage pipeline:

1. **Text representation** (§3.1) — preprocess each profile attribute, then embed:
   * *term-based* attributes (interests, hobbies, occupation) → mean of word
     vectors, OOV ignored (Eq. 1);
   * *context-based* attribute (biography) → a single sentence embedding (Eq. 2).
2. **Graph representation** (§3.2) — cosine similarity per attribute (Eq. 3),
   averaged into an aggregated score (Eq. 4) → an `N×N` matrix → a weighted graph
   where users are connected when the score exceeds `τ = 0.70`, with the edge
   weight equal to the score (Eq. 5).
3. **User matchmaking** (§3.3) — **Personalized PageRank** (Eq. 6),
   `r' = (1−d)·M·r + d·v`, with `d = 0.85` and the personalization vector `v`
   centred on the target user; rank users by descending score and return top-`k`.

Evaluation uses **P@k, R@k, F1@k** (Eq. 7–9) and **MAP@k** (Eq. 10–12) for
`k ∈ {5, 10, 15, 20}`.

### Pluggable embeddings

Embeddings sit behind one interface (`EmbeddingBackend`) with two backends:

* **`LightweightBackend`** (default) — deterministic hashing embeddings; fully
  offline, no downloads. Ideal for tests and quick iteration.
* **`Word2VecBertBackend`** — paper-faithful Word2Vec (GoogleNews 300d) + BERT
  sentence embeddings. Install the heavy extras: `pip install -e '.[full]'`.

### Gender & preferences (dating-app extension)

The paper is a general profile-matching algorithm with no notion of gender. For
dating use, `UserProfile` adds two fields (never embedded):

* `gender` — the user's gender identity (e.g. `"man"`, `"woman"`, `"non-binary"`).
* `seeking` — a set of genders the user wants to match with; empty = open to all.

When `apply_gender_filter=True` (the default), the similarity graph is still
built **gender-agnostically** — so Personalized PageRank can discover a great
match *through* a same-gender intermediary (high-order relationship) — and only
the **final ranked list** is restricted to users who are *mutually* compatible
(`a.is_compatible_with(b)`). Set `apply_gender_filter=False` for the original
paper behaviour.

```python
from aphrodite import MatchmakingAlgorithm
from aphrodite.datasets import generate_dataset

profiles, gt = generate_dataset(n_users=90, seed=42, with_gender=True)
algo = MatchmakingAlgorithm(apply_gender_filter=True).fit(profiles)
algo.recommend("u000", k=10)   # only gender-compatible users returned
```

---

## Phase 2 — Reciprocal matchmaking

Phase 1 scores compatibility **symmetrically** (`sim(A, B) == sim(B, A)`). Real
dating is **reciprocal**: how much A likes B is not how much B likes A, and a
good match needs *mutual* interest. Phase 2 adds two content-based reciprocal
recommenders that run on the *same* profile data (no interaction logs required)
via `ReciprocalRecommender`.

### Why not MINER?

The originally-planned Phase-2 paper, **MINER** (ACL 2022 Findings), is a *news*
recommender — it matches a user to news *items* using poly-attention over their
click history on the MIND benchmark. That is a **user→item** model; a dating app
needs **user↔user** reciprocal matching, and MINER needs large click logs we
don't have. So we kept its useful *multi-interest* idea and dropped the rest.
Modern reciprocal recommenders (CRRS, KAERR, LFRR) all require like/pass logs and
are deferred to a future Phase 3.

### The two methods

* **`method="recon"`** — a content-based take on **RECON** (Pizzato et al.,
  RecSys 2010). A directional score `s(A→B) = Σ_a w_A[a]·cos(emb_A[a], emb_B[a])`
  weights each attribute by how much A emphasises it (a proxy for A's preferences
  in the absence of history). The two directions are combined with the
  **harmonic mean**, which collapses toward the *smaller* side and so penalises
  lopsided matches — the reciprocal property a plain average lacks.
* **`method="multi_interest"`** — keeps MINER's multi-interest DNA. Each user's
  term tokens are clustered into `n_interests` **interest facets**; the
  directional score measures how well the other person *covers each of A's
  facets*, blended with biography similarity, then combined per direction with
  the harmonic mean.

```python
from aphrodite import ReciprocalRecommender
from aphrodite.datasets import generate_dataset

profiles, gt = generate_dataset(n_users=90, seed=42, with_gender=True)
rec = ReciprocalRecommender(method="recon", apply_gender_filter=True).fit(profiles)
rec.recommend("u000", k=10)          # mutually-compatible, reciprocally-ranked
rec.score_pair("u000", "u012")       # symmetric pair score (harmonic mean)
```

### Reciprocity metric & comparison

`reciprocity_rate(recommendations, k)` reports the fraction of top-`k` matches
that are **mutual** (if B is offered to A, is A also offered to B?). On a 90-user
gender-aware dataset (`python -m examples.compare_algorithms`), accuracy is on par
while the reciprocal methods are noticeably more bilaterally consistent:

| algorithm | P@5 | F1@10 | MAP@10 | **Recip@10** |
|-----------|-----|-------|--------|--------------|
| `ppr` (Phase 1) | 0.902 | 0.718 | 1.000 | 0.661 |
| `recon` | 0.902 | 0.718 | 1.000 | **0.740** |
| `multi_interest` | 0.902 | 0.718 | 1.000 | **0.740** |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core (numpy, networkx)
pip install -e '.[dev]'     # + pytest
pip install -e '.[full]'    # + gensim, sentence-transformers, nltk (paper-faithful)
```

## Quick start

```python
from aphrodite import MatchmakingAlgorithm
from aphrodite.datasets import generate_dataset

profiles, ground_truth = generate_dataset(n_users=60, seed=42)

algo = MatchmakingAlgorithm(threshold=0.70, damping=0.85).fit(profiles)
print(algo.recommend("u000", k=10))   # top-10 matches for user u000
```

Using the paper-faithful backend:

```python
from aphrodite import MatchmakingAlgorithm
from aphrodite.embeddings import Word2VecBertBackend

algo = MatchmakingAlgorithm(backend=Word2VecBertBackend()).fit(profiles)
```

## Run the demo

**Offline (no downloads):**
```bash
python -m examples.run_matchmaking --n 90
```
```
Users: 90  |  graph edges: 297  |  tau=0.7  d=0.85

        @5       @10      @15      @20
P       0.982    0.973    0.879    0.662
R       0.351    0.695    0.942    0.945
F1      0.517    0.811    0.910    0.778
MAP     0.351    0.693    0.940    0.941
```

**Paper-faithful (Word2Vec 300d + BERT, requires `pip install -e '.[full]'`):**
```python
from aphrodite import MatchmakingAlgorithm
from aphrodite.datasets import generate_dataset
from aphrodite.embeddings import Word2VecBertBackend

profiles, gt = generate_dataset(n_users=90, seed=42)
be = Word2VecBertBackend()  # downloads models on first use (~1.7 GB + 80 MB)
algo = MatchmakingAlgorithm(backend=be, threshold=0.70, damping=0.85).fit(profiles)
```
```
Users: 90  |  graph edges: 590  |  tau=0.7  d=0.85

        @5       @10      @15      @20
F1      0.526    0.833    0.966    0.824
MAP     0.357    0.714    1.000    1.000
```

**Phase 2 reciprocal + head-to-head comparison:**
```bash
python -m examples.run_matchmaking --algorithm recon --with-gender --n 90
python -m examples.compare_algorithms --n 90          # ppr vs recon vs multi_interest
```

## Tests

```bash
python -m pytest
```

---

## Package layout

```
aphrodite/
  profiles.py        UserProfile model (Table 1 attributes)
  preprocessing.py   tokenize / lowercase / stopwords / stemming (§3.1.1)
  embeddings/        base interface + lightweight + word2vec_bert backends (§3.1.2)
  similarity.py      cosine, aggregate, pairwise matrix (§3.2.1, Eq. 3–4)
  graph.py           weighted graph construction (§3.2.2, Eq. 5)
  ppr.py             Personalized PageRank + ranking (§3.3, Eq. 6)
  matchmaker.py      Algorithm 1 orchestrator (Phase 1)
  reciprocal.py      reciprocal recommenders: recon + multi_interest (Phase 2)
  metrics.py         P/R/F1/MAP@k (§4.2, Eq. 7–12) + reciprocity_rate
  datasets.py        synthetic clustered dataset + ground truth
examples/            runnable demos (run_matchmaking, compare_algorithms)
tests/               offline, deterministic unit tests
data/                sample generated dataset
```

## Notes on faithfulness

* The dataset generator stands in for the paper's 150 GPT-generated profiles
  (not publicly released). Profiles are organised into interest clusters so the
  ground-truth match set for a user is its cluster co-members.
* The PPR update is implemented exactly as written in the paper (Eq. 6), with `d`
  as the weight on the teleport-to-target term. `damping` is configurable if you
  prefer the conventional PageRank convention.
* Phase 2's reciprocal recommenders are **content-only** adaptations: RECON
  normally derives preferences from interaction history, which we proxy with
  per-attribute emphasis since no like/pass logs exist yet. Deep/causal
  reciprocal recommenders (CRRS, KAERR, LFRR) and the paper's graph baselines
  (Node2Vec, SimRank, Louvain, Random Walk, Girvan–Newman) are deferred to a
  future Phase 3 once real interaction data is collected.
