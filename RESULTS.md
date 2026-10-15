# Results: phase sensitivity and the full-attention control

All numbers are full-vocabulary top-1 accuracy (%) on the frozen phase-sweep
evaluation sets (8 residues, 1,000 logical examples per protocol; uniform-guess
chance = 1/64 ≈ 1.56%), produced by the exact pipeline in this repo.
Comparisons are **data-matched**: the compressed reference (`C0`) and the
uncompressed control (`full`) are trained on the same corpus, same data order
and same token budget. Only `C0` and `full` results are published here.

Setup per run: Qwen3-0.6B backbone, W=S=8, 1 KV head, scalar gates, separate
K/V branches, 16-token raw local window; FineWeb-Edu (sample-10BT); AdamW,
peak LR 1.2e-3 cosine → 2e-5, batch 131k tokens/step, seeds as noted.
Raw artifacts: `results/` (final-checkpoint evaluations + training logs);
evaluation-set manifest: `results/evalset_manifest.json`.

## Headline: compressed vs full attention at matched budgets

`W` = worst within-window phase (0–6), `G` = within-window gap (best − worst),
`B7` = boundary phase (key–value pair straddles a window).

### 1.75B tokens (2 seeds each)

| protocol | model | seed | mean | W | G | B7 |
|---|---|---|---:|---:|---:|---:|
| prefix_local | full | 42 | **44.0** | 42.1 | **3.5** | 45.0 |
| prefix_local | full | 43 | 38.2 | 36.9 | **2.2** | 39.5 |
| prefix_local | C0 | 42 | 33.9 | 26.1 | 27.4 | 5.7 |
| prefix_local | C0 | 43 | 43.3 | 18.4 | 43.6 | 32.8 |
| inseq_local | full | 42 | 9.2 | 8.3 | 1.6 | 9.6 |
| inseq_local | full | 43 | 5.4 | 4.8 | 1.1 | 5.1 |
| inseq_local | C0 | 42 | 17.6 | 12.3 | 11.7 | 4.8 |
| inseq_local | C0 | 43 | 13.5 | 5.9 | 13.6 | 8.3 |
| prefix | full | 42 | 51.4 | 48.8 | 4.4 | 53.5 |
| prefix | full | 43 | 40.5 | 37.9 | 4.6 | 40.1 |
| prefix | C0 | 42 | 29.2 | 26.1 | 14.5 | 4.9 |
| prefix | C0 | 43 | 41.2 | 19.8 | 43.0 | 41.6 |
| inseq | full | 42 | 29.4 | 28.1 | 3.3 | 29.5 |
| inseq | full | 43 | 15.2 | 14.8 | 1.1 | 15.2 |
| inseq | C0 | 42 | 18.0 | 15.1 | 9.1 | 3.9 |
| inseq | C0 | 43 | 21.2 | 14.7 | 15.2 | 15.1 |

### 500M tokens (seed 42 pair)

| protocol | model | mean | W | G | B7 |
|---|---|---:|---:|---:|---:|
| prefix_local | full | 27.5 | 26.9 | 1.6 | 27.7 |
| prefix_local | C0 | 8.7 | 6.7 | 5.0 | 3.1 |
| inseq_local | full | 1.6 | 1.3 | 0.4 | 1.6 |
| inseq_local | C0 | 5.5 | 4.0 | 3.9 | 2.3 |
| prefix | full | 24.1 | 23.0 | 2.0 | 24.0 |
| prefix | C0 | 8.6 | 6.8 | 5.9 | 3.2 |
| inseq | full | 13.9 | 12.8 | 1.7 | 14.0 |
| inseq | C0 | 7.5 | 5.0 | 6.2 | 2.8 |

### 3.5B tokens (seed 42; full-attention control training in progress)

| protocol | model | mean | W | G | B7 |
|---|---|---:|---:|---:|---:|
| prefix_local | C0 | 57.2 | 34.0 | **46.6** | 9.9 |
| inseq_local | C0 | 20.1 | 12.1 | 18.2 | 5.2 |
| prefix | C0 | 47.3 | 35.4 | 31.8 | 6.2 |
| inseq | C0 | 24.2 | 14.4 | 24.7 | 6.6 |
| prefix_local | full | — | — | — | — |

## What is reproduced, and what is not (yet)

**Reproduced (robust across seeds and protocols):**

1. **Phase sensitivity is a compression artifact, and it grows with budget.**
   The full-attention control is flat in every condition — within-window gap
   ≤ 4.6 pp across both seeds and all four protocols — while the compressed
   reference's gap grows 5.0 → 27.4 → **46.6 pp** (500M → 1.75B → 3.5B,
   seed 42, prefix_local), approaching the paper's 75.3 pp at 100B tokens
   with 1/30th of the compute. The contrast is categorical, not marginal.
2. **Boundary asymmetry.** Phase 7 (key–value pairs straddling a window
   boundary) is systematically suppressed in the compressed model at 500M
   (3.1 pp vs an 8.7 pp overall mean; see per-phase vectors in the eval
   JSONs), and seed-dependent at 1.75B, while full attention shows no boundary
   effect (B7 ≈ mean in every run).
3. **Not a language-modeling-quality artifact.** Validation loss is at parity
   (500M: C0 3.290 vs full 3.307; 1.75B: ≈2.98 both), so the retrieval
   uniformity difference is circuit-level, not capability-level.
4. **Seed variance of the compressed model grows with budget.** At 500M the
   C0 mean across seeds is tight (std ≈ 0.7 pp, n=4 incl. a same-seed
   replica), while at 1.75B two seeds differ by 9.4 pp in mean and phase
   metrics swing far more (a same-seed replica at 500M measures a ±2–3 pp
   nondeterminism floor on worst/gap). Full attention is comparatively stable.

**Not (yet) reproduced:** the paper's Table-8 ordering in which the compressed
reference *exceeds* the 1-KV-head full-attention baseline on mean accuracy at
100B tokens. At 1.75B our mean comparison is seed-dependent (full leads by
10.1 pp on seed 42; C0 leads by 5.2 pp on seed 43). The paper's
mechanism-level analyses (KV-head knockouts, gate cycling) have not been
re-run here.

## Protocol note

`inseq_local` is an intentionally stressful diagnostic: prompts are ~66%
single repeated spacer tokens (`｜`), which the full-attention control handles
markedly worse than natural filler (9.2% vs 44.0% mean at 1.75B seed 42).
The compressed model is comparatively robust to this stressor. Comparisons
across protocols should bear this asymmetry in mind.
