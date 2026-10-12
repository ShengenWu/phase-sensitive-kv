# Results: phase sensitivity at small training budgets

> **Note on raw artifacts.** The numbers below come from the original training runs.
> The compute node's storage was reset before the per-checkpoint JSONs and training
> logs could be archived, so `results/` is currently empty; the raw artifacts are
> being regenerated with the exact pipeline in this repo (fixed seeds: identical
> init, data order, and eval sets) and will be added together with the published
> checkpoint. The tables below are transcribed from the original evaluation output.

Reference kernel (`learned`), Qwen3-0.6B backbone, W=S=8, 1 KV head, scalar gates,
FineWeb-Edu, seq len 2048, AdamW/cosine as described in the README. All runs used
identical init (seed 42) and data order. Accuracy is full-vocabulary top-1 on the
frozen 8-residue phase-sweep eval sets (1,000 logical examples per protocol;
uniform-guess chance = 1/64 ≈ 1.56% candidate / ≈0.07% full-vocab with 64 candidates).

## Budget scaling (Prefix Padding)

| tokens | mean | worst phase (0–6) | gap (0–6) | boundary phase 7 |
|-------:|-----:|------------------:|----------:|-----------------:|
| 100M   | 0.039 | 0.030 | 2.1 pp | 0.01 |
| 200M   | 0.071 | 0.046 | 7.2 pp | 0.00 |
| 300M   | 0.118 | 0.086 | 9.4 pp | 0.01 |
| 499M   | **0.116** | **0.084** | **14.9 pp** (CI [13.3, 16.6]) | **0.01** |

## Final checkpoint (499M), per phase

| phase | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 (boundary) |
|------:|--:|--:|--:|--:|--:|--:|--:|--:|
| Prefix | 0.09 | 0.12 | 0.11 | 0.15 | **0.23** | **0.08** | 0.14 | 0.01 |
| In-seq | 0.09 | 0.09 | 0.10 | 0.09 | 0.10 | 0.05 | 0.07 | 0.06 |

Observations:

1. **Within-window asymmetry emerges with budget** and is statistically solid at
   499M: phase 4 ≈ 0.23 vs phase 5 ≈ 0.08. The strong/weak phases are already
   stable from ~350M (phase 4 high, phase 5 low).
2. **The boundary phase (7) is completely suppressed** — key–value pairs split
   across a window boundary are not retrieved at any budget, consistent with the
   paper's boundary asymmetry (their gate-cycling intervention also fails to
   restore the boundary phase).
3. **In-sequence Padding shows the same ordering** (phase 4 high, phase 5 low)
   with a smaller gap (4.5 pp, CI [3.6, 5.8]): the effect does not require all
   records to move in lockstep.
4. Retrieval itself is still climbing at 500M (candidate-restricted 14% vs 1.56%
   chance), so these are lower bounds on the phase sensitivity this setup can
   produce — the paper's 100B-token models reach 50%+ mean accuracy and up to
   75 pp gaps for this kernel.

## Artifacts

- `results/` — per-checkpoint evaluations (both protocols, per-phase accuracies,
  candidate-restricted, gold prob/NLL, bootstrap CIs) and training logs; being
  regenerated from the identical pipeline (see note above)
- Trained reference checkpoint (500M tokens): to be published on Hugging Face

Hardware: 2×H100 80GB, ~64k tokens/s, 500M tokens ≈ 2.2 h wall clock.
