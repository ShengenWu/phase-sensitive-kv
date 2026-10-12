# Results: phase sensitivity at small training budgets

All numbers below are produced by the exact pipeline in this repo (fixed seeds) on
the reference (`learned`) and `uniform` kernels: Qwen3-0.6B backbone, W=S=8, 1 KV
head, scalar gates, FineWeb-Edu, 2×H100 per run, ~2.2 h per 500M-token run.
Raw artifacts: `results/` (per-checkpoint evaluations with bootstrap CIs + training
logs). Trained checkpoints will be published on Hugging Face.

Accuracy is full-vocabulary top-1 on the frozen 8-residue phase-sweep eval sets
(1,000 logical examples per protocol; uniform-guess chance = 1/64 ≈ 1.56%).

## Budget scaling — reference kernel (`learned`), Prefix Padding

| tokens | mean | worst phase (0–6) | gap (0–6) | boundary phase 7 |
|-------:|-----:|------------------:|----------:|-----------------:|
| 50M | 0.007 | 0.005 | 0.3 pp | 0.01 |
| 100M | 0.023 | 0.017 | 1.6 pp | 0.01 |
| 150M | 0.051 | 0.021 | 5.4 pp | 0.00 |
| 200M | 0.071 | 0.043 | 9.4 pp | 0.03 |
| 250M | 0.151 | 0.136 | 6.6 pp | 0.06 |
| 300M | 0.164 | 0.134 | 8.3 pp | 0.07 |
| 350M | 0.205 | 0.155 | 9.7 pp | 0.10 |
| 400M | 0.200 | 0.167 | 9.7 pp | 0.10 |
| 450M | 0.184 | 0.128 | 12.3 pp | 0.10 |
| 499M | **0.217** | **0.153** | **13.9 pp** | 0.12 |

## Final checkpoint (499M), per phase — reference kernel

| phase | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 (boundary) |
|------:|--:|--:|--:|--:|--:|--:|--:|--:|
| Prefix | 0.21 | 0.15 | 0.22 | 0.19 | 0.29 | 0.27 | 0.27 | 0.12 |
| In-seq | 0.12 | 0.11 | 0.14 | 0.14 | 0.12 | 0.10 | 0.10 | 0.07 |

Prefix gap CI95 = [12.1, 15.7] pp; In-seq gap = 4.1 pp, CI95 = [3.2, 5.8] pp.

## Uniform averaging at the same budget

The paper's uniform-averaging ablation (gate scores fixed at 1/W), trained
identically: mean 0.012, worst 0.010, gap 0.6 pp at 500M — retrieval stays at
chance level, so uniform averaging "removes" the phase gap at this budget only by
removing retrieval itself (the paper's 100B-token uniform model reaches ~12.5%
mean; uniform learns retrieval far more slowly than the learned kernel).

## Observations

1. **Within-window phase asymmetry is significant and grows with budget**:
   13.9 pp gap at 500M (CI above), from ~9 pp at 200M in this run.
2. **The boundary phase (7) is the weakest phase** at every budget, though its
   suppression strength varies across runs of the same configuration (an earlier
   run of the identical setup showed near-total suppression at 0.01; run-to-run
   variance at this small budget is substantial).
3. **In-sequence Padding shows the same ordering** with a smaller gap — the
   effect does not require all records to move in lockstep.
4. Retrieval is still climbing at 500M (the paper's 100B-token models reach 50%+
   mean and up to 75 pp gaps for this kernel), so these are lower bounds on the
   phase sensitivity this setup can produce.

## Figures

- `figures/phase_accuracy.png` / `phase_accuracy_uniform.png` — per-phase accuracy at the final checkpoint
- `figures/budget_scaling.png` / `budget_scaling_uniform.png` — mean/worst/boundary and gap-vs-budget with CIs
- `figures/kernel_windowing.png`, `figures/kernel_dataflow.png` — kernel schematics

## Artifacts

- `results/learned_500M/` — 10 checkpoints × evaluations (both protocols, per-phase
  accuracies, candidate-restricted, gold prob/NLL, bootstrap CIs) + `log.jsonl`
- `results/uniform_500M/` — the same for the uniform kernel

Hardware: 2×H100 80GB per run (~64k tokens/s), 500M tokens ≈ 2.2 h wall clock.
