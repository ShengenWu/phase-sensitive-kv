"""Frozen-configuration phase evaluator.

Loads evalset_{prefix,inseq}.npz, scores a checkpoint, reports per-residue metrics:
full-vocabulary top-1 (primary), candidate-restricted top-1, gold probability, NLL;
mean / worst-phase / gap with phase 7 (cross-window boundary) reported separately;
paired bootstrap over logical examples for CI on worst-phase accuracy and gap.

Run:  python -m pskv.eval.run_eval --ckpt /root/rescsa/runs/C0/ckpt_200M.pt \
        [--out results.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pskv.model import ResCSAConfig, ResCSAForCausalLM
from pskv.paths import DATA, RUNS

DATA = DATA + ""
N_BOOT = 2000


@torch.no_grad()
def score_protocol(model, data, dev, batch_size=96):
    ids, offs = data["ids"], data["offsets"]
    n = len(data["residues"])
    order = np.argsort(np.diff(offs))[::-1]  # longest first for tight batching
    full_correct = np.zeros(n, dtype=bool)
    cand_correct = np.zeros(n, dtype=bool)
    gold_probs = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        idxs = order[start : start + batch_size]
        seqs = [ids[offs[i] : offs[i + 1]].tolist() for i in idxs]
        maxlen = max(len(s) for s in seqs)
        batch = torch.zeros(len(seqs), maxlen, dtype=torch.long, device=dev)
        for j, s in enumerate(seqs):
            batch[j, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
        last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            hidden = model.model(batch).last_hidden_state  # (B, L, d)
            h_last = hidden[torch.arange(len(seqs), device=dev), last]
            logits = model.lm_head(h_last).float()  # (B, V)
        golds = torch.tensor(data["golds"][idxs].astype(np.int64), device=dev)
        cands = torch.tensor(data["candidates"][idxs].astype(np.int64), device=dev)  # (B, 64)
        full_correct[idxs] = (logits.argmax(-1) == golds).cpu().numpy()
        cand_correct[idxs] = (cands[torch.arange(len(seqs), device=dev),
                               logits.gather(1, cands).argmax(-1)] == golds).cpu().numpy()
        logz = torch.logsumexp(logits, dim=-1)
        gold_lp = logits.gather(1, golds[:, None]).squeeze(1) - logz
        gold_probs[idxs] = gold_lp.exp().cpu().numpy().astype(np.float64)
    return full_correct, cand_correct, gold_probs


def per_phase_stats(correct, residues):
    acc = {int(r): float(correct[residues == r].mean()) for r in np.unique(residues)}
    within = [acc[r] for r in sorted(acc) if r < 7]  # phases 0-6 share a window
    stats = {
        "acc_by_phase": acc,
        "mean": float(correct.mean()),
        "mean_within": float(np.mean(within)),
        "worst_within": float(np.min(within)),
        "best_within": float(np.max(within)),
        "gap_within": float(np.max(within) - np.min(within)),
        "boundary_phase7": acc.get(7, None),
    }
    return stats


def paired_bootstrap(full_correct, residues, ex_ids, rng):
    """CI for worst-within accuracy and gap, resampling logical examples."""

    def stats_of(sel):
        acc = {}
        for r in range(7):
            m = sel & (residues == r)
            if m.sum() == 0:
                return None, None
            acc[r] = full_correct[m].mean()
        vals = list(acc.values())
        return min(vals), max(vals) - min(vals)

    exs = np.unique(ex_ids)
    member = {e: ex_ids == e for e in exs}
    worsts, gaps = [], []
    for _ in range(N_BOOT):
        sel = np.zeros_like(full_correct, dtype=bool)
        for e in rng.choice(exs, size=len(exs), replace=True):
            sel |= member[e]
        w, g = stats_of(sel)
        if w is not None:
            worsts.append(w)
            gaps.append(g)
    return {
        "worst_within_ci95": [float(np.percentile(worsts, 2.5)), float(np.percentile(worsts, 97.5))],
        "gap_within_ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--protocols", nargs="+", default=["prefix", "inseq"])
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config: ResCSAConfig = ck["config"]
    model = ResCSAForCausalLM(config).to(dev).eval()
    model.load_state_dict(ck["model"])
    tag = {"variant": config.variant, "ckpt": args.ckpt}

    report = {**tag, "protocols": {}}
    for protocol in args.protocols:
        data = np.load(f"{DATA}/evalset_{protocol}.npz")
        full_correct, cand_correct, gold_probs = score_protocol(model, data, dev)
        residues, ex_ids = data["residues"], data["ex_ids"]
        rng = np.random.default_rng(0)
        report["protocols"][protocol] = {
            "full_vocab": per_phase_stats(full_correct, residues),
            "candidate_restricted": per_phase_stats(cand_correct, residues),
            "gold_prob_mean": float(gold_probs.mean()),
            "gold_nll_mean": float(-np.log(np.clip(gold_probs, 1e-12, 1)).mean()),
            "bootstrap": paired_bootstrap(full_correct, residues, ex_ids, rng),
        }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
