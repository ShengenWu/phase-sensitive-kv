"""Frozen-configuration phase evaluator.

Loads evalset_{prefix,inseq}.npz, scores a checkpoint, reports per-residue metrics:
full-vocabulary top-1 (primary), candidate-restricted top-1, gold probability, NLL;
mean / worst-phase / gap with phase 7 (cross-window boundary) reported separately;
paired bootstrap over logical examples for CI on worst-phase accuracy and gap.

Run:  python -m rescsa.eval.run_eval --ckpt /root/rescsa/runs/C0/ckpt_200M.pt \
        [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from rescsa.model import ResCSAConfig, ResCSAForCausalLM
from rescsa.eval.metrics import per_phase_stats, paired_bootstrap

DATA = "/root/rescsa/data"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--protocols", nargs="+", default=["prefix", "inseq", "prefix_local", "inseq_local"])
    ap.add_argument("--eval_dir", default=DATA)
    ap.add_argument("--batch_size", type=int, default=96)
    args = ap.parse_args()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        if Path(args.out).exists():
            ap.error("evaluation output exists; use a fresh path to preserve historical results")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config: ResCSAConfig = ck["config"]
    model = ResCSAForCausalLM(config).to(dev).eval()
    model.load_state_dict(ck["model"])
    tag = {"variant": config.variant, "rho": config.rho, "ckpt": args.ckpt,
           "evaluation_schema_version": 3,
           "training_schema_version": ck.get("training_schema_version", 1),
           "tokens_seen": ck.get("tokens_seen"),
           "training_args": ck.get("args", {}), "data_identity": ck.get("data_identity"),
           "world_size": ck.get("world_size"),
           "legacy_training_budget_unverified": ck.get("training_schema_version") != 2}
    tag["method"] = {"variant": config.variant, "rho": config.rho,
                     "c3_mode": getattr(config, "c3_mode", "mix"),
                     "residual_scale": getattr(config, "residual_scale", 0.1),
                     "residual_init": getattr(config, "residual_init", "copy"),
                     "residual_scope": getattr(config, "residual_scope", "kv")}

    report = {**tag, "protocols": {}}
    for protocol in args.protocols:
        data_path = Path(args.eval_dir) / f"evalset_{protocol}.npz"
        data = np.load(data_path)
        if "schema_version" not in data or int(data["schema_version"]) != 3:
            raise ValueError("regenerate paired schema-v3 data in a fresh directory")
        if not np.array_equal(data["source_positions"] % 8, data["residues"]):
            raise ValueError("source positions do not match phase labels")
        source = data["offsets"][:-1] + data["source_positions"]
        last = data["offsets"][1:] - 1
        if (not np.array_equal(data["ids"][source], data["ids"][last])
                or not np.array_equal(data["ids"][source + 1], data["golds"])):
            raise ValueError("source positions do not point to queried key/value pairs")
        full_correct, cand_correct, gold_probs = score_protocol(model, data, dev, args.batch_size)
        residues, ex_ids = data["residues"], data["ex_ids"]
        rng = np.random.default_rng(0)
        report["protocols"][protocol] = {
            "full_vocab": per_phase_stats(full_correct, residues),
            "candidate_restricted": per_phase_stats(cand_correct, residues),
            "gold_prob_mean": float(gold_probs.mean()),
            "gold_nll_mean": float(-np.log(np.clip(gold_probs, 1e-12, 1)).mean()),
            "bootstrap": paired_bootstrap(full_correct, residues, ex_ids, rng),
        }
        if args.out:
            pred_path = Path(args.out).with_name(Path(args.out).stem + f"_{protocol}_predictions.npz")
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
            np.savez_compressed(pred_path, full_correct=full_correct,
                                candidate_correct=cand_correct, gold_probs=gold_probs,
                                residues=residues, ex_ids=ex_ids, schema_version=np.array(3),
                                dataset_sha256=np.array(digest), protocol=np.array(protocol))
            report["protocols"][protocol]["predictions_file"] = pred_path.name
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out + ".tmp", "w") as f:
            json.dump(report, f, indent=2)
        Path(args.out + ".tmp").replace(args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
