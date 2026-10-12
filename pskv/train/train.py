"""Pretraining loop for the compressed-attention reference models
(torchrun DDP, bf16 autocast, fp32 master weights).

Run (2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 -m pskv.train.train \
    --variant learned --run_name learned_500M --total_tokens 500_000_000 \
    --microbatch 8 --accum 4 --lr 1.2e-3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from pskv.model import build_model
from pskv.paths import DATA
from pskv.train.bench_throughput import chunked_ce_loss

LOGIT_CHUNK = 4096
INIT_SEED = 42
DATA_SEED = 777


class PackedData:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.train = np.memmap(f"{DATA}/packed/train.bin", dtype=np.uint32, mode="r")
        self.val = np.memmap(f"{DATA}/packed/val.bin", dtype=np.uint32, mode="r")

    def rows(self, which="train"):
        arr = self.train if which == "train" else self.val
        return len(arr) // self.seq_len

    def get_batch(self, row_indices, pin_dev):
        arr = self.train
        buf = np.stack([arr[i * self.seq_len : (i + 1) * self.seq_len] for i in row_indices])
        buf2 = np.stack([arr[i * self.seq_len + 1 : (i + 1) * self.seq_len + 1] for i in row_indices])
        x = torch.from_numpy(buf.astype(np.int64))
        y = torch.from_numpy(buf2.astype(np.int64))
        return x.to(pin_dev, non_blocking=True), y.to(pin_dev, non_blocking=True)

    def val_batch(self, row0, n, pin_dev):
        arr = self.val
        x = torch.from_numpy(np.stack(
            [arr[(row0 + i) * self.seq_len : (row0 + i + 1) * self.seq_len] for i in range(n)]
        ).astype(np.int64))
        y = torch.from_numpy(np.stack(
            [arr[(row0 + i) * self.seq_len + 1 : (row0 + i + 1) * self.seq_len + 1] for i in range(n)]
        ).astype(np.int64))
        return x.to(pin_dev), y.to(pin_dev)


def make_param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "gate" in name or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def val_loss(model, data, dev, n_rows=256):
    losses, ntok = 0.0, 0
    for i in range(0, n_rows, 32):
        x, y = data.val_batch(i, min(32, n_rows - i), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            hidden = model.model(x).last_hidden_state
        h = hidden.reshape(-1, hidden.size(-1))
        t = y.reshape(-1)
        for j in range(0, t.numel(), LOGIT_CHUNK):
            logits = model.lm_head(h[j : j + LOGIT_CHUNK]).float()
            losses += torch.nn.functional.cross_entropy(
                logits, t[j : j + LOGIT_CHUNK], reduction="sum").item()
            ntok += min(LOGIT_CHUNK, t.numel() - j)
    return losses / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["learned", "uniform"])
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--total_tokens", type=int, default=500_000_000)
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--min_lr", type=float, default=2e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.02)
    ap.add_argument("--ckpt_every_tokens", type=int, default=50_000_000)
    ap.add_argument("--val_every_steps", type=int, default=250)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    ddp = int(os.environ.get("RANK", -1)) >= 0
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank, world = 0, 1
    dev = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0

    run_dir = os.path.join(os.environ.get("PSKV_RUNS_DIR", "/root/rescsa/runs"), args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = f"{run_dir}/log.jsonl"

    model = build_model(args.variant, seed=INIT_SEED).to(dev)
    opt = torch.optim.AdamW(make_param_groups(model), lr=args.lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], gradient_as_bucket_view=True)
    base = model.module if ddp else model

    tokens_per_step = world * args.microbatch * args.accum * args.seq_len
    total_steps = args.total_tokens // tokens_per_step
    warmup_steps = max(10, int(args.warmup_frac * total_steps))
    data = PackedData(args.seq_len)

    step, tokens_seen, epoch = 0, 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        base.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, tokens_seen, epoch = ck["step"], ck["tokens_seen"], ck["epoch"]
        if is_main:
            print(f"resumed from {args.resume} at step {step}", flush=True)

    def lr_at(s):
        if s < warmup_steps:
            return args.lr * (s + 1) / warmup_steps
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * p))

    def save(kind):
        if not is_main:
            return
        path = f"{run_dir}/{kind}.pt" if kind != "last" else f"{run_dir}/last.pt"
        torch.save({
            "model": base.state_dict(), "config": base.config, "step": step,
            "tokens_seen": tokens_seen, "epoch": epoch, "args": vars(args),
            **({"optimizer": opt.state_dict()} if kind == "last" else {}),
        }, path)
        print(f"[ckpt] {path} (step {step}, {tokens_seen/1e6:.0f}M tokens)", flush=True)

    rng_rows = np.random.default_rng(DATA_SEED + epoch)
    row_perm = rng_rows.permutation(data.rows())
    cursor = 0
    next_ckpt_tokens = (tokens_seen // args.ckpt_every_tokens + 1) * args.ckpt_every_tokens

    if is_main:
        print(json.dumps({"event": "start", "variant": args.variant, "world": world,
                          "tokens_per_step": tokens_per_step, "total_steps": total_steps,
                          "train_rows": data.rows()}), flush=True)

    t0 = time.time()
    model.train()
    while step < total_steps:
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.accum):
            if cursor + world * args.microbatch > len(row_perm):
                epoch += 1
                row_perm = np.random.default_rng(DATA_SEED + epoch).permutation(data.rows())
                cursor = 0
            rows = row_perm[cursor + rank * args.microbatch :
                            cursor + (rank + 1) * args.microbatch]
            cursor += world * args.microbatch
            x, y = data.get_batch(rows, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                hidden = base.model(x).last_hidden_state
                loss = chunked_ce_loss(base, hidden, y) / args.accum
            loss.backward()
            accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tokens_seen += tokens_per_step

        if is_main:
            if step % 25 == 0:
                dt = time.time() - t0
                tps = 25 * tokens_per_step / dt
                mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0
                print(json.dumps({"event": "train", "step": step, "loss": round(accum_loss, 4),
                                  "lr": round(lr, 6), "tokens_M": round(tokens_seen / 1e6, 1),
                                  "tokens_per_s": round(tps), "peak_gb": round(mem, 1)}), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"event": "train", "step": step, "loss": accum_loss,
                                        "lr": lr, "tokens_seen": tokens_seen,
                                        "tokens_per_s": tps, "peak_gb": mem}) + "\n")
                t0 = time.time()
            if step % args.val_every_steps == 0:
                model.eval()
                vl = val_loss(base, data, dev)
                model.train()
                print(json.dumps({"event": "val", "step": step, "val_loss": round(vl, 4)}), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"event": "val", "step": step, "val_loss": vl}) + "\n")
        if tokens_seen >= next_ckpt_tokens or step == total_steps:
            save(f"ckpt_{tokens_seen // 1_000_000}M")
            next_ckpt_tokens += args.ckpt_every_tokens
    save("last")
    if is_main:
        print(json.dumps({"event": "done", "step": step, "tokens_seen": tokens_seen}), flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
