"""Pretraining loop for ResCSA variants (torchrun DDP, bf16 autocast, fp32 master).

Pairing guarantees across variants (same CLI apart from --variant):
- same init seed (42) => identical shared-parameter values; C3's P starts as a copy of C
- same data seed => identical batch sequence (row order depends only on epoch + seed)
- same token budget, LR schedule, clip; report wall time and peak memory separately

Run (2 GPUs):
  CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 -m rescsa.train.train \
    --variant C0 --run_name C0_200M --total_tokens 200_000_000 \
    --microbatch 8 --accum 4 --lr 1.2e-3
"""

from __future__ import annotations

import argparse
import json
import os
import time
import random
import hashlib
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist


from rescsa.model import build_model

DATA = "/root/rescsa/data"
LOGIT_CHUNK = 4096
INIT_SEED = 42
DATA_SEED = 777


class PackedData:
    def __init__(self, seq_len: int, data_dir=DATA):
        self.seq_len = seq_len
        self.train = np.memmap(f"{data_dir}/packed/train.bin", dtype=np.uint32, mode="r")
        self.val = np.memmap(f"{data_dir}/packed/val.bin", dtype=np.uint32, mode="r")

    def rows(self, which="train"):
        arr = self.train if which == "train" else self.val
        # Every input row needs one additional token for its shifted labels.
        return max(0, (len(arr) - 1) // self.seq_len)

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
    n_rows = min(n_rows, data.rows("val"))
    if n_rows == 0:
        raise ValueError("validation stream has no complete shifted row")
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


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def assert_synced(model):
    """Compare all parameter bytes across ranks after a real optimizer update."""
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, digest.hexdigest())
    if len(set(hashes)) != 1:
        raise RuntimeError(f"DDP parameters diverged across ranks: {hashes}")


def main():
    from torch.utils.tensorboard import SummaryWriter
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["C0", "C1", "C2", "C3"])
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--layout_kind", choices=["fixed", "full"], default="fixed")
    ap.add_argument("--c3_mode", choices=["mix", "add"], default="mix")
    ap.add_argument("--residual_scale", type=float, default=0.1)
    ap.add_argument("--residual_init", choices=["copy", "zero"], default="copy")
    ap.add_argument("--residual_scope", choices=["kv", "v"], default="kv")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--total_tokens", type=int, default=200_000_000)
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--min_lr", type=float, default=2e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.02)
    ap.add_argument("--ckpt_every_tokens", type=int, default=50_000_000)
    ap.add_argument("--val_every_steps", type=int, default=250)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=INIT_SEED)
    ap.add_argument("--data_seed", type=int, default=DATA_SEED)
    ap.add_argument("--data_dir", default=DATA)
    ap.add_argument("--runs_dir", default="/root/rescsa/runs")
    ap.add_argument("--schedule_tokens", type=int, default=None,
                    help="fixed LR horizon; must be unchanged on resume")
    ap.add_argument("--verify_sync_steps", type=int, default=1)
    args = ap.parse_args()
    if min(args.microbatch, args.accum, args.seq_len, args.total_tokens,
           args.ckpt_every_tokens, args.val_every_steps) <= 0:
        ap.error("batch, lengths, budgets and intervals must be positive")
    if args.schedule_tokens is None:
        args.schedule_tokens = args.total_tokens
    if args.schedule_tokens < args.total_tokens or not 0 <= args.warmup_frac <= 1:
        ap.error("schedule_tokens must cover total_tokens")
    if not 0 <= args.rho <= 1 or args.verify_sync_steps < 0:
        ap.error("rho must lie in [0,1]; verify_sync_steps must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)

    ddp = int(os.environ.get("RANK", -1)) >= 0
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, world = 0, 1
        local_rank = 0
    dev = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0

    run_dir = f"{args.runs_dir}/{args.run_name}"
    if not args.resume and os.path.isdir(run_dir) and os.listdir(run_dir):
        raise ValueError("run directory is not empty; use a new name or --resume")
    os.makedirs(run_dir, exist_ok=True)
    log_path = f"{run_dir}/log.jsonl"
    tb_writer = SummaryWriter(
        log_dir=f"{os.environ.get('RESCSA_TB_ROOT', '/root/rescsa/tb_logs')}/{args.run_name}") \
        if is_main else None

    model = build_model(args.variant, seed=args.seed, rho=args.rho,
                        c3_mode=args.c3_mode, residual_scale=args.residual_scale,
                        residual_init=args.residual_init, residual_scope=args.residual_scope,
                        layout_kind=args.layout_kind).to(dev)
    opt = torch.optim.AdamW(make_param_groups(model), lr=args.lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], gradient_as_bucket_view=True)
    base = model.module if ddp else model

    tokens_per_step = world * args.microbatch * args.accum * args.seq_len
    total_steps = args.total_tokens // tokens_per_step
    schedule_steps = args.schedule_tokens // tokens_per_step
    if total_steps < 1:
        raise ValueError("token budget is smaller than one optimizer step")
    warmup_steps = min(schedule_steps, max(1, int(args.warmup_frac * schedule_steps)))
    data = PackedData(args.seq_len, args.data_dir)
    if data.rows() < world * args.microbatch:
        raise ValueError("training stream is smaller than one distributed microbatch")
    data_identity = {"train_tokens": len(data.train), "val_tokens": len(data.val)}
    for which in ("train", "val"):
        st = os.stat(f"{args.data_dir}/packed/{which}.bin")
        data_identity[which + "_mtime_ns"] = st.st_mtime_ns
    manifest_path = f"{args.data_dir}/packed/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "rb") as f:
            data_identity["manifest_sha256"] = hashlib.sha256(f.read()).hexdigest()

    step, tokens_seen, epoch, cursor = 0, 0, 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck.get("training_schema_version") != 2:
            raise ValueError("legacy checkpoint lacks reliable DDP/resume state; start a new audited run")
        if ck["world_size"] != world or ck["data_identity"] != data_identity:
            raise ValueError("resume world size or packed data identity changed")
        versions = {"torch": str(torch.__version__),
                    "transformers": __import__("transformers").__version__}
        if ck["versions"] != versions:
            raise ValueError("resume software versions changed; exact continuation is not supported")
        for key in ("variant", "rho", "seed", "data_seed", "microbatch", "accum",
                    "seq_len", "lr", "min_lr", "warmup_frac", "schedule_tokens",
                    "c3_mode", "residual_scale", "residual_init", "residual_scope"):
            defaults = {"c3_mode": "mix", "residual_scale": 0.1,
                        "residual_init": "copy", "residual_scope": "kv"}
            if ck["args"].get(key, defaults.get(key)) != vars(args)[key]:
                raise ValueError(f"resume changes {key}; exact continuation requires the saved setting")
        base.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, tokens_seen, epoch = ck["step"], ck["tokens_seen"], ck["epoch"]
        cursor = ck["cursor"]
        if not 0 <= cursor <= data.rows() or tokens_seen != step * tokens_per_step:
            raise ValueError("checkpoint cursor or token accounting is inconsistent")
        if step > total_steps:
            raise ValueError("resume checkpoint is beyond requested training budget")
        for existing in Path(run_dir).glob("ckpt_*M.pt"):
            checkpoint_m = int(existing.stem.removeprefix("ckpt_").removesuffix("M"))
            if checkpoint_m > tokens_seen // 1_000_000:
                raise ValueError("run directory contains later checkpoints; resume into a fresh run name")
        restore_rng(ck["rng_by_rank"][rank])
        if is_main:
            print(f"resumed from {args.resume} at step {step}", flush=True)

    def lr_at(s):
        if s < warmup_steps:
            return args.lr * (s + 1) / warmup_steps
        p = min(1.0, (s - warmup_steps) / max(1, schedule_steps - warmup_steps))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * p))

    def save(kind):
        rng_states = [None] * world
        if ddp:
            dist.all_gather_object(rng_states, capture_rng())
        else:
            rng_states[0] = capture_rng()
        if not is_main:
            dist.barrier()
            return
        path = f"{run_dir}/{kind}.pt" if kind != "last" else f"{run_dir}/last.pt"
        torch.save({
            "training_schema_version": 2, "world_size": world,
            "cursor": cursor, "rng_by_rank": rng_states, "data_identity": data_identity,
            "model": base.state_dict(), "config": base.config, "step": step,
            "tokens_seen": tokens_seen, "epoch": epoch, "args": vars(args),
            "optimizer": opt.state_dict(),
            "versions": {"torch": str(torch.__version__),
                         "transformers": __import__("transformers").__version__},
        }, path + ".tmp")
        os.replace(path + ".tmp", path)
        if kind != "last":
            # Atomic latest pointer, without doubling checkpoint disk usage.
            latest_tmp = f"{run_dir}/last.pt.tmp"
            if os.path.exists(latest_tmp):
                os.unlink(latest_tmp)
            os.link(path, latest_tmp)
            os.replace(latest_tmp, f"{run_dir}/last.pt")
        print(f"[ckpt] {path} (step {step}, {tokens_seen/1e6:.0f}M tokens)", flush=True)
        if ddp:
            dist.barrier()

    rng_rows = np.random.default_rng(args.data_seed + epoch)
    row_perm = rng_rows.permutation(data.rows())
    next_ckpt_tokens = (tokens_seen // args.ckpt_every_tokens + 1) * args.ckpt_every_tokens

    if is_main:
        print(json.dumps({"event": "start", "variant": args.variant, "world": world,
                          "tokens_per_step": tokens_per_step, "total_steps": total_steps,
                          "training_schema_version": 2, "seed": args.seed,
                          "data_seed": args.data_seed, "schedule_tokens": args.schedule_tokens,
                          "effective_total_tokens": total_steps * tokens_per_step,
                          "train_rows": data.rows()}), flush=True)

    t0 = time.time()
    model.train()
    while step < total_steps:
        for layer in base.model.layers:
            layer.self_attn.record_branch_stats = is_main and (step + 1) % 250 == 0
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for microstep in range(args.accum):
            if cursor + world * args.microbatch > len(row_perm):
                epoch += 1
                row_perm = np.random.default_rng(args.data_seed + epoch).permutation(data.rows())
                cursor = 0
            rows = row_perm[cursor + rank * args.microbatch :
                            cursor + (rank + 1) * args.microbatch]
            cursor += world * args.microbatch
            x, y = data.get_batch(rows, dev)
            sync = model.no_sync() if ddp and microstep < args.accum - 1 else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    loss = model(input_ids=x, labels=y).loss / args.accum
                loss.backward()
            accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tokens_seen += tokens_per_step
        if ddp and step <= args.verify_sync_steps:
            assert_synced(base)

        loss_sum = torch.tensor(accum_loss, device=dev)
        if ddp:
            dist.all_reduce(loss_sum)
        accum_loss = loss_sum.item() / world

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
                    if step % 250 == 0 and args.variant == "C3":
                        f.write(json.dumps({"event": "branch_stats", "step": step,
                                            "layers": {str(i): layer.self_attn.branch_stats
                                                       for i, layer in enumerate(base.model.layers)}}) + "\n")
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", accum_loss, step)
                    tb_writer.add_scalar("train/lr", lr, step)
                    tb_writer.add_scalar("train/tokens_per_s", tps, step)
                    tb_writer.add_scalar("train/peak_gb", mem, step)
                t0 = time.time()
        if step % args.val_every_steps == 0:
            if ddp:
                dist.barrier()
            if is_main:
                model.eval()
                vl = val_loss(base, data, dev)
                model.train()
                print(json.dumps({"event": "val", "step": step, "val_loss": round(vl, 4)}), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"event": "val", "step": step, "val_loss": vl}) + "\n")
                if tb_writer is not None:
                    tb_writer.add_scalar("val/loss", vl, step)
            if ddp:
                dist.barrier()
        if tokens_seen >= next_ckpt_tokens or step == total_steps:
            save(f"ckpt_{tokens_seen // 1_000_000}M")
            next_ckpt_tokens += args.ckpt_every_tokens
    if is_main:
        print(json.dumps({"event": "done", "step": step, "tokens_seen": tokens_seen}), flush=True)
    if tb_writer is not None:
        tb_writer.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
