"""Stage-A throughput / memory benchmark for the ResCSA training step.

Single-GPU sweep over microbatch sizes, or a real 2-GPU DDP benchmark via torchrun.
Reports steady-state tokens/s and peak VRAM for the full 0.6B reference model.

Run:  CUDA_VISIBLE_DEVICES=6 python -m rescsa.train.bench_throughput
      CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 -m rescsa.train.bench_throughput
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from rescsa.model import build_model

LOGIT_CHUNK = 4096  # tokens per lm-head/CE recompute chunk (peak ~2.5 GB fp32 logits)


def run(local_rank: int, world_size: int, args):
    dev = torch.device("cuda", local_rank)
    torch.manual_seed(42)
    model = build_model(args.variant).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], gradient_as_bucket_view=True)

    L = args.seq_len
    g = torch.Generator(device="cpu").manual_seed(123 + local_rank)

    def batch(bs):
        ids = torch.randint(0, 151936, (bs, L + 1), generator=g).to(dev)
        return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()

    results = []
    for bs in args.microbatches:
        try:
            ids, tgt = batch(bs)
            torch.cuda.reset_peak_memory_stats(dev)
            for _ in range(3):  # warmup
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, labels=tgt).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            if world_size > 1:
                torch.distributed.barrier()
            t0 = time.time()
            for _ in range(args.steps):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, labels=tgt).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            elapsed = torch.tensor(dt, device=dev)
            if world_size > 1:
                torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
            dt = elapsed.item()
            tps = bs * L * args.steps / dt * world_size
            peak = torch.cuda.max_memory_allocated(dev) / 1e9
            results.append((bs, tps, peak, float(loss)))
            if local_rank == 0:
                print(f"[bench] microbatch={bs:3d} | {tps/1e3:8.1f}k tokens/s (world={world_size}) "
                      f"| peak {peak:5.1f} GB | loss {float(loss):.3f}", flush=True)
            del ids, tgt, loss
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            if world_size > 1:
                # A peer may already be in a collective; let torchrun terminate
                # the whole group instead of continuing with mismatched steps.
                raise
            if local_rank == 0:
                print(f"[bench] microbatch={bs:3d} | OOM", flush=True)
            torch.cuda.empty_cache()
            break
    if local_rank == 0 and results:
        best = max(results, key=lambda r: r[1])
        print(f"BENCH_RESULT best_microbatch={best[0]} tokens_per_s={best[1]:.0f} "
              f"peak_gb={best[2]:.1f}", flush=True)
    if world_size > 1:
        torch.distributed.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="C0")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--microbatches", type=int, nargs="+", default=[4, 8, 16, 24, 32])
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    if int(os.environ.get("LOCAL_RANK", -1)) >= 0 or int(os.environ.get("RANK", -1)) >= 0:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = torch.distributed.get_world_size()
    else:
        local_rank, world_size = 0, 1
    run(local_rank, world_size, args)


if __name__ == "__main__":
    main()
