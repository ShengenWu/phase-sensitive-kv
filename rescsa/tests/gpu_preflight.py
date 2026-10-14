"""Tiny two-GPU BF16/NCCL acceptance; run only on an idle allocated pair."""
import os
import torch
import torch.distributed as dist
from rescsa.tests.test_training_runtime import small_model
from rescsa.train.train import assert_synced


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        for mode, initialization, scope in (("mix", "copy", "kv"), ("add", "zero", "kv"),
                                             ("add", "copy", "kv"), ("add", "zero", "v")):
            model = small_model(c3_mode=mode, residual_init=initialization,
                                residual_scope=scope, residual_scale=0.05).cuda()
            ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            opt = torch.optim.AdamW(ddp.parameters(), lr=0.001)
            ids = torch.randint(0, 32, (1, 33), generator=torch.Generator().manual_seed(123+rank)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(input_ids=ids[:, :-1], labels=ids[:, 1:]).loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite BF16 loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("non-finite BF16 gradients")
            opt.step()
            assert_synced(model)
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                parallel = model(ids[:, :-1]).logits
                incremental = model.forward_incremental(ids[:, :-1])
            torch.testing.assert_close(parallel, incremental, atol=0.01, rtol=0.02)
            del ddp, model, opt, loss, ids, parallel, incremental
            torch.cuda.empty_cache()
        if rank == 0:
            print("GPU_BF16_NCCL_PREFLIGHT_PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
