"""Small CPU/Gloo acceptance tests for the production model training path.

Requires torch + transformers. These run only tiny unit-test models, never GPU jobs.
Run: python -m unittest rescsa.tests.test_training_runtime -v
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import random
import tempfile
import unittest
import sys
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


def small_model(**options):
    import torch
    from rescsa.model import ResCSAConfig, ResCSAForCausalLM
    torch.manual_seed(42)
    settings = dict(vocab_size=32, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=16, attention_dropout=0.0, tie_word_embeddings=True,
                       rope_parameters={"rope_theta": 1000000, "rope_type": "default"},
                       variant="C3", rho=0.1)
    settings.update(options)
    cfg = ResCSAConfig(**settings)
    model = ResCSAForCausalLM(cfg)
    model.sync_projections()
    return model


def ddp_worker(rank, init_file, options=None):
    import torch
    import torch.distributed as dist
    from rescsa.train.train import assert_synced
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    try:
        model = small_model(**(options or {}))
        reference = small_model(**(options or {}))
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        ids = torch.randint(0, 32, (4, 17), generator=torch.Generator().manual_seed(123))
        ref_opt = torch.optim.SGD(reference.parameters(), lr=0.05)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
        reference(input_ids=ids[:, :-1], labels=ids[:, 1:]).loss.backward()
        ref_opt.step()
        for microstep in range(2):
            row = rank + 2 * microstep
            sync = ddp.no_sync() if microstep == 0 else contextlib.nullcontext()
            with sync:
                loss = ddp(input_ids=ids[row:row+1, :-1], labels=ids[row:row+1, 1:]).loss / 2
                loss.backward()
        opt.step()
        assert_synced(model)
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(AVAILABLE, "requires torch and transformers")
class TrainingRuntimeTests(unittest.TestCase):
    def test_zero_additive_initialization_matches_c0(self):
        import torch
        baseline = small_model(variant="C0")
        ids = torch.arange(32).reshape(1, 32)
        for scope in ("kv", "v"):
            model = small_model(c3_mode="add", residual_init="zero", residual_scope=scope)
            torch.testing.assert_close(model(ids).logits, baseline(ids).logits, atol=0, rtol=0)
            torch.testing.assert_close(model(ids).logits, model.forward_incremental(ids), atol=2e-4, rtol=0)

    def test_additive_formula_and_v_only_preserve_key(self):
        import torch
        model = small_model(c3_mode="add", residual_scale=0.05, residual_scope="v")
        att = model.model.layers[0].self_attn
        h = torch.randn(1, 4, 32)
        k, v = torch.randn(1, 4, 16), torch.randn(1, 4, 16)
        new_k, new_v = att.apply_residual(k, v, h)
        torch.testing.assert_close(new_k, k, atol=0, rtol=0)
        torch.testing.assert_close(new_v, v + 0.05 * att.p_v(h), atol=0, rtol=0)
        self.assertFalse(att.p_k.weight.requires_grad)

    def test_ddp_additive_v_only_and_copy_initialization(self):
        import torch.multiprocessing as mp
        for settings in ({"c3_mode": "add", "residual_scope": "v", "residual_init": "zero"},
                         {"c3_mode": "add", "residual_init": "copy"}):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as d:
                mp.spawn(ddp_worker, args=(str(Path(d, "gloo")), settings), nprocs=2, join=True)

    def test_production_entry_checkpoint_resume_matches_full_run(self):
        import numpy as np
        import torch
        from rescsa.train import train as training
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            packed = root / "data" / "packed"
            packed.mkdir(parents=True)
            for name in ("train", "val"):
                (np.arange(512, dtype=np.uint32) % 32).tofile(packed / f"{name}.bin")
            common = ["train", "--variant", "C3", "--microbatch", "1", "--accum", "2",
                      "--seq_len", "16", "--schedule_tokens", "128",
                      "--ckpt_every_tokens", "32", "--val_every_steps", "2",
                      "--runs_dir", str(root / "runs"), "--data_dir", str(root / "data")]
            def run(name, budget, resume=None):
                argv = common + ["--run_name", name, "--total_tokens", str(budget)]
                if resume:
                    argv += ["--resume", str(resume)]
                with patch.object(sys, "argv", argv), \
                     patch.object(training, "build_model", side_effect=lambda *a, **kw: small_model()), \
                     patch.dict(__import__("os").environ, {"RESCSA_TB_ROOT": str(root / "tb")}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    training.main()
            run("full", 128)
            run("split", 64)
            last = root / "runs" / "split" / "last.pt"
            saved = torch.load(last, weights_only=False)
            self.assertEqual(saved["cursor"], 4)
            self.assertEqual(saved["tokens_seen"], 64)
            self.assertIn("optimizer", saved)
            run("split", 128, last)
            a = torch.load(root / "runs" / "full" / "last.pt", weights_only=False)
            b = torch.load(last, weights_only=False)
            self.assertEqual(a["cursor"], b["cursor"])
            self.assertEqual(a["step"], b["step"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)

    def test_ddp_accumulation_matches_single_process_global_batch(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(ddp_worker, args=(str(Path(d, "gloo")),), nprocs=2, join=True)

    def test_shifted_stream_drops_last_incomplete_target_row(self):
        import numpy as np
        from rescsa.train.train import PackedData
        with tempfile.TemporaryDirectory() as d:
            packed = Path(d, "packed")
            packed.mkdir()
            for name in ("train", "val"):
                np.arange(24, dtype=np.uint32).tofile(packed / f"{name}.bin")
            data = PackedData(8, d)
            self.assertEqual(data.rows(), 2)
            x, y = data.get_batch([1], "cpu")
            np.testing.assert_array_equal(y.numpy(), x.numpy()+1)

    def test_optimizer_and_rng_resume_matches_uninterrupted_update(self):
        import numpy as np
        import torch
        from rescsa.train.train import capture_rng, restore_rng
        def update(model, opt):
            # Consume all restored RNG streams, including data-selection randomness.
            random.random()
            np.random.random()
            ids = torch.randint(0, 32, (1, 17))
            opt.zero_grad(set_to_none=True)
            model(input_ids=ids[:, :-1], labels=ids[:, 1:]).loss.backward()
            opt.step()
        model = small_model()
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        update(model, opt)
        stream = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "rng": capture_rng()}, stream)
        update(model, opt)
        expected_rng = capture_rng()
        stream.seek(0)
        ck = torch.load(stream, weights_only=False)
        resumed = small_model()
        resumed.load_state_dict(ck["model"])
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.001)
        resumed_opt.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        update(resumed, resumed_opt)
        for p, q in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        torch.testing.assert_close(capture_rng()["torch"], expected_rng["torch"])


if __name__ == "__main__":
    unittest.main()
