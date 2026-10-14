"""CPU regressions for phase construction, bootstrap and report contracts.

Run: python -m unittest rescsa.tests.test_audit_regressions -v
Only numpy is required; no training or real evaluation corpus is used.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from rescsa.eval import gen_evalset as gen
from rescsa.eval.metrics import bootstrap_matrix, paired_difference, per_phase_stats
from rescsa.eval.compare import load_report, screening_passed


class AuditTests(unittest.TestCase):
    def test_screening_rewards_within_window_with_boundary_guard(self):
        result = {"delta": {"mean": 0.01}, "bootstrap": {
            "worst_within_ci95": [0.02, 0.05], "gap_within_ci95": [-0.08, -0.03],
            "boundary_phase7_ci95": [-0.01, 0.01],
            "worst_all_ci95": [0, 0], "gap_all_ci95": [0, 0]}}
        self.assertTrue(screening_passed(result))
        result["bootstrap"]["worst_within_ci95"] = [0, 0]
        result["bootstrap"]["boundary_phase7_ci95"] = [0.1, 0.2]
        self.assertFalse(screening_passed(result))
        result["bootstrap"]["worst_within_ci95"] = [0.02, 0.05]
        result["bootstrap"]["boundary_phase7_ci95"] = [-0.04, -0.025]
        self.assertFalse(screening_passed(result))

    def test_neighbor_gap_is_uniform_conditional_on_target_phase(self):
        for r in range(8):
            gaps = np.stack([gen.sample_local_gaps(ex, r, 84) for ex in range(1000)])
            np.testing.assert_array_equal((84 + gen.GUARD + gaps[:, :gen.J_STAR].sum(axis=1)) % 8,
                                          np.full(1000, r))
            for i in (gen.J_STAR - 1, gen.J_STAR):
                probs = np.bincount(gaps[:, i], minlength=8)/1000
                self.assertLess(np.abs(probs - 0.125).sum()/2, 0.1)
    def test_bootstrap_retains_repeated_draws(self):
        class FixedDraw:
            def integers(self, *args, **kwargs):
                return np.array([0, 0, 1])
        matrix = np.repeat(np.array([[0.], [1.], [1.]]), 8, axis=1)
        ci = bootstrap_matrix(matrix, FixedDraw(), n_boot=1)
        self.assertEqual(ci["mean_ci95"], [1/3, 1/3])

    def test_paired_identical_runs_have_zero_difference(self):
        p = {"schema_version": np.array(2), "protocol": np.array("prefix"),
             "dataset_sha256": np.array("same"), "residues": np.tile(np.arange(8), 5),
             "ex_ids": np.repeat(np.arange(5), 8),
             "full_correct": np.random.default_rng(2).integers(0, 2, 40)}
        q = {k: v.copy() for k, v in p.items()}
        order = np.arange(40)[::-1]
        for k in ("residues", "ex_ids", "full_correct"):
            q[k] = q[k][order]
        result = paired_difference(q, p, np.random.default_rng(0), n_boot=40)
        self.assertTrue(all(v == 0 for v in result["delta"].values()))
        self.assertTrue(all(v == [0, 0] for v in result["bootstrap"].values()))
        q["dataset_sha256"] = np.array("different")
        with self.assertRaises(ValueError):
            paired_difference(q, p, np.random.default_rng(0))

    def test_primary_metrics_include_boundary(self):
        r = np.tile(np.arange(8), 10)
        correct = (r != 7)
        result = per_phase_stats(correct, r)
        self.assertEqual(result["worst_all"], 0)
        self.assertEqual(result["gap_all"], 1)
        self.assertEqual(result["worst_within"], 1)
        self.assertEqual(result["gap_within"], 0)

    def test_legacy_reports_cannot_supply_significance(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "ckpt_499M_eval.json").write_text('{"variant":"C0"}')
            with self.assertRaises(ValueError):
                load_report(d)

    def test_pre_target_gap_count(self):
        for x in range(0, 37):
            gaps = gen.sample_gaps(np.random.default_rng(x), x)
            self.assertEqual(len(gaps), 63)
            self.assertEqual(gaps[:gen.J_STAR].sum(), x)
            self.assertEqual(gaps.sum(), 126)
            self.assertTrue(np.all((gaps >= 0) & (gaps <= 4)))

    def test_generated_prompts_have_actual_source_phase(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                if isinstance(text, list):
                    return {"input_ids": [self(t)["input_ids"] for t in text]}
                fixed = {gen.DESC_TEXT: [7, 8], gen.HDR_TEXT: [9], gen.RECALL_TEXT: [10]}
                return {"input_ids": fixed[text] if text in fixed else [int(t) for t in text.split()]}
            def encode(self, text, **kwargs):
                return [11 if text == "｜" else 12]
            def decode(self, ids):
                return " ".join(str(i) for i in ids)
        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda _: Tokenizer())
        with tempfile.TemporaryDirectory() as d:
            pools = Path(d, "pools.json")
            pools.write_text(json.dumps({"animals": [[str(i), i+100] for i in range(64)],
                                         "names": [[str(i), i+200] for i in range(64)]}))
            with patch.dict(sys.modules, {"transformers": fake}), \
                 patch.object(gen, "POOLS", str(pools)), patch.object(gen, "N_EXAMPLES", 4), \
                 patch.object(gen, "build_sentence_bins", return_value={i: [[42]*i] for i in range(8, 49)}), \
                 patch.object(sys, "argv", ["gen_evalset", "--out_dir", d]), \
                 contextlib.redirect_stdout(io.StringIO()):
                gen.main()
            for protocol in gen.PROTOCOLS:
                with np.load(Path(d, f"evalset_{protocol}.npz")) as p:
                    source = p["offsets"][:-1] + p["source_positions"]
                    last = p["offsets"][1:] - 1
                    np.testing.assert_array_equal(p["source_positions"] % 8, p["residues"])
                    np.testing.assert_array_equal(p["ids"][source], p["ids"][last])
                    np.testing.assert_array_equal(p["ids"][source+1], p["golds"])
            with np.load(Path(d, "evalset_prefix_local.npz")) as a, np.load(Path(d, "evalset_inseq_local.npz")) as b:
                for key in ("record_keys", "record_values", "demo_indices", "golds", "source_positions"):
                    np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_array_equal(np.diff(a["offsets"]), np.diff(b["offsets"]))


if __name__ == "__main__":
    unittest.main()
