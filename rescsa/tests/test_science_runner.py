"""Resource gating and idempotency checks; never launch real processes or GPUs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from rescsa.train import science_runner as runner


class RunnerTests(unittest.TestCase):
    def ready_fixture(self, root):
        (root / "code" / "rescsa").mkdir(parents=True)
        (root / "code" / "science_plan_v3.json").write_text(json.dumps({"runs": []}))
        (root / "eval_v3").mkdir()
        for protocol in ("prefix", "inseq", "prefix_local", "inseq_local"):
            (root / "eval_v3" / f"evalset_{protocol}.npz").write_bytes(b"fixture")
        (root / "eval_v3" / "evalset_manifest.json").write_text("{}")
        (root / "READY.json").write_text(json.dumps({"cpu_tests_passed": True,
                                                     "code_sha256": runner.code_digest(root),
                                                     "evalset_sha256": runner.evalset_hashes(root)}))

    def test_memory_headroom_is_not_idle_when_compute_process_exists(self):
        def query(command, **kwargs):
            from types import SimpleNamespace
            if "--query-compute-apps=gpu_uuid" in command:
                return SimpleNamespace(stdout="GPU-busy\n")
            return SimpleNamespace(stdout="0, GPU-busy, 100, 0\n1, GPU-free, 0, 0\n2, GPU-hot, 0, 80\n")
        with patch.object(runner.subprocess, "run", side_effect=query):
            self.assertEqual(runner.idle_gpus(), [1])

    def test_busy_node_never_spawns_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.ready_fixture(root)
            with patch.object(sys, "argv", ["runner", "--root", d]), \
                 patch.object(runner, "idle_gpus", return_value=[0]), \
                 patch.object(runner.subprocess, "Popen") as spawn, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                spawn.assert_not_called()
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "waiting")

    def test_done_plan_does_not_launch_again(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.ready_fixture(root)
            runner.write_state(root, state="done")
            with patch.object(sys, "argv", ["runner", "--root", d]), \
                 patch.object(runner.subprocess, "Popen") as spawn, \
                 patch.object(runner, "idle_gpus") as gpu_query, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                spawn.assert_not_called(); gpu_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
