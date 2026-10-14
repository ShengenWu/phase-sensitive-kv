"""Exercise launcher success/failure paths with fake executables, never GPUs/network."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_default_entry_uses_frozen_plan_instead_of_legacy_matrix(self):
        script = Path(__file__).resolve().parents[2] / "regen.sh"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "code").mkdir()
            (root / "bin").mkdir()
            python = root / "bin" / "python3"
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" > "$CAPTURE"\n')
            python.chmod(0o755)
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                       RESCSA_EXPERIMENT_ROOT=d, RESCSA_ALLOW_LEGACY_REGEN="0", CAPTURE=str(root / "args"))
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertIn("rescsa.train.science_runner --root", (root / "args").read_text())

    def test_completion_requires_every_child_to_succeed(self):
        script = Path(__file__).resolve().parents[2] / "regen.sh"
        for failure in ("", "download", "train", "eval"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                bin_dir = root / "bin"
                bin_dir.mkdir()
                commands = {
                    "curl": '''#!/bin/bash
[[ "${TEST_FAIL:-}" != download ]] || exit 9
while (($#)); do
 if [[ "$1" == --output ]]; then shift; printf '{}' > "$1"; fi
 shift
done
''',
                    "gunzip": "#!/bin/bash\nexit 0\n",
                    "nvidia-smi": "#!/bin/bash\nprintf '0\\n0\\n0\\n0\\n0\\n0\\n0\\n0\\n'\n",
                    "python3": '''#!/bin/bash
if [[ "$*" == *rescsa.eval.run_eval* && "${TEST_FAIL:-}" == eval ]]; then exit 7; fi
while (($#)); do
 if [[ "$1" == --out ]]; then shift; printf '{}' > "$1"; fi
 shift
done
exit 0
''',
                    "torchrun": '''#!/bin/bash
while (($#)); do
 case "$1" in
 --run_name) shift; run=$1;;
 --variant) shift; variant=$1;;
 esac
 shift
done
[[ "${TEST_FAIL:-}" != train || "$variant" != C2 ]] || exit 8
mkdir -p "$RESCSA_RUNS_DIR/$run"
touch "$RESCSA_RUNS_DIR/$run/ckpt_500M.pt"
''',
                }
                for name, code in commands.items():
                    p = bin_dir / name
                    p.write_text(code)
                    p.chmod(0o755)
                env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
                           TEST_FAIL=failure, RESCSA_ALLOW_LEGACY_REGEN="1", RESCSA_LOG_DIR=str(root / "logs"),
                           RESCSA_CODE_DIR=d, RESCSA_DATA_DIR=str(root / "data"),
                           RESCSA_RUNS_DIR=str(root / "runs"))
                result = subprocess.run(["bash", str(script)], env=env, capture_output=True, timeout=20)
                marker = root / "logs" / "REGEN_COMPLETE"
                if failure:
                    self.assertNotEqual(result.returncode, 0, result.stderr.decode())
                    self.assertFalse(marker.exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                    self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
