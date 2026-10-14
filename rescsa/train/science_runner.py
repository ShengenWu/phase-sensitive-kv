"""Idempotent launch of a frozen screening plan on exactly two idle H100 GPUs.

Default invocation performs bounded availability checks and returns immediately
when busy. --execute is used only by the detached, lock-protected child runner.
No torch import or CUDA context is created while waiting for resources.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def code_digest(root):
    digest = hashlib.sha256()
    for path in sorted((root / "code" / "rescsa").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update((root / "code" / "science_plan_v3.json").read_bytes())
    return digest.hexdigest()


def evalset_hashes(root):
    names = [f"eval_v3/evalset_{p}.npz" for p in ("prefix", "inseq", "prefix_local", "inseq_local")]
    names.append("eval_v3/evalset_manifest.json")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def write_state(root, **values):
    state = {"updated_at": time.time(), **values}
    temp = root / f"status.json.{os.getpid()}.tmp"
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(root / "status.json")
    return state


def idle_gpus():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                "--format=csv,noheader"], check=True, capture_output=True, text=True)
    occupied = {line.strip() for line in processes.stdout.splitlines() if line.strip().startswith("GPU-")}
    free = []
    for row in csv.reader(result.stdout.splitlines()):
        index, uuid, memory, utilization = (v.strip() for v in row)
        if uuid not in occupied and int(memory) < 512 and int(utilization) <= 5:
            free.append(int(index))
    return free


def pair_stays_idle(pair, checks=3):
    for i in range(checks):
        if not set(pair).issubset(idle_gpus()):
            return False
        if i + 1 < checks:
            time.sleep(5)
    return True


def wait_pair_idle(pair, timeout=60):
    """Allow utilization counters/contexts to settle after our previous child exits."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pair_stays_idle(pair):
            return True
        time.sleep(5)
    return False


def run_command(root, command, log_name, env):
    with (root / "logs" / log_name).open("a") as log:
        subprocess.run(command, cwd=root / "code", env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def execute(root, pair, plan):
    lock = (root / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = os.getpid()
    finished = []
    try:
        if not pair_stays_idle(pair):
            write_state(root, state="waiting", reason="selected pair became occupied")
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, pair)),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
                   RESCSA_TB_ROOT=str(root / "tb_logs"))
        write_state(root, state="gpu_preflight", pid=pid, gpus=pair)
        run_command(root, [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc_per_node=2", "-m", "rescsa.tests.gpu_preflight"], "gpu_preflight.log", env)
        for i, settings in enumerate(plan["runs"]):
            required = ((len(plan["runs"])-i) * 75 + 80) * 1024**3
            if shutil.disk_usage(root).free < required:
                raise RuntimeError("insufficient free disk for remaining full-state checkpoints")
            if not wait_pair_idle(pair):
                raise RuntimeError("selected GPU pair was taken by another task between stages")
            name = settings["name"]
            write_state(root, state="training", pid=pid, gpus=pair, current_run=name, completed=finished)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                       "-m", "rescsa.train.train", "--run_name", name,
                       "--data_dir", "/root/rescsa/data", "--runs_dir", str(root / "runs")]
            args = {"total_tokens": plan["total_tokens_per_run"],
                    "schedule_tokens": plan["total_tokens_per_run"],
                    "seed": plan["seed"], "data_seed": plan["data_seed"],
                    "seq_len": plan["seq_len"], "microbatch": plan["microbatch"],
                    "accum": plan["accum"], "lr": plan["lr"], "min_lr": plan["min_lr"],
                    "warmup_frac": plan["warmup_frac"],
                    "ckpt_every_tokens": plan["checkpoint_interval_tokens"],
                    **{k: v for k, v in settings.items() if k != "name"}}
            for key, value in args.items():
                command.extend(("--" + key, str(value)))
            run_command(root, command, name + "_train.log", env)
            checkpoints = list((root / "runs" / name).glob("ckpt_*M.pt"))
            if not checkpoints:
                raise RuntimeError("training returned without checkpoints")
            final = max(checkpoints, key=lambda p: int(p.stem[5:-1]))
            if not wait_pair_idle(pair):
                raise RuntimeError("selected GPU pair was taken before evaluation")
            write_state(root, state="evaluating", pid=pid, gpus=pair, current_run=name, completed=finished)
            evaluation = final.with_name(final.stem + "_eval.json")
            run_command(root, [sys.executable, "-m", "rescsa.eval.run_eval", "--ckpt", str(final),
                               "--eval_dir", str(root / "eval_v3"), "--out", str(evaluation),
                               "--batch_size", "32"], name + "_eval.log",
                        dict(env, CUDA_VISIBLE_DEVICES=str(pair[0])))
            if not evaluation.is_file():
                raise RuntimeError("evaluation returned without its report")
            finished.append(name)
        write_state(root, state="comparing", pid=pid, gpus=pair, completed=finished)
        run_command(root, [sys.executable, "-m", "rescsa.eval.compare",
                           *[str(root / "runs" / run["name"]) for run in plan["runs"]]],
                    "comparison.log", dict(env, CUDA_VISIBLE_DEVICES=""))
        write_state(root, state="done", gpus=pair, completed=finished,
                    comparison=str(root / "logs" / "comparison.log"))
    except Exception as error:
        write_state(root, state="failed", pid=pid, gpus=pair, completed=finished, error=str(error))
        raise
    finally:
        lock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--gpus", default="")
    args = ap.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "code" / "science_plan_v3.json").read_text())
    ready = json.loads((root / "READY.json").read_text())
    if ready.get("code_sha256") != code_digest(root) or not ready.get("cpu_tests_passed"):
        raise RuntimeError("code changed after acceptance or readiness is incomplete")
    if ready.get("evalset_sha256") != evalset_hashes(root):
        raise RuntimeError("frozen evaluation datasets changed after acceptance")
    if args.execute:
        pair = [int(v) for v in args.gpus.split(",")]
        if len(pair) != 2 or len(set(pair)) != 2:
            ap.error("exactly two distinct GPUs are required")
        execute(root, pair, plan)
        return
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status_path = root / "status.json"
        if status_path.exists():
            state = json.loads(status_path.read_text())
            if state["state"] in {"done", "failed"}:
                print(json.dumps(state)); return
            if state["state"] not in {"waiting", "ready"}:
                pid = state.get("pid")
                if pid and Path(f"/proc/{pid}/cmdline").exists():
                    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                    if b"rescsa.train.science_runner" in cmdline and b"--execute" in cmdline:
                        print(json.dumps(state)); return
                raise RuntimeError("stale runner state; inspect before retrying, do not launch a duplicate")
        free = idle_gpus()
        if len(free) < 2:
            print(json.dumps(write_state(root, state="waiting", free_gpus=free))); return
        pair = free[:2]
        if not pair_stays_idle(pair):
            print(json.dumps(write_state(root, state="waiting", reason="GPU availability changed"))); return
        needed = (len(plan["runs"])*75 + 80) * 1024**3
        if shutil.disk_usage(root).free < needed:
            print(json.dumps(write_state(root, state="failed", error="insufficient disk before launch"))); return
        write_state(root, state="launching", gpus=pair)
        with (root / "logs" / "runner.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-B", "-m", "rescsa.train.science_runner",
                                        "--root", str(root), "--execute", "--gpus", ",".join(map(str, pair))],
                                       cwd=root / "code", stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, stdin=subprocess.DEVNULL)
        current = json.loads(status_path.read_text())
        if current["state"] == "launching":
            current["pid"] = process.pid
            write_state(root, **{k: v for k, v in current.items() if k != "updated_at"})
        print(json.dumps({"state": "launched", "pid": process.pid, "gpus": pair}))


if __name__ == "__main__":
    main()
