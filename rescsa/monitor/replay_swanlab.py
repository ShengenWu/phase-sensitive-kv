"""Replay/tail training log.jsonl files into SwanLab offline experiments.

One process per run (swanlab.init is once-per-process). Replays existing history,
then tails for new lines until the run's "done" event appears.

Run:  nohup python3 -m rescsa.monitor.replay_swanlab <run_dir> &
"""

from __future__ import annotations

import json
import os
import sys
import time

RUN_DIR = sys.argv[1]
RUN_NAME = os.path.basename(RUN_DIR.rstrip("/"))
LOG = f"{RUN_DIR}/log.jsonl"

import swanlab

swanlab.init(
    project="rescsa-regen",
    experiment_name=RUN_NAME,
    mode="offline",
    logdir="/root/rescsa/swanlab_logs",
    description="ResCSA round-1 paired runs (replayed from log.jsonl)",
)


def handle(line: str) -> bool:
    """Log one event; returns False when the run is done."""
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return True
    step = e.get("step", 0)
    if e.get("event") == "train":
        swanlab.log({
            "train/loss": e["loss"],
            "train/lr": e["lr"],
            "train/tokens_per_s": e.get("tokens_per_s", 0),
            "train/peak_gb": e.get("peak_gb", 0),
        }, step=step)
    elif e.get("event") == "val":
        swanlab.log({"val/loss": e["val_loss"]}, step=step)
    elif e.get("event") == "done":
        return False
    return True


# replay existing history, then tail (single file handle keeps its offset)
alive = True
f = None
while f is None:
    if os.path.exists(LOG):
        f = open(LOG)
    else:
        time.sleep(5)
for line in f:
    alive = handle(line)
while alive:
    line = f.readline()
    if not line:
        time.sleep(10)
        continue
    alive = handle(line)
f.close()

swanlab.finish()
print(f"replay done: {RUN_NAME}")
