"""Replay/tail a run's log.jsonl into TensorBoard event files.

One process per run. Replays existing history, then tails until the "done" event.
Run:  nohup python3 -m rescsa.monitor.replay_tb <run_dir> &
"""

from __future__ import annotations

import json
import os
import sys
import time

from torch.utils.tensorboard import SummaryWriter

RUN_DIR = sys.argv[1]
RUN_NAME = os.path.basename(RUN_DIR.rstrip("/"))
LOG = f"{RUN_DIR}/log.jsonl"
TB_ROOT = os.environ.get("RESCSA_TB_ROOT", "/root/rescsa/tb_logs")

writer = SummaryWriter(log_dir=f"{TB_ROOT}/{RUN_NAME}")


def handle(line: str) -> bool:
    """Log one event; returns False when the run is done."""
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return True
    step = e.get("step", 0)
    if e.get("event") == "train":
        writer.add_scalar("train/loss", e["loss"], step)
        writer.add_scalar("train/lr", e["lr"], step)
        writer.add_scalar("train/tokens_per_s", e.get("tokens_per_s", 0), step)
        writer.add_scalar("train/peak_gb", e.get("peak_gb", 0), step)
    elif e.get("event") == "val":
        writer.add_scalar("val/loss", e["val_loss"], step)
    elif e.get("event") == "done":
        return False
    return True


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
writer.flush()
writer.close()
print(f"tb replay done: {RUN_NAME}")
