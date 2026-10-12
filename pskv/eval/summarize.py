"""Compact summary table over a run directory's checkpoint eval JSONs."""
import glob
import json
import os
import re
import sys
from pskv.paths import DATA, RUNS

run_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PSKV_RUNS_DIR", "/root/rescsa/runs") + "/learned_500M"
files = sorted(glob.glob(f"{run_dir}/ckpt_*M_eval.json"),
               key=lambda f: int(re.search(r"ckpt_(\d+)M", f).group(1)))
hdr = f"{'ckpt':>6s} {'proto':7s} " + " ".join(f"r{i:<4d}" for i in range(8)) + \
      f" {'mean':>6s} {'worst':>6s} {'gap':>6s} {'cand':>6s} {'goldp':>9s}"
print(hdr)
for f in files:
    m = re.search(r"ckpt_(\d+)M", f).group(1)
    r = json.load(open(f))
    for proto in ["prefix", "inseq"]:
        p = r["protocols"][proto]
        fv = p["full_vocab"]
        a = fv["acc_by_phase"]
        row = " ".join(f"{a[str(i)]:5.2f}" for i in range(8))
        print(f"{m:>5s}M {proto:7s} {row} {fv['mean']:6.3f} {fv['worst_within']:6.3f} "
              f"{fv['gap_within']:6.3f} {p['candidate_restricted']['mean']:6.3f} "
              f"{p['gold_prob_mean']:9.5f}")
