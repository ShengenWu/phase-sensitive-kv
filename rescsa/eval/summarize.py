"""Compact summary table over a run directory's checkpoint eval JSONs."""
import glob
import json
import os
import re
import sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/rescsa/runs/C0_500M"
files = sorted(glob.glob(f"{run_dir}/ckpt_*M_eval.json"),
               key=lambda f: int(re.search(r"ckpt_(\d+)M", f).group(1)))
hdr = f"{'ckpt':>6s} {'proto':7s} " + " ".join(f"r{i:<4d}" for i in range(8)) + \
      f" {'mean':>6s} {'worst_all':>9s} {'gap_all':>8s} {'worst_in':>8s} {'gap_in':>7s} {'cand':>6s} {'goldp':>9s}"
print(hdr)
for f in files:
    m = re.search(r"ckpt_(\d+)M", f).group(1)
    r = json.load(open(f))
    if r.get('evaluation_schema_version') != 3:
        print('WARNING: legacy evaluation; CI invalid and training budget unverified')
    for proto in r["protocols"]:
        p = r["protocols"][proto]
        fv = p["full_vocab"]
        a = fv["acc_by_phase"]
        worst_all = min(a.values())
        gap_all = max(a.values()) - worst_all
        row = " ".join(f"{a[str(i)]:5.2f}" for i in range(8))
        print(f"{m:>5s}M {proto:7s} {row} {fv['mean']:6.3f} {worst_all:9.3f} {gap_all:8.3f} {fv['worst_within']:8.3f} "
              f"{fv['gap_within']:6.3f} {p['candidate_restricted']['mean']:6.3f} "
              f"{p['gold_prob_mean']:9.5f}")
