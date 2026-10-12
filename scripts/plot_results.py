"""Result figures from the frozen phase-sweep evaluations.

Run after evaluation:  python scripts/plot_results.py --run_dir <runs>/learned_500M
Outputs figures/phase_accuracy.png (final checkpoint, both protocols)
and figures/budget_scaling.png (metrics vs training tokens).
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
CHANCE = 1 / 64


def load(run_dir):
    files = sorted(glob.glob(f"{run_dir}/ckpt_*M_eval.json"),
                   key=lambda f: int(re.search(r"ckpt_(\d+)M", f).group(1)))
    out = []
    for f in files:
        r = json.load(open(f))
        out.append({
            "tokens_M": int(re.search(r"ckpt_(\d+)M", f).group(1)),
            "variant": r.get("variant", "?"),
            "prefix": r["protocols"]["prefix"],
            "inseq": r["protocols"]["inseq"],
        })
    return out


def phase_accuracy(ckpts, suffix=""):
    last = ckpts[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key, title in zip(axes, ["prefix", "inseq"],
                              ["Prefix Padding", "In-sequence Padding"]):
        p = last[key]["full_vocab"]
        acc = [p["acc_by_phase"][str(i)] for i in range(8)]
        ax.axvspan(6.5, 7.5, color="#FFE0E0", zorder=0)
        ax.axhline(CHANCE, color="#999", lw=1, ls=":", label="uniform guess (1/64)")
        ax.plot(range(8), acc, "o-", color="#4C72B0", lw=2, ms=7, label="full-vocab top-1")
        cr = last[key]["candidate_restricted"]
        ax.plot(range(8), [cr["acc_by_phase"][str(i)] for i in range(8)],
                "s--", color="#DD8452", lw=1.5, ms=6, label="candidate-restricted")
        ax.set_xticks(range(8))
        ax.set_xlabel("source-key phase (position mod 8)")
        ax.set_title(f"{title} — {last['tokens_M']}M tokens")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.01, None)
    axes[0].set_ylabel("retrieval accuracy")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].text(7, axes[1].get_ylim()[1] * 0.92, "boundary\nphase", ha="center",
                 fontsize=8, color="#B71C1C")
    fig.suptitle(f"Phase sensitivity — variant '{last['variant']}' (reference kernel)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/phase_accuracy{suffix}.png", dpi=200, bbox_inches="tight")
    print(f"wrote figures/phase_accuracy{suffix}.png")


def budget_scaling(ckpts, suffix=""):
    toks = [c["tokens_M"] for c in ckpts]

    def series(key, field):
        return [c[key]["full_vocab"][field] for c in ckpts]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(toks, series("prefix", "mean_within"), "o-", label="mean (phases 0–6)", color="#4C72B0")
    ax.plot(toks, series("prefix", "worst_within"), "s-", label="worst phase (0–6)", color="#C44E52")
    ax.plot(toks, series("prefix", "boundary_phase7"), "^-", label="boundary phase 7", color="#8C8C8C")
    ax.axhline(CHANCE, color="#999", lw=1, ls=":")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("accuracy (Prefix Padding)")
    ax.set_title("Retrieval accuracy vs budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, color, label in [("prefix", "#4C72B0", "Prefix Padding"),
                              ("inseq", "#55A868", "In-sequence Padding")]:
        gaps, lo, hi = [], [], []
        for c in ckpts:
            b = c[key]["bootstrap"]["gap_within_ci95"]
            gaps.append(c[key]["full_vocab"]["gap_within"])
            lo.append(b[0])
            hi.append(b[1])
        gaps, lo, hi = np.array(gaps), np.array(lo), np.array(hi)
        ax.plot(toks, gaps, "o-", color=color, label=label)
        ax.fill_between(toks, lo, hi, color=color, alpha=0.18)
    ax.axhline(0, color="#999", lw=1, ls=":")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("within-window phase gap (pp)")
    ax.set_title("Phase gap vs budget (95% paired bootstrap CI)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/budget_scaling{suffix}.png", dpi=200)
    print(f"wrote figures/budget_scaling{suffix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="directory with ckpt_*M_eval.json")
    ap.add_argument("--suffix", default="", help="output filename suffix (e.g. _uniform)")
    args = ap.parse_args()
    ckpts = load(args.run_dir)
    print(f"{len(ckpts)} checkpoints found")
    phase_accuracy(ckpts, args.suffix)
    budget_scaling(ckpts, args.suffix)
