"""Compare audited evaluations using cluster-paired difference CIs."""
import argparse
import glob
import json
import re
from pathlib import Path
import numpy as np
from rescsa.eval.metrics import paired_difference


def final_eval(run_dir):
    files = sorted(glob.glob(f"{run_dir}/ckpt_*M_eval.json"),
                   key=lambda f: int(re.search(r"ckpt_(\d+)M", f).group(1)))
    if not files:
        raise ValueError(f"no checkpoint evaluations: {run_dir}")
    return Path(files[-1])


def load_report(run_dir):
    path = final_eval(run_dir)
    report = json.loads(path.read_text())
    if report.get("evaluation_schema_version") != 3:
        raise ValueError(f"legacy evaluation/invalid CI: {path}; re-evaluate")
    method = report["method"]
    if method["variant"] != report["variant"] or method["variant"] != report["training_args"].get("variant"):
        raise ValueError("variant metadata differs from training arguments")
    for key in ("rho", "c3_mode", "residual_scale", "residual_init", "residual_scope"):
        if method[key] != report["training_args"].get(key):
            raise ValueError(f"method metadata differs from training arguments: {key}")
    return path, report


def screening_passed(result, max_mean_drop=0.02, max_boundary_drop=0.02):
    """Within-window improvement is primary; boundary has a non-inferiority guard."""
    d, ci = result["delta"], result["bootstrap"]
    return (d["mean"] >= -max_mean_drop
            and ci["worst_within_ci95"][0] > 0
            and ci["gap_within_ci95"][1] < 0
            and ci["boundary_phase7_ci95"][0] >= -max_boundary_drop)


def load_predictions(path, report, protocol):
    return np.load(path.parent / report["protocols"][protocol]["predictions_file"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="first directory must be the C0 baseline")
    ap.add_argument("--max_mean_drop", type=float, default=0.02)
    ap.add_argument("--max_boundary_drop", type=float, default=0.02)
    args = ap.parse_args()
    reports = [load_report(run) for run in args.runs]
    base_path, base = reports[0]
    if base["variant"] != "C0":
        ap.error("first run must be C0")
    print("Primary: within-window worst/gap, mean guard, boundary non-inferiority guard.")
    primary_protocols = {"prefix_local", "inseq_local"}
    for path, report in reports:
        if not primary_protocols.issubset(report["protocols"]):
            raise ValueError("screening requires paired prefix_local and inseq_local protocols")
        outcomes = []
        print("method=" + json.dumps(report["method"], sort_keys=True))
        if report.get("legacy_training_budget_unverified"):
            print(f"WARNING {path.parent.name}: legacy training token budget is unverified")
        for proto, p in report["protocols"].items():
            f = p["full_vocab"]
            print(f"{path.parent.name} {proto}: mean={f['mean']:.4f} "
                  f"worst_all={f['worst_all']:.4f} gap_all={f['gap_all']:.4f} "
                  f"worst_within={f['worst_within']:.4f} gap_within={f['gap_within']:.4f} "
                  f"boundary7={f['boundary_phase7']:.4f}")
            if path == base_path:
                continue
            if report.get("tokens_seen") != base.get("tokens_seen"):
                raise ValueError("comparison requires the same token budget")
            if report.get("training_schema_version") != base.get("training_schema_version"):
                raise ValueError("cannot mix legacy and audited training runs")
            for key in ("world_size", "data_identity"):
                if report.get(key) != base.get(key):
                    raise ValueError(f"paired run mismatch: {key}")
            for key in ("seed", "data_seed", "microbatch", "accum", "seq_len",
                        "lr", "min_lr", "warmup_frac", "schedule_tokens"):
                if report["training_args"].get(key) != base["training_args"].get(key):
                    raise ValueError(f"paired training configuration mismatch: {key}")
            with load_predictions(path, report, proto) as cand, load_predictions(base_path, base, proto) as ref:
                result = paired_difference(cand, ref, np.random.default_rng(0))
            print("  paired differences: " + json.dumps(result, sort_keys=True))
            if proto in primary_protocols:
                passed = (screening_passed(result, args.max_mean_drop, args.max_boundary_drop)
                          and not report.get("legacy_training_budget_unverified"))
                outcomes.append(passed)
                print(f"  screening criteria {proto}: {'PASS' if passed else 'FAIL'}")
            else:
                print(f"  {proto}: diagnostic only")
        if path != base_path:
            print(f"OVERALL {path.parent.name}: {'PASS' if len(outcomes) == 2 and all(outcomes) else 'FAIL'}")


if __name__ == "__main__":
    main()
