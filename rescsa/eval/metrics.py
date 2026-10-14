"""Phase metrics and cluster bootstrap retaining draw multiplicities."""
import numpy as np


def per_phase_stats(correct, residues):
    acc = {int(r): float(correct[residues == r].mean()) for r in np.unique(residues)}
    if set(acc) != set(range(8)):
        raise ValueError("evaluation must cover every phase 0..7")
    all_values = np.array([acc[r] for r in range(8)])
    within = all_values[:7]
    return {"acc_by_phase": acc, "mean": float(correct.mean()),
            "worst_all": float(all_values.min()), "best_all": float(all_values.max()),
            "gap_all": float(np.ptp(all_values)), "mean_within": float(within.mean()),
            "worst_within": float(within.min()), "best_within": float(within.max()),
            "gap_within": float(np.ptp(within)), "boundary_phase7": acc[7]}


def phase_matrix(correct, residues, ex_ids):
    exs, index = np.unique(ex_ids, return_inverse=True)
    matrix = np.zeros((len(exs), 8), dtype=float)
    count = np.zeros_like(matrix, dtype=int)
    np.add.at(matrix, (index, residues), correct)
    np.add.at(count, (index, residues), 1)
    if not np.all(count == 1):
        raise ValueError("each logical example must have exactly one row per phase")
    return exs, matrix


def metrics_of(accuracy):
    return {"mean": float(accuracy.mean()), "worst_all": float(accuracy.min()),
            "gap_all": float(np.ptp(accuracy)), "worst_within": float(accuracy[:7].min()),
            "gap_within": float(np.ptp(accuracy[:7])), "boundary_phase7": float(accuracy[7])}


def bootstrap_matrix(matrix, rng, n_boot=2000, baseline=None):
    if baseline is not None and baseline.shape != matrix.shape:
        raise ValueError("paired matrices have different shapes")
    draws = {k: [] for k in metrics_of(matrix.mean(axis=0))}
    for _ in range(n_boot):
        idx = rng.integers(0, len(matrix), size=len(matrix))
        values = metrics_of(matrix[idx].mean(axis=0))
        if baseline is not None:
            reference = metrics_of(baseline[idx].mean(axis=0))
            values = {k: values[k] - reference[k] for k in values}
        for k, value in values.items():
            draws[k].append(value)
    return {k + "_ci95": np.percentile(v, [2.5, 97.5]).tolist() for k, v in draws.items()}


def paired_bootstrap(correct, residues, ex_ids, rng, n_boot=2000):
    _, matrix = phase_matrix(correct, residues, ex_ids)
    return bootstrap_matrix(matrix, rng, n_boot)


def paired_difference(candidate, baseline, rng, n_boot=2000):
    """Match rows using (example, phase), then resample the same examples in both runs."""
    for key in ("dataset_sha256", "protocol", "schema_version"):
        if np.asarray(candidate[key]).item() != np.asarray(baseline[key]).item():
            raise ValueError(f"paired predictions mismatch: {key}")
    exs, matrix = phase_matrix(candidate["full_correct"], candidate["residues"], candidate["ex_ids"])
    base_exs, base_matrix = phase_matrix(baseline["full_correct"], baseline["residues"], baseline["ex_ids"])
    if not np.array_equal(exs, base_exs):
        raise ValueError("paired predictions have different logical examples")
    a, b = metrics_of(matrix.mean(axis=0)), metrics_of(base_matrix.mean(axis=0))
    return {"delta": {k: a[k] - b[k] for k in a},
            "bootstrap": bootstrap_matrix(matrix, rng, n_boot, base_matrix)}
