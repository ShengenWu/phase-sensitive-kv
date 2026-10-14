"""Generate paired phase-sweep evaluation sets, with strong local randomization.

Adapts the paper's App. C.2/C.3 protocol to a single stride S=8 (M=8 residues):

- 64 single-token animal->name pairs, target at body index 24, 8 in-body demos
- Prefix Padding: one filler sentence of exactly Lpre tokens before the body,
  one of exactly 48-Lpre after it; Lpre ~ X | X == r - c (mod 8) with
  X = M + U1 + U2 + U3 + F,  U, F ~ Unif{0..M-1}
  (residue-independent conditional moments; Lpre + Lpost fixed at 6M = 48)
- In-sequence Padding: two fixed 8-token guards plus 63 bounded gaps g_i in {0..4}
  of single-token U+FF5C spacers, 126 spacers in total, with the pre-target gaps
  summing to x = Lpre - 8 (latent draw from the same conditional law)
- prompts are assembled as exact token sequences; every prompt is verified to
  re-encode to the constructed ids (including with the gold token appended)

Output: /root/rescsa/data/evalset_{prefix,inseq}.npz (frozen; never regenerate with
different seeds) + evalset_manifest.json.
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np

DATA_ROOT = os.environ.get("RESCSA_DATA_DIR", "/root/rescsa/data")
TOK_DIR = f"{DATA_ROOT}/qwen3-base"
POOLS = f"{DATA_ROOT}/pools.json"
FILLER_GZ = f"{DATA_ROOT}/hqes/train.txt.gz"
OUT_DIR = DATA_ROOT

M = 8              # residues (equals the compression stride this round)
N_EXAMPLES = 1000  # logical examples per protocol
J_STAR = 24        # target body index
N_DEMOS = 8
GUARD = 8          # in-seq guard length (tokens)
GAP_MAX = 4
N_GAPS = 63
TOTAL_SPACERS = 126
SEED_SHUFFLE = 20261010   # filler corpus shuffle
SEED_GEN = 20261011       # mapping / demos / fillers / gaps

DESC_TEXT = "Remember each animal's assigned human name.\n\n"
HDR_TEXT = "Animal-name pairs:\n"
RECALL_TEXT = "Recall the requested animal names:\n"


def build_sentence_bins(tok, pool_ids):
    """Render sentences with a trailing blank line; bucket by exact token length."""
    bins: dict[int, list[list[int]]] = {}
    with gzip.open(FILLER_GZ, "rt") as f:
        sentences = [line.strip() for line in f if line.strip()]
    rng = np.random.default_rng(SEED_SHUFFLE)
    order = rng.permutation(len(sentences))
    kept = [sentences[i] + "\n\n" for i in order]
    for start in range(0, len(kept), 4096):
        encs = tok(kept[start : start + 4096], add_special_tokens=False)["input_ids"]
        for e in encs:
            if M <= len(e) <= 6 * M and not (set(e) & pool_ids):
                if len(bins.setdefault(len(e), [])) < 64:
                    bins[len(e)].append(e)
    missing = [L for L in range(M, 6 * M + 1) if not bins.get(L)]
    assert not missing, f"empty filler bins: {missing}"
    return bins


def sample_conditional_Lpre(rng, target_residue: int, c: int) -> int:
    """Lpre ~ law of X = M + U1 + U2 + U3 + F conditioned on X == r - c (mod M)."""
    while True:
        x = M + rng.integers(0, M) + rng.integers(0, M) + rng.integers(0, M) + rng.integers(0, M)
        if x % M == (target_residue - c) % M:
            return int(x)


def sample_gaps(rng, x: int) -> np.ndarray:
    """Uniform bounded compositions over the actual pre/post-target gaps."""

    def comp(n, total):
        if not 0 <= total <= n * GAP_MAX:
            raise ValueError("infeasible bounded composition")
        # Count completions exactly; feasible-range uniform draws bias the gaps.
        ways = [[0] * (total + 1) for _ in range(n + 1)]
        ways[0][0] = 1
        for k in range(1, n + 1):
            for s in range(total + 1):
                ways[k][s] = sum(ways[k - 1][s - v]
                                 for v in range(min(GAP_MAX, s) + 1))
        g = np.zeros(n, dtype=int)
        rem = total
        for i in range(n):
            count = ways[n - i][rem]
            # Rejection sampling supports Python-sized counts beyond uint64.
            nbytes = (count.bit_length() + 7) // 8
            while True:
                draw = int.from_bytes(rng.bytes(nbytes), "little")
                draw &= (1 << count.bit_length()) - 1
                if draw < count:
                    break
            for v in range(min(GAP_MAX, rem) + 1):
                weight = ways[n - i - 1][rem - v]
                if draw < weight:
                    g[i] = v
                    rem -= v
                    break
                draw -= weight
        assert rem == 0
        return g

    return np.concatenate([comp(J_STAR, x), comp(N_GAPS - J_STAR, TOTAL_SPACERS - x)])


LOCAL_SPACER_BUDGET = N_GAPS * (M - 1)
PROTOCOLS = ("prefix", "inseq", "prefix_local", "inseq_local")


def sample_local_gaps(ex, r, c):
    """All neighboring gaps are uniform mod S conditional on target phase.

    Only gap 0, far from the target, compensates the target residue. Every other
    gap is independently uniform 0..7. A remote trailing guard absorbs total
    length variation. With 24 pre-target gaps, the first two source-position
    moments are phase-independent in distribution.
    """
    rng = np.random.default_rng(SEED_GEN + 9_000_001 * ex + 103 * r)
    gaps = rng.integers(0, M, size=N_GAPS)
    gaps[0] = (r - c - GUARD - int(gaps[1:J_STAR].sum())) % M
    return gaps


def filler_tokens(length, bins, rng):
    """Compose natural sentences without introducing pool tokens."""
    out = []
    while length:
        chunk = min(6 * M, length)
        if 0 < length - chunk < M:
            chunk -= M - (length - chunk)
        if chunk < M:
            raise ValueError("filler cannot be composed from available sentence lengths")
        out.extend(bins[chunk][int(rng.integers(0, len(bins[chunk])))])
        length -= chunk
    return out


def phase_diagnostics(data):
    """Report local phase dependence rather than assuming random gaps suffice."""
    r = data["residues"]
    positions = data["key_positions"]
    report = {"source_position_by_phase": {}, "neighbor_phase_by_target_phase": {}}
    for phase in range(M):
        sel = r == phase
        source = data["source_positions"][sel]
        report["source_position_by_phase"][str(phase)] = {
            "mean": float(source.mean()), "variance": float(source.var())}
    for offset in (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16):
        rows = {}
        for phase in range(M):
            values = positions[r == phase, J_STAR + offset] % M
            probs = np.bincount(values, minlength=M) / len(values)
            rows[str(phase)] = {"probabilities": probs.tolist(),
                                "tv_to_uniform": float(np.abs(probs - 1/M).sum()/2)}
        report["neighbor_phase_by_target_phase"][str(offset)] = rows
    return report


def main():
    import argparse
    import hashlib
    from pathlib import Path
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / f"evalset_{p}.npz").exists() for p in PROTOCOLS):
        raise FileExistsError("use a fresh output directory; frozen evalsets are never overwritten")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    with open(POOLS) as f:
        pools = json.load(f)
    animals = [t for _, t in pools["animals"]]
    names = [t for _, t in pools["names"]]
    pool_ids = set(animals) | set(names)
    spacer = tok.encode("｜", add_special_tokens=False)
    nl = tok.encode("\n", add_special_tokens=False)
    assert len(spacer) == len(nl) == 1
    spacer, nl = spacer[0], nl[0]
    desc = tok(DESC_TEXT, add_special_tokens=False)["input_ids"]
    hdr = tok(HDR_TEXT, add_special_tokens=False)["input_ids"]
    recall = tok(RECALL_TEXT, add_special_tokens=False)["input_ids"]
    bins = build_sentence_bins(tok, pool_ids)
    c = len(desc) + len(hdr) + J_STAR * 3

    # Draw semantic tasks ONCE, shared by every padding protocol.
    examples = []
    for ex in range(N_EXAMPLES):
        rng = np.random.default_rng(SEED_GEN + 5_000_003 * ex)
        keys = [animals[i] for i in rng.permutation(64)]
        vals = [names[i] for i in rng.permutation(64)]
        demo_idx = np.sort(rng.choice([i for i in range(64) if i != J_STAR],
                                     size=N_DEMOS, replace=False))
        examples.append((keys, vals, demo_idx.tolist()))
    semantic_hash = hashlib.sha256(json.dumps(examples).encode()).hexdigest()
    results = {}
    for protocol in PROTOCOLS:
        ids_flat, offsets = [], [0]
        residues, golds, ex_ids, cand, source_positions = [], [], [], [], []
        all_keys, all_vals, all_demos, all_key_positions = [], [], [], []
        for ex, (keys, vals, demo_idx) in enumerate(examples):
            demos = [v for i in demo_idx for v in (keys[i], vals[i], nl)]
            for r in range(M):
                rng = np.random.default_rng(SEED_GEN + 1_000_003 * ex + 37 * r)
                key_positions = []
                if protocol.endswith("_local"):
                    gaps = sample_local_gaps(ex, r, c)
                    Lpre = GUARD + int(gaps[:J_STAR].sum())
                    if protocol == "prefix_local":
                        pre = filler_tokens(Lpre, bins, rng)
                        post = filler_tokens(2 * GUARD + LOCAL_SPACER_BUDGET - Lpre, bins, rng)
                    else:
                        pre = [spacer] * GUARD
                        post = [spacer] * (GUARD + LOCAL_SPACER_BUDGET - int(gaps.sum()))
                else:
                    Lpre = sample_conditional_Lpre(rng, r, c)
                    if protocol == "prefix":
                        pre = list(bins[Lpre][int(rng.integers(0, len(bins[Lpre])))])
                        post = list(bins[6*M - Lpre][int(rng.integers(0, len(bins[6*M - Lpre])))])
                        gaps = None
                    else:
                        gaps = sample_gaps(rng, Lpre - GUARD)
                        pre, post = [spacer] * GUARD, [spacer] * GUARD
                body = []
                for i in range(64):
                    key_positions.append(len(desc) + len(pre) + len(hdr) + len(body))
                    body.extend((keys[i], vals[i], nl))
                    if i < 63 and protocol in {"inseq", "inseq_local"}:
                        body.extend([spacer] * int(gaps[i]))
                ids = desc + list(pre) + hdr + body + list(post) + recall + demos + [keys[J_STAR]]
                source = key_positions[J_STAR]
                assert source % M == r
                assert ids[source:source+2] == [keys[J_STAR], vals[J_STAR]]
                for i, position in enumerate(key_positions):
                    assert ids[position:position+2] == [keys[i], vals[i]]
                ids_flat.extend(ids)
                offsets.append(len(ids_flat))
                residues.append(r); golds.append(vals[J_STAR]); ex_ids.append(ex)
                cand.append(vals); source_positions.append(source)
                all_keys.append(keys); all_vals.append(vals); all_demos.append(demo_idx)
                all_key_positions.append(key_positions)
        results[protocol] = dict(
            ids=np.array(ids_flat, dtype=np.uint32), offsets=np.array(offsets, dtype=np.int64),
            residues=np.array(residues, dtype=np.int8), golds=np.array(golds, dtype=np.uint32),
            ex_ids=np.array(ex_ids, dtype=np.int32), candidates=np.array(cand, dtype=np.uint32),
            source_positions=np.array(source_positions, dtype=np.int64), schema_version=np.array(3),
            record_keys=np.array(all_keys, dtype=np.uint32), record_values=np.array(all_vals, dtype=np.uint32),
            demo_indices=np.array(all_demos, dtype=np.int8), key_positions=np.array(all_key_positions),
            semantic_sha256=np.array(semantic_hash))
        print(f"{protocol}: {len(residues)} prompts, mean len {len(ids_flat)/len(residues):.1f}", flush=True)

    for protocol, data in results.items():
        for key in ("record_keys", "record_values", "demo_indices", "golds", "residues", "ex_ids"):
            assert np.array_equal(data[key], results["prefix"][key]), (protocol, key)
        # Verify real tokenizer round-trips, including appending the gold token.
        for start in range(0, len(data["residues"]), 1024):
            expected = [data["ids"][data["offsets"][i]:data["offsets"][i+1]].tolist()
                        for i in range(start, min(start+1024, len(data["residues"])))]
            for gold_suffix in (False, True):
                batch = [ids + ([int(data["golds"][start+j])] if gold_suffix else [])
                         for j, ids in enumerate(expected)]
                encoded = tok([tok.decode(ids) for ids in batch], add_special_tokens=False)["input_ids"]
                assert encoded == batch, f"{protocol} round-trip mismatch at batch {start}"
    a, b = results["prefix_local"], results["inseq_local"]
    assert np.array_equal(a["source_positions"], b["source_positions"])
    assert np.array_equal(np.diff(a["offsets"]), np.diff(b["offsets"]))
    diagnostics = {p: phase_diagnostics(data) for p, data in results.items()}
    if N_EXAMPLES >= 1000:
        for offset in (-1, 1):
            for row in diagnostics["inseq_local"]["neighbor_phase_by_target_phase"][str(offset)].values():
                assert row["tv_to_uniform"] <= 0.10, "local phase randomization failed acceptance"
    for protocol, data in results.items():
        np.savez_compressed(out_dir / f"evalset_{protocol}.npz", **data)
    manifest = {"schema_version": 3, "M": M, "n_examples": N_EXAMPLES,
                "j_star": J_STAR, "n_demos": N_DEMOS, "semantic_sha256": semantic_hash,
                "primary_protocols": ["prefix_local", "inseq_local"],
                "diagnostic_protocols": ["prefix", "inseq"],
                "local_spacer_budget": LOCAL_SPACER_BUDGET,
                "neighbor_gap_support": list(range(M)),
                "seeds": {"shuffle": SEED_SHUFFLE, "gen": SEED_GEN},
                "diagnostics": diagnostics}
    with open(out_dir / "evalset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("EVALSET_V3_FROZEN", flush=True)


if __name__ == "__main__":
    main()
