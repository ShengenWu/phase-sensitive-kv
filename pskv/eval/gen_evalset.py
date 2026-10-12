"""Generate the FROZEN phase-sweep evaluation sets (two padding protocols).

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

import numpy as np
from transformers import AutoTokenizer
from pskv.paths import DATA

TOK_DIR = DATA + "/qwen3-base"
POOLS = DATA + "/pools.json"
FILLER_GZ = DATA + "/hqes/train.txt.gz"
OUT_DIR = DATA + ""

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
    """Bounded composition: 23 pre-target gaps summing to x and 40 post-target gaps
    summing to 126 - x, each in {0..GAP_MAX}; sequential feasible-range sampling."""

    def comp(n, total):
        g = np.zeros(n, dtype=int)
        rem = total
        for i in range(n):
            lo = max(0, rem - GAP_MAX * (n - i - 1))
            hi = min(GAP_MAX, rem)
            v = int(rng.integers(lo, hi + 1))
            g[i] = v
            rem -= v
        assert rem == 0
        return g

    return np.concatenate([comp(23, x), comp(40, TOTAL_SPACERS - x)])


def main():
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    pools = json.load(open(POOLS))
    animals = [t for _, t in pools["animals"]]
    names = [t for _, t in pools["names"]]
    pool_ids = set(animals) | set(names)

    spacer = tok.encode("｜", add_special_tokens=False)
    assert len(spacer) == 1
    spacer = spacer[0]
    nl = tok.encode("\n", add_special_tokens=False)
    assert len(nl) == 1
    nl = nl[0]

    desc = tok(DESC_TEXT, add_special_tokens=False)["input_ids"]
    hdr = tok(HDR_TEXT, add_special_tokens=False)["input_ids"]
    recall = tok(RECALL_TEXT, add_special_tokens=False)["input_ids"]

    bins = build_sentence_bins(tok, pool_ids)
    c = len(desc) + len(hdr) + J_STAR * 3
    print(f"desc={len(desc)} hdr={len(hdr)} pair=3 => c={c}; bins ready", flush=True)

    rng_master = np.random.default_rng(SEED_GEN)
    results = {}
    for protocol in ["prefix", "inseq"]:
        ids_flat: list[int] = []
        offsets = [0]
        residues, golds, ex_ids, cand = [], [], [], []
        for ex in range(N_EXAMPLES):
            keys = [animals[i] for i in rng_master.permutation(64)]
            vals = [names[j] for j in rng_master.permutation(64)]
            demo_idx = np.sort(rng_master.choice(
                [i for i in range(64) if i != J_STAR], size=N_DEMOS, replace=False))
            demos = []
            for i in demo_idx:
                demos += [keys[i], vals[i], nl]
            query = [keys[J_STAR]]

            for r in range(M):
                rng = np.random.default_rng(SEED_GEN + 1_000_003 * ex + 37 * r)
                Lpre = sample_conditional_Lpre(rng, r, c)
                assert M <= Lpre <= 6 * M
                if protocol == "prefix":
                    pre = bins[Lpre][rng.integers(0, len(bins[Lpre]))]
                    post = bins[6 * M - Lpre][rng.integers(0, len(bins[6 * M - Lpre]))]
                    body = []
                    for i in range(64):
                        body += [keys[i], vals[i], nl]
                    ids = desc + list(pre) + hdr + body + list(post) + recall + demos + query
                    expect = len(desc) + Lpre + len(hdr) + 192 + (6 * M - Lpre) \
                        + len(recall) + 3 * N_DEMOS + 1
                else:
                    x = Lpre - GUARD
                    gaps = sample_gaps(rng, x)
                    assert gaps[:23].sum() == x and gaps.sum() == TOTAL_SPACERS
                    body = []
                    for i in range(64):
                        body += [keys[i], vals[i], nl]
                        if i < 63:
                            body += [spacer] * int(gaps[i])
                    ids = (desc + [spacer] * GUARD + hdr + body + [spacer] * GUARD
                           + recall + demos + query)
                    expect = len(desc) + GUARD + len(hdr) + 192 + TOTAL_SPACERS + GUARD \
                        + len(recall) + 3 * N_DEMOS + 1
                assert len(ids) == expect, (protocol, ex, r, len(ids), expect)
                assert (c + Lpre) % M == r
                ids_flat += ids
                offsets.append(len(ids_flat))
                residues.append(r)
                golds.append(vals[J_STAR])
                ex_ids.append(ex)
                cand.append(list(vals))

        n = len(residues)
        print(f"{protocol}: {n} prompts, mean len {len(ids_flat)/n:.1f}", flush=True)
        results[protocol] = dict(
            ids=np.array(ids_flat, dtype=np.uint32),
            offsets=np.array(offsets, dtype=np.int64),
            residues=np.array(residues, dtype=np.int8),
            golds=np.array(golds, dtype=np.uint32),
            ex_ids=np.array(ex_ids, dtype=np.int32),
            candidates=np.array(cand, dtype=np.uint32),
        )

    # tokenization round-trip verification on ALL prompts (batched)
    for protocol, data in results.items():
        texts, texts_gold = [], []
        for idx in range(len(data["residues"])):
            ids = data["ids"][data["offsets"][idx] : data["offsets"][idx + 1]].tolist()
            gold = int(data["golds"][idx])
            texts.append(tok.decode(ids))
            texts_gold.append(tok.decode(ids + [gold]))
        for start in range(0, len(texts), 1024):
            re_ids = tok(texts[start : start + 1024], add_special_tokens=False)["input_ids"]
            for j, rei in enumerate(re_ids):
                idx = start + j
                ids = data["ids"][data["offsets"][idx] : data["offsets"][idx + 1]].tolist()
                assert rei == ids, f"{protocol} prompt {idx} round-trip failed"
            re_ids = tok(texts_gold[start : start + 1024], add_special_tokens=False)["input_ids"]
            for j, rei in enumerate(re_ids):
                idx = start + j
                ids = data["ids"][data["offsets"][idx] : data["offsets"][idx + 1]].tolist()
                gold = int(data["golds"][idx])
                assert rei == ids + [gold], f"{protocol} prompt {idx} gold append failed"
        print(f"{protocol}: full round-trip verification passed", flush=True)

    for protocol, data in results.items():
        np.savez_compressed(f"{OUT_DIR}/evalset_{protocol}.npz", **data)
    manifest = {
        "M": M, "n_examples": N_EXAMPLES, "j_star": J_STAR, "n_demos": N_DEMOS,
        "guard": GUARD, "gap_max": GAP_MAX, "total_spacers": TOTAL_SPACERS,
        "seeds": {"shuffle": SEED_SHUFFLE, "gen": SEED_GEN},
        "c": c, "desc_len": len(desc), "hdr_len": len(hdr),
        "prompt_counts": {p: int(len(d["residues"])) for p, d in results.items()},
    }
    with open(f"{OUT_DIR}/evalset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    print("EVALSET_FROZEN", flush=True)


if __name__ == "__main__":
    main()
