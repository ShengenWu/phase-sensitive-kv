"""Tokenize FineWeb-Edu sample-10BT and pack into fixed-length rows for pretraining.

- documents joined with EOS, packed into consecutive 2048-token blocks (standard packing)
- validation split: first VAL_TOKENS tokens from file 013 (fixed for all runs)
- output: train.bin / val.bin (uint32 streams) + manifest.json
- deterministic given the input files; the same files feed every variant (paired data order)

Run:  python -m rescsa.data.prepare_data --data_dir /root/rescsa/data/fineweb_10bt \
        --out_dir /root/rescsa/data/packed --seq_len 2048
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer

DATA_ROOT = os.environ.get("RESCSA_DATA_DIR", "/root/rescsa/data")
TOKENIZER_PATH = f"{DATA_ROOT}/qwen3-base/tokenizer.json"
EOS_ID = 151643  # <|endoftext|>
VAL_TOKENS = 12_000_000
BATCH_DOCS = 512


def iter_texts(data_dir: str, files: list[str]):
    for fname in files:
        path = os.path.join(data_dir, fname)
        table = pq.read_table(path, columns=["text"])
        for text in table.column("text").to_pylist():
            yield text
        print(f"  read {fname}", flush=True)


def encode_stream(texts, tok) -> list[int]:
    out = []
    buf = []
    for text in texts:
        buf.append(text if text else " ")
        if len(buf) >= BATCH_DOCS:
            for enc in tok.encode_batch(buf):
                out.extend(enc.ids)
                out.append(EOS_ID)
            buf = []
    if buf:
        for enc in tok.encode_batch(buf):
            out.extend(enc.ids)
            out.append(EOS_ID)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=f"{DATA_ROOT}/fineweb_10bt")
    ap.add_argument("--out_dir", default=f"{DATA_ROOT}/packed")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_train_tokens", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(TOKENIZER_PATH)

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".parquet"))
    val_file, train_files = files[-1], files[:-1]
    t0 = time.time()

    # ---- validation split (first VAL_TOKENS tokens of the smallest file)
    print(f"validation source: {val_file}", flush=True)
    val_ids = encode_stream(iter_texts(args.data_dir, [val_file]), tok)
    train_tail = val_ids[VAL_TOKENS:]  # remainder goes to train, order preserved
    val_ids = val_ids[:VAL_TOKENS]
    print(f"val tokens: {len(val_ids):,}; train tail from val file: {len(train_tail):,}", flush=True)

    import numpy as np

    with open(os.path.join(args.out_dir, "val.bin"), "wb") as f:
        f.write(np.asarray(val_ids, dtype=np.uint32).tobytes())

    # ---- train split (streamed in chunks to bound memory)
    print("streaming train files", flush=True)
    n_written = 0
    with open(os.path.join(args.out_dir, "train.bin"), "wb") as f:
        buf_ids: list[int] = []

        def flush():
            nonlocal n_written, buf_ids
            if buf_ids:
                arr = np.asarray(buf_ids, dtype=np.uint32)
                f.write(arr.tobytes())
                n_written += len(buf_ids)
                buf_ids = []

        if train_tail:
            buf_ids = list(train_tail)
            flush()
        for fname in train_files:
            table = pq.read_table(os.path.join(args.data_dir, fname), columns=["text"])
            texts = table.column("text").to_pylist()
            for i in range(0, len(texts), BATCH_DOCS):
                for enc in tok.encode_batch(texts[i : i + BATCH_DOCS]):
                    buf_ids.extend(enc.ids)
                    buf_ids.append(EOS_ID)
                if len(buf_ids) > 50_000_000:
                    flush()
                    if args.max_train_tokens and n_written >= args.max_train_tokens:
                        break
            print(f"  {fname}: cumulative {n_written:,} tokens ({(time.time()-t0)/60:.0f} min)", flush=True)
            if args.max_train_tokens and n_written >= args.max_train_tokens:
                break
        flush()

    manifest = {
        "seq_len": args.seq_len,
        "val_tokens": len(val_ids),
        "train_tokens": n_written,
        "eos_id": EOS_ID,
        "files": files,
        "val_file": val_file,
        "train_rows": n_written // args.seq_len,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"DATA_PREP_DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
