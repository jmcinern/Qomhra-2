#!/usr/bin/env python3
"""Agent2 — Tokenize the FULL text corpus and store token ids for training.

No sampling. Every document in every parquet is tokenized with the Qwen3-8B
tokenizer (add_special_tokens=False) and a document separator id is appended
after each doc so the stream can be packed into context windows downstream.

Output (nanoGPT/Megatron style flat token stream):
  out-dir/shards/<source>__rg<NN>.bin   uint32 little-endian, one per (file,row_group)
  out-dir/train.bin                      all shards concatenated (deterministic order)
  out-dir/meta.json                      dtype, eos/sep id, total tokens, per-source counts

Memory is bounded: each worker streams its row group in small row batches and
appends to its shard file incrementally (never holds a whole file in RAM).
"""
import os
import json
import glob
import time
import argparse
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = "/scratch/project_465002364/Denorm/train/data"
FILES = [
    "conversations_ga",
    "corpas_full_clean",
    "finepdfs",
    "hplt3_mono",
    "nce",
    "oireachtas_ga",
    "uni-archives",
]
DTYPE = np.uint32          # Qwen3 vocab (>151k) needs >16 bits
ROW_BATCH = 256            # rows per tokenizer call (bounds worker memory)

_tok = None
_sep = None
_shard_dir = None


def _init_worker(model_path, sep_id, shard_dir):
    global _tok, _sep, _shard_dir
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(model_path)
    _sep = sep_id
    _shard_dir = shard_dir


def _process_unit(unit):
    source, path, rg = unit
    shard = os.path.join(_shard_dir, f"{source}__rg{rg:02d}.bin")
    n_tokens = 0
    n_docs = 0
    pf = pq.ParquetFile(path)
    with open(shard, "wb") as out:
        for batch in pf.iter_batches(batch_size=ROW_BATCH, columns=["text"],
                                     row_groups=[rg]):
            texts = [t for t in batch.column("text").to_pylist() if t]
            if not texts:
                continue
            enc = _tok(texts, add_special_tokens=False)["input_ids"]
            flat = []
            for ids in enc:
                flat.extend(ids)
                flat.append(_sep)        # document separator
                n_docs += 1
            arr = np.asarray(flat, dtype=DTYPE)
            arr.tofile(out)
            n_tokens += arr.size
    return {"source": source, "rg": rg, "shard": shard,
            "n_tokens": n_tokens, "n_docs": n_docs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model", required=True, help="local tokenizer dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sep-id", type=int, default=151643,
                    help="document separator id (default <|endoftext|>)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    t0 = time.time()
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    units = []
    for f in FILES:
        path = os.path.join(args.data_dir, f + ".parquet")
        for rg in range(pq.ParquetFile(path).num_row_groups):
            units.append((f, path, rg))
    print(f"{len(units)} work units; workers={args.workers}; "
          f"sep_id={args.sep_id}; dtype={np.dtype(DTYPE).name}", flush=True)

    results = []
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(args.model, args.sep_id, shard_dir)) as pool:
        for r in pool.imap_unordered(_process_unit, units):
            results.append(r)
            print(f"  done {r['source']} rg{r['rg']:02d}: "
                  f"{r['n_tokens']:,} tok, {r['n_docs']:,} docs", flush=True)

    # Per-source totals + deterministic shard order.
    results.sort(key=lambda r: (r["source"], r["rg"]))
    per_source = {}
    total_tokens = 0
    total_docs = 0
    for r in results:
        s = per_source.setdefault(r["source"], {"n_tokens": 0, "n_docs": 0, "shards": []})
        s["n_tokens"] += r["n_tokens"]
        s["n_docs"] += r["n_docs"]
        s["shards"].append(os.path.basename(r["shard"]))
        total_tokens += r["n_tokens"]
        total_docs += r["n_docs"]

    # Concatenate shards -> train.bin (streamed, bounded memory).
    train_bin = os.path.join(args.out_dir, "train.bin")
    written = 0
    with open(train_bin, "wb") as out:
        for r in results:
            with open(r["shard"], "rb") as sh:
                while True:
                    buf = sh.read(1 << 24)  # 16 MiB chunks
                    if not buf:
                        break
                    out.write(buf)
                    written += len(buf)
    assert written == total_tokens * np.dtype(DTYPE).itemsize, \
        f"size mismatch: {written} vs {total_tokens * np.dtype(DTYPE).itemsize}"

    meta = {
        "model": args.model,
        "dtype": np.dtype(DTYPE).name,
        "sep_id": args.sep_id,
        "sep_after_every_doc": True,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "train_bin": train_bin,
        "train_bin_bytes": written,
        "per_source": per_source,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n=== Done ===")
    for s, d in sorted(per_source.items()):
        print(f"{s:22} {d['n_tokens']:>15,} tok  {d['n_docs']:>10,} docs")
    print(f"{'TOTAL':22} {total_tokens:>15,} tok  {total_docs:>10,} docs")
    print(f"train.bin: {train_bin}  ({written/1e9:.2f} GB, dtype={meta['dtype']})")
    print(f"elapsed: {meta['elapsed_sec']} s")


if __name__ == "__main__":
    main()
