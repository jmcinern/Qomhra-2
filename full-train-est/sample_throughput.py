#!/usr/bin/env python3
"""Agent5 — Sample a small, size-proportional text slice for throughput testing.

Produces a tiny token stream in the SAME format as tokenize_corpus.py's
train.bin (uint32 LE, <|endoftext|> separator after every doc) so it is a
drop-in input for the training-throughput harness (agent4).

The ~TARGET_WORDS total is split across sources proportionally to each source's
exact total word count (from text_token_ratio_report.json), so the per-source
token mix matches the full corpus. Rows are accepted with a uniform probability
so the sample is spread across each file rather than taken from the top.

Output:
  out-dir/sample.bin        uint32 LE token ids, sep after every doc
  out-dir/sample_meta.json  word/token counts, per-source, sep id, dtype
"""
import os
import json
import time
import argparse

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = "/scratch/project_465002364/Denorm/train/data"
DTYPE = np.uint32
ROW_BATCH = 256

# exact total words per source (from text_token_ratio_report.json)
TOTAL_WORDS = {
    "conversations_ga": 91515834,
    "corpas_full_clean": 101764312,
    "finepdfs": 157390996,
    "hplt3_mono": 497309731,
    "nce": 34471908,
    "oireachtas_ga": 477517,
    "uni-archives": 7744811,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model", required=True, help="local tokenizer dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-words", type=int, default=100_000)
    ap.add_argument("--sep-id", type=int, default=151643)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    os.makedirs(args.out_dir, exist_ok=True)
    grand = sum(TOTAL_WORDS.values())
    t0 = time.time()

    per_source = {}
    total_words = 0
    total_tokens = 0
    total_docs = 0
    out_path = os.path.join(args.out_dir, "sample.bin")

    with open(out_path, "wb") as out:
        for source, sw in TOTAL_WORDS.items():
            target = max(1, round(args.target_words * sw / grand))
            path = os.path.join(args.data_dir, source + ".parquet")
            pf = pq.ParquetFile(path)
            s_words = s_tokens = s_docs = 0
            buf = []

            def flush():
                nonlocal s_tokens
                if buf:
                    enc = tok(buf, add_special_tokens=False)["input_ids"]
                    flat = []
                    for ids in enc:
                        flat.extend(ids)
                        flat.append(args.sep_id)
                    arr = np.asarray(flat, dtype=DTYPE)
                    arr.tofile(out)
                    s_tokens += int(arr.size)
                    buf.clear()

            # Deterministic stride spreads the draw across the whole file and
            # always accepts at least row 0. Oversample ~3x target words, then
            # the per-source word cap stops the loop.
            stride = max(1, round(sw / (3.0 * target)))
            done = False
            row_idx = -1
            for batch in pf.iter_batches(batch_size=ROW_BATCH, columns=["text"]):
                for t in batch.column("text").to_pylist():
                    row_idx += 1
                    if not t:
                        continue
                    if row_idx % stride != 0:
                        continue
                    w = len(t.split())
                    s_words += w
                    s_docs += 1
                    buf.append(t)
                    if len(buf) >= ROW_BATCH:
                        flush()
                    if s_words >= target:
                        done = True
                        break
                if done:
                    break
            flush()

            per_source[source] = {
                "target_words": target,
                "sample_words": s_words,
                "sample_tokens": s_tokens,
                "sample_docs": s_docs,
                "tokens_per_word": (s_tokens / s_words) if s_words else None,
            }
            total_words += s_words
            total_tokens += s_tokens
            total_docs += s_docs
            tpw = per_source[source]["tokens_per_word"]
            print(f"{source:22} words={s_words:>7,} tokens={s_tokens:>8,} "
                  f"docs={s_docs:>6,} tpw={tpw:.3f}" if tpw else
                  f"{source:22} words={s_words:>7,} (EMPTY)", flush=True)

    meta = {
        "purpose": "throughput-test sample, NOT for real training",
        "model": args.model,
        "dtype": np.dtype(DTYPE).name,
        "sep_id": args.sep_id,
        "sep_after_every_doc": True,
        "target_words": args.target_words,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "pooled_tokens_per_word": (total_tokens / total_words) if total_words else None,
        "sample_bin": out_path,
        "sample_bin_bytes": total_tokens * np.dtype(DTYPE).itemsize,
        "per_source": per_source,
        "seed": args.seed,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "sample_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nTOTAL words={total_words:,} tokens={total_tokens:,} "
          f"docs={total_docs:,}  tpw={meta['pooled_tokens_per_word']:.4f}")
    print(f"sample.bin: {out_path} ({meta['sample_bin_bytes']/1e6:.2f} MB)")
    print(f"elapsed: {meta['elapsed_sec']} s")


if __name__ == "__main__":
    main()
