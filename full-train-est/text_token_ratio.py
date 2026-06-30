#!/usr/bin/env python3
"""Agent2 — Text words/token ratio estimator.

Single streaming pass over every text parquet on LUMI that, in parallel across
CPU cores (one task per (file, row_group)):

  1. Counts the EXACT total number of words in the whole corpus (whitespace
     split), per source.
  2. Draws a representative subset by accepting each row with a uniform
     probability `p`. Because word-count is ~proportional to byte size, uniform
     row acceptance yields a sample whose per-source word share is proportional
     to that source's size -> size-proportional sampling, as required.
     Small sources (< SMALL_FILE_BYTES) are taken in full (cheap, gives a clean
     per-source ratio; negligible weight in the aggregate).
  3. Tokenizes the sampled text with the Qwen3-8B tokenizer
     (add_special_tokens=False) and counts tokens.

Headline estimate uses the per-source ratio against the EXACT total words:
    est_total_tokens = sum_i ( total_words_i * tokens_per_word_i )

Word definition: str.split() (whitespace). Token definition: Qwen3-8B subword
ids, no special tokens.
"""
import os
import json
import time
import argparse
from multiprocessing import Pool

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
MODEL = "Qwen/Qwen3-8B"
SMALL_FILE_BYTES = 5 * 1024 * 1024  # files below this are sampled in full
TOK_CHUNK = 1000  # texts per tokenizer call in a worker

_tok = None  # per-worker tokenizer
MODEL_PATH = MODEL  # set in main(); a LOCAL dir avoids transformers' network call


def _init_worker():
    # Loading from a local directory makes transformers treat it as local
    # (_is_local=True) and skip the model_info() network probe that breaks in
    # offline mode on compute nodes.
    global _tok
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)


def _process_unit(unit):
    """Process one (file, row_group). Returns per-source partial counts."""
    import random as _random
    source, path, rg, accept_p, seed = unit
    rng = _random.Random(seed)

    pf = pq.ParquetFile(path)
    table = pf.read_row_group(rg, columns=["text"])
    texts = table.column("text").to_pylist()

    total_words = 0
    total_chars = 0
    total_rows = 0
    sample_words = 0
    sample_tokens = 0
    sample_rows = 0
    sample_buf = []

    def _flush():
        nonlocal sample_tokens
        if sample_buf:
            enc = _tok(sample_buf, add_special_tokens=False)["input_ids"]
            sample_tokens += sum(len(x) for x in enc)
            sample_buf.clear()

    for t in texts:
        if not t:
            total_rows += 1
            continue
        w = len(t.split())
        total_words += w
        total_chars += len(t)
        total_rows += 1
        if accept_p >= 1.0 or rng.random() < accept_p:
            sample_words += w
            sample_rows += 1
            sample_buf.append(t)
            if len(sample_buf) >= TOK_CHUNK:
                _flush()
    _flush()

    return {
        "source": source,
        "total_words": total_words,
        "total_chars": total_chars,
        "total_rows": total_rows,
        "sample_words": sample_words,
        "sample_tokens": sample_tokens,
        "sample_rows": sample_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model", default=MODEL,
                    help="local tokenizer dir (preferred) or HF repo id")
    ap.add_argument("--budget", type=int, default=3_000_000,
                    help="approx. target total sampled words")
    ap.add_argument("--bytes-per-word", type=float, default=7.0,
                    help="rough UTF-8 bytes/word, only used to set sample size")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="text_token_ratio_report.json")
    args = ap.parse_args()

    global MODEL_PATH
    MODEL_PATH = args.model  # inherited by forked workers

    t0 = time.time()

    # File sizes -> uniform acceptance probability p.
    sizes = {}
    for f in FILES:
        path = os.path.join(args.data_dir, f + ".parquet")
        sizes[f] = os.path.getsize(path)
    total_bytes = sum(sizes.values())
    est_total_words = total_bytes / args.bytes_per_word
    p = min(1.0, args.budget / est_total_words)
    print(f"total_bytes={total_bytes:,}  est_total_words~{est_total_words:,.0f}  "
          f"uniform accept p={p:.5f}", flush=True)

    # Build work units: one per (file, row_group). Small files sampled in full.
    units = []
    for f in FILES:
        path = os.path.join(args.data_dir, f + ".parquet")
        accept_p = 1.0 if sizes[f] < SMALL_FILE_BYTES else p
        nrg = pq.ParquetFile(path).num_row_groups
        for rg in range(nrg):
            units.append((f, path, rg, accept_p, args.seed + rg * 100 + hash(f) % 97))
    print(f"{len(units)} work units across {len(FILES)} files; "
          f"workers={args.workers}", flush=True)

    # Run.
    per_source = {f: {"file_bytes": sizes[f], "total_words": 0, "total_chars": 0,
                      "total_rows": 0, "sample_words": 0, "sample_tokens": 0,
                      "sample_rows": 0} for f in FILES}
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(_process_unit, units):
            d = per_source[r["source"]]
            for k in ("total_words", "total_chars", "total_rows",
                      "sample_words", "sample_tokens", "sample_rows"):
                d[k] += r[k]

    # Derived metrics.
    grand_total_words = 0
    grand_sample_words = 0
    grand_sample_tokens = 0
    est_total_tokens = 0.0
    for f, d in per_source.items():
        sw, st = d["sample_words"], d["sample_tokens"]
        tpw = (st / sw) if sw else float("nan")
        wpt = (sw / st) if st else float("nan")
        d["tokens_per_word"] = tpw
        d["words_per_token"] = wpt
        d["est_source_tokens"] = d["total_words"] * tpw if sw else 0.0
        grand_total_words += d["total_words"]
        grand_sample_words += sw
        grand_sample_tokens += st
        est_total_tokens += d["est_source_tokens"]

    pooled_tpw = grand_sample_tokens / grand_sample_words if grand_sample_words else float("nan")

    report = {
        "model": args.model,
        "word_def": "whitespace split (str.split())",
        "token_def": "Qwen3-8B ids, add_special_tokens=False",
        "accept_p": p,
        "budget": args.budget,
        "seed": args.seed,
        "elapsed_sec": round(time.time() - t0, 1),
        "total_corpus_words": grand_total_words,
        "sample_words": grand_sample_words,
        "sample_tokens": grand_sample_tokens,
        "pooled_tokens_per_word": pooled_tpw,
        "pooled_words_per_token": (1.0 / pooled_tpw) if pooled_tpw else float("nan"),
        "est_total_tokens": est_total_tokens,
        "per_source": per_source,
    }

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    # Console summary.
    print("\n=== Per source ===")
    hdr = f"{'source':22} {'tot_words':>14} {'samp_words':>12} {'samp_tok':>12} {'tpw':>7} {'wpt':>7}"
    print(hdr)
    for f, d in per_source.items():
        print(f"{f:22} {d['total_words']:>14,} {d['sample_words']:>12,} "
              f"{d['sample_tokens']:>12,} {d['tokens_per_word']:>7.3f} {d['words_per_token']:>7.3f}")
    print("\n=== Aggregate ===")
    print(f"total corpus words      : {grand_total_words:,}")
    print(f"pooled tokens/word      : {pooled_tpw:.4f}")
    print(f"pooled words/token      : {1.0/pooled_tpw:.4f}")
    print(f"est total TEXT tokens   : {est_total_tokens:,.0f}  "
          f"(= sum_i total_words_i * tpw_i)")
    print(f"elapsed                 : {report['elapsed_sec']} s")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
