#!/usr/bin/env python3
"""Ablations — parallel Qwen2.5-Omni text tokenizer with a cached token stream.

Tokenizes the Irish text parquet sources with the **Qwen2.5-Omni-3B** tokenizer
(add_special_tokens=False) — the exact tokenizer of the ablation base model, so the
emitted ids line up with the Omni model's text embeddings. Output format is
byte-compatible with the reference loader in
`ablations/train/qomhra/data.py` (uint32 `.bin`, documents separated by `sep_id`):

  out-dir/shards/<source>__rg<NN>.bin   uint32 little-endian, one per (file,row_group)
  out-dir/train.bin                      all shards concatenated (deterministic order)
  out-dir/meta.json                      dtype, sep id, counts, throughput, sampling info

Work is split one task per (file, row_group) across a multiprocessing Pool — sized
for a LUMI-C `small` node (128 cores). Each worker streams its row group in small
row batches and appends to its shard incrementally, so worker RAM is bounded.

Subset mode (--word-budget N): draw a size-proportional sample of whole documents
so the per-source WORD share is proportional to that source's byte size (uniform
per-document acceptance over a corpus whose doc sizes vary — no assumption that
docs are equal length). Small files are taken in full. This is the same sampling
scheme as `full-train-est/text_token_ratio.py`, reused here for a fast, honest
throughput benchmark. --word-budget 0 tokenizes the FULL corpus.

The `.bin` shards are the cache: a shard that already exists (non-empty) is skipped
on re-run unless --force. With a fixed --seed the sample is deterministic, so skips
are safe.
"""
import os
import json
import math
import time
import random
import argparse
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = "/scratch/project_465002364/Denorm/train/data"
# All Irish text sources (see ablations/DATA_OVERVIEW.md). conversations_ga is the
# ASR-transcript source; keep it here for a complete text tokenizer, select/exclude
# it downstream when composing the ablation mix.
FILES = [
    "conversations_ga",
    "corpas_full_clean",
    "finepdfs",
    "hplt3_mono",
    "nce",
    "oireachtas_ga",
    "uni-archives",
]
DTYPE = np.uint32          # Qwen2.5 vocab (>151k) needs >16 bits
ROW_BATCH = 256            # rows per tokenizer call (bounds worker memory)

# Globals set once per worker (fork-inherited / initializer).
_tok = None
_sep = None
_shard_dir = None


def _init_worker(model_path, sep_id, shard_dir):
    global _tok, _sep, _shard_dir
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Cap intra-op thread pools: we already parallelise across processes, so letting
    # each of 128 workers spin up its own tokenizer threads would oversubscribe.
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from transformers import AutoTokenizer
    try:
        _tok = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        # Omni repo may register a custom tokenizer class.
        _tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _sep = sep_id
    _shard_dir = shard_dir


def _process_unit(unit):
    # word_target: stop after ~this many words are tokenized (subset mode); None
    # tokenizes the whole row group (full corpus).
    source, path, rg, word_target, seed = unit
    rng = random.Random(seed)
    shard = os.path.join(_shard_dir, f"{source}__rg{rg:02d}.bin")

    n_tokens = 0
    n_docs = 0            # documents actually tokenized (post-sampling)
    n_words = 0           # words in tokenized docs (for the budget report)
    n_seen_docs = 0       # documents scanned (pre-sampling)
    tok_seconds = 0.0     # time spent inside the tokenizer (pure throughput)
    reached = False       # hit the per-unit word target

    pf = pq.ParquetFile(path)
    with open(shard, "wb") as out:
        for batch in pf.iter_batches(batch_size=ROW_BATCH, columns=["text"],
                                     row_groups=[rg]):
            rows = [t for t in batch.column("text").to_pylist() if t]
            n_seen_docs += len(rows)
            if word_target is not None:
                # Shuffle within the batch so the subset isn't purely the file head.
                rng.shuffle(rows)
            texts = []
            for t in rows:
                texts.append(t)
                n_words += len(t.split())
                if word_target is not None and n_words >= word_target:
                    reached = True
                    break     # accept whole docs; overshoot by at most one doc
            if not texts:
                if reached:
                    break
                continue
            t0 = time.perf_counter()
            enc = _tok(texts, add_special_tokens=False)["input_ids"]
            tok_seconds += time.perf_counter() - t0
            flat = []
            for ids in enc:
                flat.extend(ids)
                flat.append(_sep)        # document separator
                n_docs += 1
            arr = np.asarray(flat, dtype=DTYPE)
            arr.tofile(out)
            n_tokens += arr.size
            if reached:
                break

    # Drop an empty shard so cache-skip logic (exists & non-empty) stays clean.
    if n_tokens == 0:
        try:
            os.remove(shard)
        except OSError:
            pass

    return {"source": source, "rg": rg, "shard": shard,
            "n_tokens": n_tokens, "n_docs": n_docs, "n_words": n_words,
            "n_seen_docs": n_seen_docs, "tok_seconds": tok_seconds,
            "skipped": False}


def _resolve_sep_id(model_path, override):
    if override is not None:
        return int(override)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sid = tok.convert_tokens_to_ids("<|endoftext|>")
    if sid is None or sid < 0:
        sid = tok.eos_token_id if tok.eos_token_id is not None else 151643
    return int(sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model", required=True, help="local Qwen2.5-Omni-3B tokenizer dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--files", nargs="+", default=FILES,
                    help="subset of sources to tokenize (default: all)")
    ap.add_argument("--word-budget", type=int, default=0,
                    help="approx total sampled words, size-proportional across "
                         "sources; 0 = tokenize the full corpus")
    ap.add_argument("--sep-id", type=int, default=None,
                    help="document separator id (default: auto <|endoftext|>)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force", action="store_true",
                    help="re-tokenize even if a shard already exists")
    args = ap.parse_args()

    t0 = time.time()
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    sep_id = _resolve_sep_id(args.model, args.sep_id)

    # File sizes -> each source gets a hard WORD target proportional to its byte
    # size (target_words_f = budget * bytes_f / total_bytes). That target is split
    # evenly across the source's row groups, and each worker stops once it reaches
    # its slice. Proportionality is then exact and independent of any bytes/word
    # guess. --word-budget 0 => no target (full corpus).
    sizes, nrgs = {}, {}
    for f in args.files:
        sizes[f] = os.path.getsize(os.path.join(args.data_dir, f + ".parquet"))
        nrgs[f] = pq.ParquetFile(os.path.join(args.data_dir, f + ".parquet")).num_row_groups
    total_bytes = sum(sizes.values())
    src_target = {}
    if args.word_budget > 0:
        for f in args.files:
            src_target[f] = args.word_budget * sizes[f] / total_bytes
    print(f"model={args.model}\nsep_id={sep_id} dtype={np.dtype(DTYPE).name} "
          f"workers={args.workers}", flush=True)
    print(f"sources={args.files}\ntotal_bytes={total_bytes:,} "
          f"word_budget={args.word_budget or 'FULL'}", flush=True)

    # Build work units: one per (file, row_group), each carrying its per-unit word
    # target (or None for the full corpus).
    units, skipped = [], []
    for f in args.files:
        path = os.path.join(args.data_dir, f + ".parquet")
        # Ceil so tiny sources still get >=1 doc rather than rounding to zero.
        unit_target = (max(1, math.ceil(src_target[f] / nrgs[f]))
                       if args.word_budget > 0 else None)
        for rg in range(nrgs[f]):
            shard = os.path.join(shard_dir, f"{f}__rg{rg:02d}.bin")
            if not args.force and os.path.exists(shard) and os.path.getsize(shard) > 0:
                skipped.append((f, rg, shard))
                continue
            units.append((f, path, rg, unit_target, args.seed + rg * 100 + hash(f) % 97))
    print(f"{len(units)} work units to tokenize, {len(skipped)} cached (skipped)",
          flush=True)

    results = []
    if units:
        with Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.model, sep_id, shard_dir)) as pool:
            for r in pool.imap_unordered(_process_unit, units):
                results.append(r)
                print(f"  done {r['source']} rg{r['rg']:02d}: "
                      f"{r['n_tokens']:,} tok, {r['n_docs']:,}/{r['n_seen_docs']:,} docs",
                      flush=True)

    # Fold in cached shards so train.bin + meta stay complete on resume.
    for f, rg, shard in skipped:
        n_bytes = os.path.getsize(shard)
        results.append({"source": f, "rg": rg, "shard": shard,
                        "n_tokens": n_bytes // np.dtype(DTYPE).itemsize,
                        "n_docs": 0, "n_words": 0, "n_seen_docs": 0,
                        "tok_seconds": 0.0, "skipped": True})

    # Deterministic order + per-source totals.
    results.sort(key=lambda r: (r["source"], r["rg"]))
    per_source, total_tokens, total_docs, total_words, tok_seconds = {}, 0, 0, 0, 0.0
    for r in results:
        s = per_source.setdefault(r["source"], {"n_tokens": 0, "n_docs": 0,
                                                 "n_words": 0, "shards": []})
        s["n_tokens"] += r["n_tokens"]
        s["n_docs"] += r["n_docs"]
        s["n_words"] += r["n_words"]
        s["shards"].append(os.path.basename(r["shard"]))
        total_tokens += r["n_tokens"]
        total_docs += r["n_docs"]
        total_words += r["n_words"]
        tok_seconds += r["tok_seconds"]
    # Annotate each source with its size share + proportional word target.
    for f, s in per_source.items():
        s["size_frac"] = round(sizes[f] / total_bytes, 6)
        s["target_words"] = round(src_target[f], 1) if args.word_budget > 0 else None

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

    elapsed = time.time() - t0
    # Pure tokenizer throughput = tokens emitted / summed in-tokenizer wall time
    # (parallel, so this exceeds elapsed; end-to-end docs/s uses elapsed).
    meta = {
        "model": args.model,
        "tokenizer": "Qwen2.5-Omni-3B",
        "dtype": np.dtype(DTYPE).name,
        "sep_id": sep_id,
        "sep_after_every_doc": True,
        "word_budget": args.word_budget,
        "sampling": ("size-proportional per-source word target"
                     if args.word_budget > 0 else "full corpus (no sampling)"),
        "seed": args.seed,
        "workers": args.workers,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "total_words": total_words,
        "train_bin": train_bin,
        "train_bin_bytes": written,
        "per_source": per_source,
        "elapsed_sec": round(elapsed, 2),
        "docs_per_sec": round(total_docs / elapsed, 1) if elapsed else 0.0,
        "end_to_end_tokens_per_sec": round(total_tokens / elapsed, 1) if elapsed else 0.0,
        "tokenizer_seconds_summed": round(tok_seconds, 2),
        "tokenizer_tokens_per_sec": round(total_tokens / tok_seconds, 1) if tok_seconds else 0.0,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n=== Done ===")
    for s, d in sorted(per_source.items()):
        print(f"{s:22} {d['n_tokens']:>15,} tok  {d['n_docs']:>10,} docs  "
              f"{d['n_words']:>12,} words")
    print(f"{'TOTAL':22} {total_tokens:>15,} tok  {total_docs:>10,} docs  "
          f"{total_words:>12,} words")
    print(f"train.bin: {train_bin}  ({written/1e9:.3f} GB, dtype={meta['dtype']}, "
          f"sep_id={sep_id})")
    print(f"elapsed: {meta['elapsed_sec']} s  |  "
          f"end-to-end {meta['end_to_end_tokens_per_sec']:,} tok/s  |  "
          f"tokenizer {meta['tokenizer_tokens_per_sec']:,} tok/s "
          f"(summed {meta['tokenizer_seconds_summed']} s over {args.workers} workers)")


if __name__ == "__main__":
    main()
