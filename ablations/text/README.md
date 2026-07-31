# Ablations — Text tokenization (Qwen2.5-Omni)

Tokenizes the Irish text parquet corpus with the **Qwen2.5-Omni-3B** tokenizer —
the exact tokenizer of the ablation base model — and caches the result as a flat
uint32 token stream that `ablations/train/qomhra/data.py` reads directly.

## Files

| File | Purpose |
|------|---------|
| `tokenize_text.py` | Parallel CPU tokenizer. One task per (source, row_group) over a `multiprocessing.Pool`; size-proportional word-budget subsampling; uint32 `.bin` shards + `train.bin` + `meta.json`; shard-level caching. |
| `download_qwen_omni.sh` | One-time download of Qwen2.5-Omni-3B (weights + tokenizer) into the shared repo-root HF cache. Run on a LUMI **login node**. |
| `run_tokenize_text.sh` | LUMI-C `small` (128-core) SLURM job. Defaults to the ~100K-word subset benchmark. |

## Shared paths on LUMI (for the speech agents too)

- **HF cache (shared):** `/scratch/project_465002364/Qomhra-2/hf_cache`
  - set `HF_HOME` to this; run offline with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
  - **Qwen2.5-Omni-3B snapshot:** `${HF_HOME}/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/<hash>/`
    (weights + tokenizer; load with `AutoTokenizer.from_pretrained(<snapshot dir>)`)
- **Raw text parquet:** `/scratch/project_465002364/Denorm/train/data/*.parquet`
- **Subset tokens output:** `/scratch/project_465002364/Qomhra-2/ablations/text/tokens_subset/`
  - `train.bin` (uint32, all sources concatenated) · `shards/<source>__rgNN.bin` · `meta.json`

## Output format (matches the training loader)

- `uint32` little-endian token ids, `add_special_tokens=False`.
- Each document is followed by a separator id `sep_id` (auto-resolved to
  `<|endoftext|>`; recorded in `meta.json`). Split on `sep_id` to recover documents.
- Text ids occupy `[0, 151936)`; in the combined audio+text vocab, mHuBERT audio
  units are offset `+151936` (see `ablations/DATA_OVERVIEW.md`).

## Run it

```bash
# 0. one-time, on a LUMI login node:
bash ablations/text/download_qwen_omni.sh

# 1. subset benchmark (~100K words, size-proportional across sources):
sbatch ablations/text/run_tokenize_text.sh

# 1b. full corpus later:
WORD_BUDGET=0 OUTDIR=/scratch/project_465002364/Qomhra-2/ablations/text/tokens_full \
  sbatch ablations/text/run_tokenize_text.sh
```

`meta.json` reports per-source token/doc/word counts plus throughput
(`end_to_end_tokens_per_sec` and pure `tokenizer_tokens_per_sec`).

## Subset semantics

`--word-budget N` accepts each document with probability `min(1, N / est_total_words)`,
where `est_total_words` comes from total parquet bytes. Because acceptance is uniform
per document, each source's sampled **word** share ends up proportional to its byte
size — no assumption that documents are equal length. Small files (<5 MB) are taken
in full. `--word-budget 0` tokenizes everything. A fixed `--seed` makes the sample
deterministic, so cached shards can be safely reused (skip unless `--force`).
