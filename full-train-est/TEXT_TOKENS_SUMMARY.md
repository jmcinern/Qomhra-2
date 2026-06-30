# Agent2 — Text words/token ratio (Qwen3-8B tokenizer)

**Run:** SLURM job on LUMI `debug` partition, 128 CPU cores, inside `Qomhra_v2.sif`.
Single streaming pass over all 7 text parquet files. **Elapsed: 38.6 s.**

- **Word definition:** whitespace split (`str.split()`).
- **Token definition:** Qwen3-8B tokenizer ids, `add_special_tokens=False` (no chat template, no truncation — long-doc warnings are harmless, counts are full).
- **Sampling:** uniform per-row acceptance `p = 0.00694` (≈ size-proportional, since words ∝ bytes). Small files (< 5 MB) sampled in full. ~7.0 M words sampled total.
- **Full word count is exact** (every row processed), not extrapolated.

## Headline

| Metric | Value |
|---|---|
| Total corpus words (exact) | **890,675,109** (~890 M) |
| Pooled tokens/word | **2.368** |
| Pooled words/token | **0.422** |
| **Est. total text tokens** | **≈ 2.13 billion** |

Estimate = Σ_source ( total_words_source × tokens_per_word_source ).

## Per source

| source | total words | sample words | sample tokens | tkn/word | word/tkn |
|---|---:|---:|---:|---:|---:|
| conversations_ga | 91,515,834 | 752,449 | 1,505,461 | 2.001 | 0.500 |
| corpas_full_clean | 101,764,312 | 699,780 | 1,661,577 | 2.374 | 0.421 |
| finepdfs | 157,390,996 | 1,326,407 | 3,458,935 | 2.608 | 0.383 |
| hplt3_mono | 497,309,731 | 3,475,673 | 8,383,969 | 2.412 | 0.415 |
| nce | 34,471,908 | 237,357 | 490,077 | 2.065 | 0.484 |
| oireachtas_ga | 477,517 | 477,517 (full) | 1,002,212 | 2.099 | 0.476 |
| uni-archives | 7,744,811 | 57,204 | 134,381 | 2.349 | 0.426 |

Per-source tkn/word spans 2.0–2.6; the corpus is dominated by `hplt3_mono` (~56% of words).

## For the full-run extrapolation

`total_text_tokens ≈ total_words × 2.37 ≈ 2.13e9`. Combine with the audio side:

> Total tokens = (total_audio_hrs × tkn/hr) + (total_words × **2.37**)

## Reproduce

```bash
# on LUMI
cd /scratch/project_465002364/Qomhra-2/full-train-est
sbatch run_token_ratio.sh   # writes text_token_ratio_report.json
```

Tokenizer is pre-cached at `hf_cache/` (loaded by local path to avoid the
transformers offline-mode network probe). Files: `text_token_ratio.py`,
`run_token_ratio.sh`, `text_token_ratio_report.json`.
