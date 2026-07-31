# Supervised ASR data — setanta → LUMI (2026-07-17)

## What was done
1. Transferred the cleaned supervised manifest from setanta
   (`/media/storage_ssd/liam/NeMo/data/cleaned/train_cleaned_without_tg4_remove_100dups_labelfiltered.jsonl`, 68 MB)
   to LUMI via agent-forwarded rsync (`ssh -A setanta` → rsync to `mcinerne@lumi.csc.fi`).
2. Transferred the 329,390 WAVs it references (only those, not the full 391 K-file dir)
   from setanta `/media/storage/liam/asr_data_nemo/from_storage_ssd/` — 45 GB.
3. Converted jsonl → parquet with `jsonl_to_parquet.py` (run via `run_jsonl_to_parquet.sh`,
   `Qomhra_v2.sif`), rewriting `audio_filepath` to absolute LUMI paths
   (same convention as `audio/unlabelled_full/manifest.tsv`).
4. Verified: 329,390 rows == jsonl lines == WAV count; random 200-path sample all exist.

## ASR data status on LUMI
| Item | Path | Size |
|---|---|---|
| Supervised transcripts (parquet) | `/scratch/project_465002364/Denorm/train/data/supervised_ASR.parquet` | 18 MB, 329,390 rows, 413.4 h |
| Supervised audio (flat WAV dir) | `/scratch/project_465002364/audio/supervised_asr/` | 45 GB, 329,390 files |
| Unlabelled audio (full) | `/scratch/project_465002364/audio/unlabelled_full/` (+ `manifest.tsv`) | 998 GB |
| Conversations audio | `/scratch/project_465002364/audio/conversations/` | 281 GB |
| Source jsonl (staging copy) | `/scratch/project_465002364/Denorm/train/data/staging/` | 68 MB |

Parquet schema: `audio_filepath` (absolute LUMI path), `duration` (s), `text`.
Note: `ablations/DATA_OVERVIEW.md`'s claim that the bulk audio is setanta-only is stale —
`unlabelled_full` is already on LUMI.
