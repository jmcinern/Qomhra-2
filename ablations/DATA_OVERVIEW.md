# Ablations — Data Overview & Transfer Guide

Written for the agent running the ablations. Answers two things:
**(a)** how much of each data source is on LUMI, in what form, and where;
**(b)** how to move the missing audio from setanta → LUMI.

All paths verified on LUMI 2026-07-02. Token counts are Qwen3-tokenized (from
`tokens/meta.json`); audio "tokens" = mHuBERT-147 discrete units.

The three ablation modalities are **Text**, **Speech**, and **ASR transcripts**.
For clean, unconfounded ablations everything should be equalised by **token
budget**, not by hours or files (see "How to Prepare" under each source).

---

## 1. TEXT

**Summary:** Done — the full ~2.13 B-token Irish text corpus is tokenized and on LUMI.

**Where:**
- Tokenized: `/scratch/project_465002364/Qomhra-2/full-train-est/tokens/`
  - `train.bin` — 8.5 GB, all sources concatenated (uint32)
  - `shards/<source>__rgNN.bin` — per-source shards (use these to control the mix)
  - `meta.json` — token/doc counts + shard manifest per source
- Raw (pre-tokenization): `/scratch/project_465002364/Denorm/train/data/*.parquet` (2.9 GB)

**What Form:** pre-tokenized **uint32 `.bin`** memmap, Qwen3 tokenizer, docs
separated by `sep_id = 151643`. Raw form is `.parquet` (one file per source).
Combined-vocab audio offset is +151936, so text ids sit in `[0, 151936)`.

**Per-source token counts** (total **2,128,000,774 tok / 7,636,604 docs**):

| source              | tokens        | docs      | note |
|---------------------|---------------|-----------|------|
| hplt3_mono          | 1,199,543,583 | 786,690   | web mono, largest |
| finepdfs            | 407,734,976   | 33,087    | PDF-derived |
| corpas_full_clean   | 246,218,867   | 4,925,026 | curated corpus |
| **conversations_ga**| 182,719,639   | 19,267    | **this is the ASR set — see §3** |
| nce                 | 72,798,909    | 1,776,221 | |
| uni-archives        | 17,982,388    | 96,113    | |
| oireachtas_ga       | 1,002,412     | 200       | tiny |

**How to Prepare for Ablations:** nothing to transfer. To build a fixed-size text
budget, read `shards/*.bin` and take the first N uint32 ids per source (or sample
docs on `sep_id` boundaries). If you want a *pure* text set, **exclude
`conversations_ga`** so it doesn't double-count as ASR (§3). The throughput rig's
`PackedTokenDataset` in `full-train-est/train/qomhra/data.py` already reads this
`.bin` format and is the reference loader.

---

## 2. SPEECH (audio → mHuBERT units)

**Summary:** Mostly NOT on LUMI — only a ~10 h validation subset is tokenized; the
~10 K-hour target corpus still lives on setanta and must be transferred + tokenized.

**Where:**
- Units on LUMI (subset only): `/scratch/project_465002364/Qomhra-2/full-train-est/mhubert/units/`
  — **45 `.npy` files, 4.4 MB, ~10.14 h**, in per-source subdirs
  (`podcasts_jul24_*`, `12TB_*`, `soundcloud_*`, `TG4`, `religious`, …).
- Raw 10 h WAVs on LUMI: `/scratch/project_465002364/audio/unlabelled_subset_10h/` (1.3 GB, `manifest.tsv` inside).
- **Full corpus (~10 K h): on setanta only** — `/media/storage/phonetics/asr_data_irish/unlabelled`
  and `/media/storage_12TB/unlabelled/...`. Not on LUMI.

**What Form:** units = **`.npy` uint16**, mHuBERT-147 unit ids in `[0, 1000)` at
**50 Hz** (~180 K units/hour, no dedup). In combined vocab they get **+151936**
offset. Raw audio = **16 kHz mono 16-bit PCM WAV** (already transcoded on setanta —
some master lists name `.mp3` but the files on disk are `.wav`).

**How to Prepare for Ablations:**
1. Transfer the bulk audio setanta → LUMI (see §4).
2. Tokenize to units with the mHuBERT pipeline (`full-train-est/mhubert/`, same one
   that produced the 45-file subset).
3. ~10 K h × 180 K units/h ≈ **1.8 B units** — comparable to the 2.13 B text budget,
   so token-matched ablations are feasible. Only ~10 h (<0.2 %) is present today;
   this is the critical-path Job-1 transfer.

---

## 3. ASR TRANSCRIPTS

**Summary:** Done — the ASR transcripts are the `conversations_ga` source, already
tokenized on LUMI (~183 M tokens; README's "~200 M" figure).

**Where:** same `tokens/` dir as Text:
- shard `shards/conversations_ga__rg00.bin` (731 MB, uint32)
- also inside `train.bin`
- raw: `/scratch/project_465002364/Denorm/train/data/conversations_ga.parquet` (240 MB)

**What Form:** Qwen3 uint32 `.bin` tokens (identical format to Text). These are
text transcripts of speech, so they occupy the text id range, not the audio range.

**How to Prepare for Ablations:** already tokenized — just select the
`conversations_ga` shard as its own modality. **Keep it out of the pure-Text mix**
to avoid confounding Text vs ASR comparisons. 182.7 M tokens sets a natural common
budget if you want all three modalities size-matched at ~180–200 M tokens for the
cleanest (smallest-common-denominator) ablation.

---

## 4. HOW TO TRANSFER AUDIO: setanta → LUMI (direct, agent-forwarded)

The trick: **hop onto setanta with your local SSH agent forwarded (`ssh -A`), then
rsync straight to LUMI from setanta** — the forwarded `id_ed25519` key authenticates
the setanta→LUMI leg, so the bytes never bounce through your laptop.

```bash
# 1. From your machine, land on setanta WITH agent forwarding.
#    (setanta itself is reached via ProxyJump phoneticsrv3, already in ~/.ssh/config)
ssh -A setanta

# 2. From setanta, rsync directly to LUMI scratch. Example for one source tree:
rsync -av --info=progress2 \
  /media/storage/phonetics/asr_data_irish/unlabelled/ \
  mcinerne@lumi.csc.fi:/scratch/project_465002364/audio/unlabelled_full/

#    Second location (note the .mp3-named-but-actually-.wav files here):
rsync -av --info=progress2 \
  /media/storage_12TB/unlabelled/ \
  mcinerne@lumi.csc.fi:/scratch/project_465002364/audio/unlabelled_full_12TB/
```

**Gotchas (learned building the 10 h subset):**
- `wavs.dur.list` mixes **relative** entries (under the first dir) and **absolute**
  entries under `/media/storage_12TB/unlabelled/...` — resolve both.
- Under `/media/storage_12TB/...`, `wavs.dur` lists **`.mp3`** filenames but the
  files on disk are transcoded **`.wav`** — swap the extension when resolving paths.
- Filenames contain spaces, Irish unicode, and `：`/brackets — rsync handles them
  fine; avoid hand-built shell loops that word-split on spaces.
- setanta `/media/storage` is world-writable but ~98 % full (~158 G free) — don't
  stage large copies there; rsync source→LUMI directly.
- LUMI scratch quota is 55 T (was ~669 G used) — 10 K h of 16 kHz WAV is on the
  order of a few hundred GB, well within budget.

**Access reference:**
- `ssh lumi` → `lumi.csc.fi`, user `mcinerne`, key `~/.ssh/id_ed25519`, project `project_465002364`.
- `ssh setanta` → `134.226.89.152`, user `joey`, via ProxyJump `joeymci@phoneticsrv3.lcs.tcd.ie`.
- Use `ssh -A` for any git/rsync leg that needs your key on the far side.

---

## Quick status table

| Modality | On LUMI? | Amount present | Form | Action needed |
|----------|----------|----------------|------|---------------|
| Text     | ✅ full   | 2.13 B tok (7 sources) | uint32 `.bin` + parquet | none |
| ASR      | ✅ full   | 182.7 M tok (`conversations_ga`) | uint32 `.bin` + parquet | keep separate from Text |
| Speech   | ⚠️ subset | ~10.14 h units (45 `.npy`) of ~10 K h | uint16 `.npy` units / WAV | **transfer bulk from setanta (§4) + tokenize** |
