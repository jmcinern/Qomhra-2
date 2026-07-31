# Ablations — training rig (real base models)

Adapted from the validated `full-train-est/train/` timing rig (FSDP + Accelerate +
Hydra, combined text+audio vocab, W&B, profiler, LUMI SLURM launcher). The change
for ablations: train **real pretrained base models** with real loss curves, not a
random-init model for timing.

Two base models × three modality mixtures (Text / Speech / ASR), equalised by token
budget for clean, unconfounded comparison.

## What differs from the timing rig
- `qomhra/model.py` — loads pretrained weights (`from_pretrained`) + resizes the
  embedding table to the combined vocab. Qwen2.5-Omni loads via its multimodal
  wrapper; we train its **Thinker text decoder** (`model.backbone_attr=thinker`).
- `qomhra/main.py` — FSDP transformer auto-wrap is **model-generic**: the decoder
  layer class is resolved from the model's `_no_split_modules` (not hardcoded to
  Qwen3DecoderLayer), so one launcher trains both models.
- `qomhra/configs/` — `base.yaml` (shared) + `qwen3p5_2b.yaml`, `qwen25_omni_3b.yaml`.
- `data.py`, `train_utils.py`, `logging_utils.py` — reused from the rig unchanged
  (`data.py` already reads the real LUMI token formats).

## Data / tokenization
- Text ✅ and ASR ✅ (`conversations_ga`) are tokenized on LUMI, but with the
  **Qwen3-8B tokenizer**. Qwen3.5-2B likely shares that tokenizer (verify). **Omni
  uses the Qwen2.5 tokenizer → re-tokenize** text+ASR with
  `../../full-train-est/tokenize_corpus.py --model <omni tokenizer dir> --sep-id <id>`
  into `ablations/tokens_qwen25/`, then update `qwen25_omni_3b.yaml`'s `text_bin`,
  `text_vocab`, `sep_id`, `audio_offset` from the emitted `meta.json`.
- Speech (mHuBERT units) — only a ~10 h subset on LUMI today; the bulk transfer +
  tokenization is Job 1. The rig's `audio_units_dir` already points at the subset.

## Run on LUMI
```bash
cd /scratch/project_465002364/Qomhra-2/ablations/train

# 1. one-off: cache both base models offline (login node, online)
bash cache_models.sh

# 2. one-off: build the package overlay (hydra-core + pynvml + liger-kernel)
bash setup_env.sh /scratch/project_465002364/Qomhra/Qomhra_v2.sif

# 3. one-off: wandb key (gitignored)
echo 'YOUR_WANDB_KEY' > /scratch/project_465002364/Qomhra-2/.wandb_key && chmod 600 $_

# 4. smoke test each model (interactive salloc or debug queue, ~30 steps)
sbatch train.sh --config-name qwen3p5_2b   profile.enabled=true optim.total_steps=30
sbatch train.sh --config-name qwen25_omni_3b profile.enabled=true optim.total_steps=30
```

Override any knob on the CLI (Hydra), e.g. `data.source=synthetic`,
`data.audio_token_budget=1800000`, `data.micro_batch_size=2`.

## Smoke-test success signals
- `accelerator.state`: `distributed_type=FSDP`, `num_processes=8`.
- `[fsdp] wrapping layer classes: [...]` logs the resolved decoder-layer class.
- Embeddings resized to the combined vocab (banner `vocab (comb.)`).
- **Loss starts LOW on real Irish text** (pretrained weights loaded), not
  ≈ln(vocab) — the key signal vs the random-init rig. Decreasing loss on real
  tokens = correct.
- Balanced GPU memory across 8 GCDs (`rocm-smi`), no OOM; W&B run appears; profiler
  `*.pt.trace.json` written when `profile.enabled=true`.
