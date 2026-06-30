# Agent4 — Training Setup & Timing Plan

> **STATUS (implemented):** code + LUMI scripts are in `train/` (see `train/README.md`).
> Quick first number uses **random-init weights** on synthetic tokens (same FLOPs/step
> as the real checkpoint — no download needed). Real-weights path documented for later.

## Context
We need a wall-clock / GPU-hour estimate for a full text+audio pretraining run of
Qwen3-8B on LUMI (1 node, 8 GCDs). Agent2 (text tokens) and Agent3 (audio tokens)
are still tokenizing, so this plan **stands up the full distributed training rig now
using synthetic randomly-generated tokens**, measures throughput, and is built so the
real token streams drop in later with a one-line dataset swap.

Deliverables:
1. A minimal decoder-only causal-LM training package on LUMI.
2. A timed 8-GCD FSDP run on synthetic tokens, logged to **Weights & Biases**.
3. A **PyTorch profiler chrome-trace JSON** (loadable in Perfetto) from an early window.
4. A short throughput → GPU-hour extrapolation using the Agent2/Agent3 token totals.

We **reuse** the proven infrastructure from `D:\VS-code-projects\Denorm\train\nanoT5`
rather than writing from scratch:
- `nanoT5/main.py` — `Accelerator` + `accelerator.prepare(...)` train entry.
- `utils/logging_utils.py::Logger` — main-process-only **wandb** init + `log_stats`.
- `utils/train_utils.py::make_profiler` — already writes a chrome trace to `<run>/profiler`.
- `setup_packages.sh` — venv-in-container → squashfs overlay (avoids Lustre small-file penalty).
- `train_denorm.sh` — SLURM launcher: CPU binds, `RANK/LOCAL_RANK/WORLD_SIZE` wiring, container exec.

## Key decisions (confirmed)
- **Sharding:** accelerate-native **FSDP** full-shard (8B won't fit DDP: Adam states ~96 GB/GPU).
- **Weights:** **random-init** Qwen3-8B for the quick first number (same FLOPs/step; no
  download). Real pretrained path (for a loss curve) documented in `train/README.md`.
- **Sequence length:** **4096**.
- **Precision:** bf16 mixed; activation (gradient) checkpointing ON.

## Combined vocab scheme (so real tokens are a drop-in later)
One shared embedding table over a unified id space:
- text ids: Qwen3 native `[0, 151936)` (sep/eos = 151643, per `tokenize_corpus.py`).
- audio ids: mHuBERT units `[0, 1000)` (per `audio-tknz.py`) → offset to `[151936, 152936)`.
- a few special tokens (audio-BOS / audio-EOS / modality-switch) appended after that.
- `vocab_size` rounded up to a multiple of 128 for matmul efficiency (~153 088).
Model embeddings are resized to this size via `resize_token_embeddings(..., pad_to_multiple_of=128)`;
the new audio/special rows are randomly initialised on top of the real Qwen3 weights.

---

## Directory layout (new, under `full-train-est/`)
```
full-train-est/train/
  qomhra/
    main.py            # adapted from nanoT5/main.py (Accelerator + FSDP)
    data.py            # SyntheticTokenDataset now; MemmapTokenDataset later
    model.py           # Qwen3-8B load + embedding resize + grad-checkpoint
    logging_utils.py   # Logger (wandb), copied/trimmed from nanoT5
    train_utils.py     # train loop + make_profiler, trimmed for causal LM
    configs/default.yaml
  requirements-lumi.txt # hydra-core, pynvml (torch/transformers/accelerate are in the .sif)
  setup_env.sh          # build qomhra-env.sqsh (adapted from setup_packages.sh)
  cache_model.sh        # one-off: download Qwen3-8B into HF_HOME (login node)
  fsdp_config.yaml      # accelerate FSDP plugin settings
  train.sh              # SLURM launcher (adapted from train_denorm.sh)
```

---

## Step-by-step

### 1. Scaffold the training package
Copy and trim the four reusable nanoT5 utils into `train/qomhra/`. Strip T5/seq2seq
specifics (predict/rouge/eval-generate); keep `Accelerator`, the `Logger` (wandb),
`make_profiler`, the grad-accum/no_sync loop, and `maybe_logging`. Loss comes straight
from `model(**batch).loss` (HF causal LM shifts labels internally).

### 2. Synthetic dataset now, real tokens later (`data.py`)
- `SyntheticTokenDataset`: an `IterableDataset` yielding `{input_ids, labels}` of length
  4096, ids drawn `torch.randint(0, vocab_size, ...)`, `labels = input_ids.clone()`.
  Pure on-GPU-bound load → exercises the real compute/comms path without I/O noise.
- Stub `MemmapTokenDataset` (commented) that will `np.memmap` Agent2 `train.bin` (uint32)
  + concatenated Agent3 `.npy` audio units (offset), packed into 4096 windows. **Drop-in
  swap** once tokenization lands — no training-loop changes.

### 3. Model (`model.py`)
- `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")` (from local `HF_HOME` cache).
- `resize_token_embeddings(153088, pad_to_multiple_of=128)`.
- `model.gradient_checkpointing_enable()`; `attn_implementation="sdpa"` (ROCm-safe;
  flash-attn optional later); `config.use_cache=False`.

### 4. FSDP via accelerate (`fsdp_config.yaml`)
Full-shard, `Qwen3DecoderLayer` auto-wrap policy, bf16 mixed precision, `limit_all_gathers`,
backward prefetch, `use_orig_params=true`. `Accelerator(mixed_precision="bf16")` reads the
plugin from the accelerate env/config. (Hold `torch.compile` off for the first timing run.)

### 5. W&B logging
Reuse `Logger.setup_wandb` verbatim (main-process-only init). Log per `logging.every_steps`:
`loss`, `lr`, `seconds_per_step`, and a derived **`tokens_per_second`**
(= global_batch × seq_len / seconds_per_step) — the number the estimate is built on.
`WANDB_API_KEY` provided on LUMI exactly as in `train_denorm.sh` (gitignored `.wandb_key`).

### 6. Profiler → Perfetto JSON
Reuse `make_profiler` unchanged (gated by `profile.enabled`, rank-0 only, `wait/warmup/active`
window, writes chrome trace to `<run>/profiler`). The emitted `*.pt.trace.json` opens directly
in Perfetto / `chrome://tracing`.

### 7. Environment on LUMI
- `setup_env.sh` (one-off): adapt `setup_packages.sh` to build `qomhra-env.sqsh` from a
  `--system-site-packages` venv inside `Qomhra_v2.sif`; overlay only `hydra-core` + `pynvml`
  (torch 2.5.1+rocm6.2 / transformers / accelerate / wandb are already in the .sif).
- `cache_model.sh` (one-off, **login node, online**): `huggingface-cli download Qwen/Qwen3-8B`
  into `HF_HOME=.../full-train-est/train/hf_cache` so compute nodes can run with
  `HF_HUB_OFFLINE=1`.

### 8. SLURM launcher (`train.sh`)
Adapt `train_denorm.sh` 1:1:
- `#SBATCH` 1 node / 8 tasks / 8 GPUs / 7 cpus-per-task / `standard-g` / `mem=0`.
- The 8-way `CPU_BIND` mask_cpu string (verbatim — places each rank's cores on the L3
  nearest its GCD).
- `module use /appl/local/containers/ai-modules; module load singularity-AI-bindings`.
- Per-rank wrapper: `singularity exec` with `-B qomhra-env.sqsh:/opt/qomhra-env:image-src=/`,
  scratch bind, `PYTHONPATH=/opt/qomhra-env`, `HF_HOME`, `HF_HUB_OFFLINE=1`,
  `RANK=$SLURM_PROCID`, `LOCAL_RANK=$SLURM_LOCALID`, `WORLD_SIZE=$SLURM_NTASKS`,
  `MASTER_ADDR/PORT`.
- **RCCL high-speed interconnect** env (to confirm against `D:/VS-code-projects/LUMI-AI-Guide`
  before first run): `MPICH_GPU_SUPPORT_ENABLED=1`, the aws-ofi-rccl plugin from
  singularity-AI-bindings, `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`, `NCCL_NET_GDR_LEVEL=PHB`.
- `srun --cpu-bind=$CPU_BIND ...` tee'd to a timestamped log.

### 9. Smoke test → timing run
- **Smoke** (interactive `salloc` or `debug`, ~30 steps): confirm FSDP init, 8-GCD RCCL,
  no OOM, wandb run appears, profiler json written. Run with `profile.enabled=true total_steps=30`.
- **Timing**: `profile.enabled=false`, ~300–500 steps after warmup; read steady-state
  `seconds_per_step` / `tokens_per_second` from wandb.

### 10. Extrapolate to the full run
`total_tokens = text (≈2.13B, TEXT_TOKENS_SUMMARY.md) + audio (Agent3 hrs × tkn/hr)`.
`gpu_hours = total_tokens / tokens_per_second / 3600 × 8 (GCDs)` × `epochs`.
Report wall-clock (gpu_hours / 8) and note profiler-identified low-hanging fruit.

---

## Verification
- `accelerator.state` log shows `distributed_type=FSDP`, `num_processes=8`.
- `rocm-smi` (or `pynvml`) shows balanced ~real memory use across all 8 GCDs, no OOM.
- A wandb run with decreasing-ish loss on synthetic data is sanity (loss ≈ ln(vocab) ≈ 11.9 at
  step 0 confirms the head/vocab wiring) and steady `tokens_per_second`.
- `<run>/profiler/*.pt.trace.json` opens in Perfetto; RCCL all-reduce / all-gather visible.

## Dependencies / open items
- **Agent2/Agent3 token formats** are already known (uint32 `train.bin` + meta.json; uint16
  `.npy` units in `[0,1000)`); only the **final files** are pending → swap `data.py` when ready.
- Confirm exact LUMI RCCL env vars against `LUMI-AI-Guide` at implementation time.
- Git: coordinate pushes with other agents on this project (shared repo).
