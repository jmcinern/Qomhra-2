# Agent4 — Training-time estimate (Qwen3-8B, 8-GCD FSDP on LUMI)

Stands up the **real** distributed training rig and measures throughput so we can
extrapolate full-corpus GPU-hours. **Quick first number** uses random-init Qwen3-8B
weights on **synthetic random tokens** — same FLOPs/step as the real run, no data or
checkpoint dependency. Real Agent2/Agent3 tokens drop in later via `data.py`
(`MemmapTokenDataset`) with no train-loop changes.

## What it does
- Qwen3-8B architecture (36 layers / 4096 hidden / 32 q-heads / 8 kv-heads), random init.
- Combined vocab = Qwen3 text (151936) + mHuBERT audio units (1000) + specials, padded to ×128.
- **FSDP** full-shard via `accelerate` (8B + Adam states don't fit one 64 GB GCD under DDP).
- bf16 mixed precision, activation checkpointing, seq len 4096.
- Logs `loss`, `lr`, `seconds_per_step`, **`tokens_per_second`** to **W&B** (main rank only).
- Optional **PyTorch profiler** → `<run>/profiler/*.pt.trace.json` (open in Perfetto).

## Layout
```
qomhra/main.py          Accelerator + FSDP plugin, build/prepare/train
qomhra/model.py         Qwen3-8B config + combined vocab (random init)
qomhra/data.py          SyntheticTokenDataset (+ MemmapTokenDataset stub for real tokens)
qomhra/train_utils.py   train loop + make_profiler (chrome trace)
qomhra/logging_utils.py W&B Logger (main-process-only)
qomhra/configs/default.yaml   all knobs
requirements-lumi.txt   net-new sqsh deps (hydra-core, pynvml)
setup_env.sh            build qomhra-env.sqsh inside the container
train.sh                SLURM launcher: 1 node / 8 GCD, RCCL env, CPU binds, RANK wiring
```

## Run on LUMI
Assumes the repo is at `/scratch/project_465002364/Qomhra-2/full-train-est` and the
shared container at `/scratch/project_465002364/Qomhra/Qomhra_v2.sif`.

```bash
cd /scratch/project_465002364/Qomhra-2/full-train-est/train

# 1. one-off: build the package overlay (only hydra-core + pynvml are net-new)
bash setup_env.sh /scratch/project_465002364/Qomhra/Qomhra_v2.sif

# 2. one-off: provide the W&B key (gitignored)
echo 'YOUR_WANDB_KEY' > /scratch/project_465002364/Qomhra-2/.wandb_key && chmod 600 $_

# 3. smoke test (short, with profiler) — confirm FSDP init, 8-GCD RCCL, no OOM,
#    wandb run appears, trace written. Best on an interactive node or debug queue.
sbatch train.sh profile.enabled=true optim.total_steps=30 logging.every_steps=5

# 4. timing run — steady-state seconds_per_step / tokens_per_second
sbatch train.sh optim.total_steps=400
```

Override any config knob on the CLI (Hydra), e.g. `data.micro_batch_size=2`,
`data.seq_len=8192`, `optim.grad_acc=4`.

## Reading the result → GPU-hours
Take steady-state **`tokens_per_second`** (world total, all 8 GCDs) from the W&B run, then:

```
total_tokens ≈ 2.13e9 (text, TEXT_TOKENS_SUMMARY.md) + audio_hrs × tokens_per_hr (Agent3)
seconds      = total_tokens / tokens_per_second
gpu_hours    = seconds / 3600 × 8        # 8 GCDs busy the whole time
wall_clock   = seconds / 3600            # single node
```
Multiply by number of epochs. The profiler trace shows where time goes (RCCL
all-gather/all-reduce vs. attention/MLP kernels vs. any dataloader stall) — the
low-hanging-fruit pass.

## Notes / things the LUMI smoke test will confirm
- FSDP plugin uses the **FSDP1** API (`ShardingStrategy.FULL_SHARD`, transformer
  auto-wrap on `Qwen3DecoderLayer`). If the container's `accelerate` defaults to
  FSDP2, adjust `build_fsdp_plugin()` in `main.py`.
- `attn_implementation=sdpa` (ROCm-safe). flash-attn can be tried later for speed.
- If a single GCD OOMs, keep `micro_batch_size=1`; if there's headroom, raise it
  (and lower `grad_acc`) for better throughput.
