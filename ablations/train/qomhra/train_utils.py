"""Train loop + torch profiler.

Modality-aware: a global step is either a **text** step (NTP cross-entropy) or an
**audio** step (next-audio-frame regression) — never a mix. All grad-accumulation
micro-batches within a step share the modality, and every rank agrees on it via the
seeded `ModalityScheduler` (see data.py). Losses are logged on separate W&B series
(`train/loss_text`, `train/loss_audio`) because they are different objectives on
different scales; averaging them into one curve would be meaningless.
"""
import contextlib
import os
import time

import torch

from . import checkpoint as ckpt_utils


def build_sdpa_ctx(args, logger):
    """Optional context forcing the SDPA backend priority for the forward pass.

    The profiler showed attention running on the math backend (explicit bmm +
    softmax over seq^2 ≈ 24% of CUDA time). Forcing [flash, efficient] uses the
    fused kernels instead. Returns a no-arg context-manager factory.
    """
    names = args.model.get("sdpa_backends", None)
    if not names:
        return contextlib.nullcontext
    from torch.nn.attention import sdpa_kernel, SDPBackend
    mapping = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    backends = [mapping[n] for n in names]
    if os.environ.get("RANK", "0") == "0":
        logger.log_message(f"[sdpa] forcing backend priority: {list(names)}")
    return lambda: sdpa_kernel(backends)


def make_profiler(args, logger):
    """Optional torch profiler over a small early window of micro-batches.

    Gated by args.profile.enabled. Rank 0 only writes the trace + logs the table;
    otherwise every GCD dumps a large chrome trace. The emitted *.pt.trace.json
    opens directly in Perfetto / chrome://tracing. Near-zero overhead when disabled.
    """
    prof_cfg = args.get("profile", None)
    if not (prof_cfg and prof_cfg.get("enabled", False)):
        return contextlib.nullcontext()

    trace_dir = os.path.join(os.getcwd(), "profiler")
    os.makedirs(trace_dir, exist_ok=True)
    export_trace = torch.profiler.tensorboard_trace_handler(trace_dir)
    is_rank0 = os.environ.get("RANK", "0") == "0"

    def on_trace_ready(prof):
        if not is_rank0:
            return
        export_trace(prof)
        logger.log_message(
            "\n========== PROFILER (top ops by self CUDA time) ==========\n"
            + prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
            + f"\nchrome trace -> {trace_dir} (open in Perfetto)\n"
            + "=========================================================="
        )

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=prof_cfg.get("wait", 5),
            warmup=prof_cfg.get("warmup", 3),
            active=prof_cfg.get("active", 10),
            repeat=1,
        ),
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        with_stack=False,
    )


class LossWindow:
    """Accumulates per-modality losses between logging boundaries."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums, self.counts = {}, {}

    def add(self, out, grad_acc):
        # loss_aligned is ablation 4's speech->transcript CE. Kept separate from
        # loss_text: same units (nats/token) but a different task, and averaging them
        # would hide which one is moving.
        for key in ("loss_text", "loss_audio", "loss_aligned", "audio_target_var"):
            if key in out:
                self.sums[key] = self.sums.get(key, 0.0) + float(out[key]) / grad_acc
                self.counts[key] = self.counts.get(key, 0.0) + 1.0 / grad_acc

    def stats(self):
        # Each modality is averaged over the steps that *used* it, not over the
        # whole window — otherwise a 50/50 mix would halve both reported losses.
        return {k: self.sums[k] / self.counts[k] for k in self.sums}


def train(model, dataloaders, accelerator, optimizer, lr_scheduler, logger, args,
          global_batch, scheduler=None):
    model.train()

    grad_acc = args.optim.grad_acc
    sdpa_ctx = build_sdpa_ctx(args, logger)
    profiler = make_profiler(args, logger)
    prof = profiler.__enter__()
    profiling = prof is not None  # nullcontext().__enter__() returns None

    iters = {k: iter(v) for k, v in dataloaders.items()}
    ckpt_cfg = args.get("checkpoint", {})
    ckpt_dir = ckpt_cfg.get("dir", None) or os.path.join(os.getcwd(), "checkpoints")
    every = int(ckpt_cfg.get("every_steps", 0) or 0)

    window = LossWindow()
    window_start = time.time()
    window_units = 0            # text tokens or audio frames seen this window

    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.optim.total_steps + 1):
        # One modality for the whole step, identical on every rank.
        modality = scheduler.modality(step - 1) if scheduler else next(iter(iters))

        for micro in range(grad_acc):
            batch = next(iters[modality])
            # Only all-reduce grads on the accumulation boundary; skip the comms on
            # intermediate micro-batches (mathematically identical, far less RCCL).
            is_last = (micro == grad_acc - 1)
            sync_ctx = contextlib.nullcontext() if is_last else accelerator.no_sync(model)
            with sync_ctx, sdpa_ctx():
                out = model(**batch)
                accelerator.backward(out["loss"] / grad_acc)
            window.add(out, grad_acc)
            window_units += int(out.get("n_text_tokens", 0) or
                                out.get("n_audio_frames", 0))
            if profiling:
                prof.step()

        if args.optim.grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.optim.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accelerator.unwrap_model(model).update_ema()   # no-op unless target_detach=ema

        if step % args.logging.every_steps == 0:
            elapsed = time.time() - window_start
            sec_per_step = elapsed / args.logging.every_steps
            stats = window.stats()
            stats["lr"] = optimizer.param_groups[0]["lr"]
            stats["seconds_per_step"] = sec_per_step
            # units/s is world-wide: each rank counted only its own micro-batches.
            stats["units_per_second"] = (
                window_units * accelerator.num_processes / max(elapsed, 1e-9)
            )
            logger.log_stats(stats, step=step, prefix="train/")
            window.reset()
            window_start = time.time()
            window_units = 0

        if every and step % every == 0:
            ckpt_utils.save(model, accelerator, args, step, ckpt_dir)
            if accelerator.is_main_process:
                ckpt_utils.prune(ckpt_dir, ckpt_cfg.get("keep_last", 2))

    profiler.__exit__(None, None, None)

    if ckpt_cfg.get("save_final", True):
        out = ckpt_utils.save(model, accelerator, args, args.optim.total_steps, ckpt_dir)
        logger.log_message(f"[ckpt] final checkpoint -> {out}")
