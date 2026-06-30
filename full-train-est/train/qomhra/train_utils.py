"""Train loop + torch profiler — adapted from nanoT5/utils/train_utils.py.

Trimmed for decoder-only causal-LM timing: no eval/predict/generate. The point is
a clean steady-state `seconds_per_step` / `tokens_per_second` and a Perfetto trace.
"""
import contextlib
import os
import time

import torch


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


def train(model, dataloader, accelerator, optimizer, lr_scheduler, logger, args,
          global_batch):
    model.train()

    tokens_per_step = global_batch * args.data.seq_len
    sdpa_ctx = build_sdpa_ctx(args, logger)
    profiler = make_profiler(args, logger)
    prof = profiler.__enter__()
    profiling = prof is not None  # nullcontext().__enter__() returns None

    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    window_start = time.time()
    window_loss = 0.0

    data_iter = iter(dataloader)
    while step < args.optim.total_steps:
        batch = next(data_iter)
        micro += 1

        # Only all-reduce/all-gather grads on the accumulation boundary; skip the
        # comms on intermediate micro-batches (mathematically identical, far less
        # RCCL traffic). nanoT5 profiler showed this was ~40% of CUDA time.
        is_sync = (micro % args.optim.grad_acc == 0)
        sync_ctx = contextlib.nullcontext() if is_sync else accelerator.no_sync(model)
        with sync_ctx, sdpa_ctx():
            loss = model(**batch).loss
            accelerator.backward(loss / args.optim.grad_acc)
        window_loss += loss.detach().float().item() / args.optim.grad_acc

        if profiling:
            prof.step()

        if is_sync:
            if args.optim.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.optim.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.logging.every_steps == 0:
                elapsed = time.time() - window_start
                sec_per_step = elapsed / args.logging.every_steps
                # world tokens/sec across all GCDs (each step is one global batch).
                tok_per_sec = tokens_per_step / sec_per_step
                logger.log_stats(
                    {
                        "loss": window_loss / args.logging.every_steps,
                        "lr": optimizer.param_groups[0]["lr"],
                        "seconds_per_step": sec_per_step,
                        "tokens_per_second": tok_per_sec,
                    },
                    step=step,
                    prefix="train/",
                )
                window_start = time.time()
                window_loss = 0.0

    profiler.__exit__(None, None, None)
