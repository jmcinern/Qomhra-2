"""Where does the HBM actually go? Stage-by-stage breakdown of one text step.

Motivation: `gradient_checkpointing=false` OOM'd at 60.84 GiB/GCD, but a 4B model on
8 GCDs should need roughly 6-7 GiB (bf16 params 8G/8, AdamW fp32 states 32G/8, grads
8G/8). Being ~10x over that means an assumption is wrong, and the throughput numbers
say nothing about which one. This prints the deltas so the answer is measured.

Run under the same launcher as training (8 ranks); rank 0 prints.
    srun ... python -m _probe_mem
"""
import functools
import os

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from hydra import compose, initialize
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from qomhra.main import resolve_layer_classes
from qomhra.model import apply_freezing, get_model

GIB = 2**30
RANK0 = os.environ.get("RANK", "0") == "0"


def note(label):
    if RANK0:
        print(f"{label:38s} alloc={torch.cuda.memory_allocated()/GIB:6.2f} GiB  "
              f"peak={torch.cuda.max_memory_allocated()/GIB:6.2f} GiB", flush=True)


def main():
    with initialize(config_path="qomhra/configs", version_base="1.1"):
        args = compose(config_name="omni_text", overrides=["data.micro_batch_size=1"])

    model, _ = get_model(args)
    apply_freezing(model, args)
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if RANK0:
        # If dtype is fp32 here, FSDP shards fp32 flat params and MixedPrecision only
        # casts for compute — the resident cost is 2x what "bf16 training" suggests.
        dt = next(model.parameters()).dtype
        print(f"[params] total={n_all/1e9:.2f}B trainable={n_train/1e9:.2f}B dtype={dt}")
        print(f"[expect] params {n_all*2/GIB:.1f} GiB bf16 / {n_all*4/GIB:.1f} GiB fp32 "
              f"(unsharded); AdamW fp32 m+v on trainable = {n_train*8/GIB:.1f} GiB")

    layer_classes = resolve_layer_classes(
        model, override_names=args.get("fsdp", {}).get("transformer_layer_cls", None))
    if RANK0:
        print(f"[wrap] classes={sorted(c.__name__ for c in layer_classes)}")

    plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=layer_classes),
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True, limit_all_gathers=True, use_orig_params=True,
    )
    acc = Accelerator(mixed_precision="bf16", fsdp_plugin=plugin)
    note("start")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-5)
    model, opt = acc.prepare(model, opt)
    note("after FSDP wrap")
    if RANK0:
        # The real check: how many FSDP units exist. One unit = nothing sharded per
        # layer, so every all-gather materialises the whole model.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        units = [m for m in model.modules() if isinstance(m, FSDP)]
        print(f"[wrap] FSDP units = {len(units)}  (1 means NOT sharded per layer)")
        flat = sum(p.numel() for m in units for p in
                   (getattr(m, "_flat_param", None),) if p is not None)
        print(f"[wrap] local flat-param elements = {flat/1e9:.2f}B "
              f"({flat*2/GIB:.1f} GiB bf16 / {flat*4/GIB:.1f} GiB fp32)")

    ids = torch.randint(0, 150000, (args.data.micro_batch_size, args.data.seq_len),
                        device=acc.device)
    out = model(input_ids=ids, labels=ids.clone())
    note("after forward")
    acc.backward(out["loss"])
    note("after backward")
    opt.step()
    note("after optimizer step (states materialised)")
    opt.zero_grad(set_to_none=True)
    note("after zero_grad")


if __name__ == "__main__":
    main()
