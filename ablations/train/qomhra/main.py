"""Entry point — FSDP continued-pretraining of the Qwen2.5-Omni Thinker.

One launcher, three ablations (pick with Hydra):
    sbatch --nodes=2 train.sh --config-name omni_text
    sbatch --nodes=2 train.sh --config-name omni_speech
    sbatch --nodes=2 train.sh --config-name omni_both

Launched one process per GCD by srun (see ../train.sh), which sets RANK /
LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT — the same wiring works
unchanged across nodes, since RANK comes from the *global* SLURM_PROCID.
"""
import functools
import time

import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator, DataLoaderConfiguration, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import get_cosine_schedule_with_warmup

from .data import ModalityScheduler, get_dataloader
from .logging_utils import Logger
from .model import apply_freezing, get_model
from .train_utils import train


def resolve_layer_classes(model, override_names=None):
    """Decoder-layer module classes to hand FSDP's transformer auto-wrap policy.

    Prefer the explicit `override_names` from the config: for the speech ablations we
    must wrap BOTH the text decoder block and the audio-encoder block, and only the
    config knows whether the audio tower is trained. The ModuleList heuristic below
    is the fallback — it finds the largest `layers` stack, which is the text decoder.
    We deliberately do not trust the thinker's `_no_split_modules` (it advertises the
    encoders, not the text-decoder layer).
    """
    if override_names:
        names = set(override_names)
        classes = {type(mod) for mod in model.modules()
                   if type(mod).__name__ in names}
        missing = names - {c.__name__ for c in classes}
        if missing:
            raise RuntimeError(
                f"fsdp.transformer_layer_cls names not found in the model: {sorted(missing)}"
            )
        if classes:
            return classes

    best = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and name.split(".")[-1] == "layers" \
                and len(module) > 0:
            if best is None or len(module) > len(best):
                best = module
    if best is not None:
        return {type(best[0])}

    raise RuntimeError(
        "Could not resolve transformer layer class for FSDP wrap; "
        "set fsdp.transformer_layer_cls explicitly in the config."
    )


def build_fsdp_plugin(args, layer_classes):
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=layer_classes,
    )
    strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }[args.get("fsdp", {}).get("sharding_strategy", "full_shard")]
    reduce_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.get("fsdp", {}).get("reduce_dtype", "fp32")]
    return FullyShardedDataParallelPlugin(
        sharding_strategy=strategy,
        auto_wrap_policy=auto_wrap,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=reduce_dtype,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=bool(args.get("fsdp", {}).get("forward_prefetch", False)),
        limit_all_gathers=True,
        # Required: the Text ablation freezes the audio tower, so trainable and
        # frozen params share FSDP units.
        use_orig_params=True,
    )


@hydra.main(config_path="configs", config_name="omni_text", version_base="1.1")
def main(args):
    torch.manual_seed(args.seed)

    # Build the model FIRST (on CPU) so we can resolve its layer classes for the
    # FSDP auto-wrap policy before constructing the Accelerator.
    model, vocab = get_model(args)
    freeze_msg = apply_freezing(model, args)
    num_params = sum(p.numel() for p in model.parameters())
    layer_classes = resolve_layer_classes(
        model, override_names=args.get("fsdp", {}).get("transformer_layer_cls", None)
    )

    accelerator = Accelerator(
        mixed_precision=args.precision,
        fsdp_plugin=build_fsdp_plugin(args, layer_classes),
        dataloader_config=DataLoaderConfiguration(
            non_blocking=True, dispatch_batches=False
        ),
    )
    logger = Logger(args, accelerator)
    logger.log_message(freeze_msg)
    logger.log_message(
        f"[fsdp] wrapping layer classes: {sorted(c.__name__ for c in layer_classes)}"
    )

    dataloaders = get_dataloader(args, vocab)
    modality_sched = None
    if args.task.modality == "both":
        modality_sched = ModalityScheduler(
            args.optim.total_steps, args.task.text_step_frac, seed=args.seed
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.optim.lr,
        betas=tuple(args.optim.betas),
        weight_decay=args.optim.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.optim.warmup_steps,
        num_training_steps=args.optim.total_steps,
    )

    prepared = accelerator.prepare(
        model, optimizer, lr_scheduler, *dataloaders.values()
    )
    model, optimizer, lr_scheduler = prepared[:3]
    dataloaders = dict(zip(dataloaders.keys(), prepared[3:]))

    if args.model.get("compile", False):
        model = torch.compile(model)

    global_batch = (
        args.data.micro_batch_size * accelerator.num_processes * args.optim.grad_acc
    )
    logger.log_run_banner(args, vocab, num_params, global_batch)

    t0 = time.time()
    train(model, dataloaders, accelerator, optimizer, lr_scheduler, args=args,
          logger=logger, global_batch=global_batch, scheduler=modality_sched)
    accelerator.wait_for_everyone()
    logger.log_message(f"TOTAL train time: {time.time() - t0:.1f}s "
                       f"for {args.optim.total_steps} steps")
    logger.finish()


if __name__ == "__main__":
    main()
