"""Entry point — FSDP continued-pretraining of the Qwen2.5-Omni Thinker.

One launcher, several ablations (pick with Hydra):
    sbatch --nodes=2 train.sh --config-name omni_text
    sbatch --nodes=2 train.sh --config-name omni_unit_ntp_100h
    sbatch --nodes=2 train.sh --config-name omni_unit_asr_250h

The continuous-audio speech configs are superseded and live in
configs/legacy/ — see configs/legacy/README.md.

Launched one process per GCD by srun (see ../train.sh), which sets RANK /
LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT — the same wiring works
unchanged across nodes, since RANK comes from the *global* SLURM_PROCID.
"""
import functools
import json
import time

import hydra
import os
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from accelerate import Accelerator, DataLoaderConfiguration, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import get_cosine_schedule_with_warmup

from .data import get_dataloader, get_validation_dataloaders
from . import checkpoint as ckpt_utils
from .logging_utils import Logger
from .model import apply_freezing, get_model
from .plan import ModalityScheduler, build_training_plan
from .train_utils import evaluate_unit_ntp, train


def root_only_auto_wrap_policy(module, recurse, nonwrapped_numel):
    """Traverse the tree but never create a child FSDP unit."""
    return bool(recurse)


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
    wrap_policy = args.get("fsdp", {}).get("wrap_policy", "transformer")
    if wrap_policy == "transformer":
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=layer_classes,
        )
    elif wrap_policy == "root":
        # One FSDP unit trades a larger full-parameter residency window for far
        # fewer collectives. This is useful when layer-wise HYBRID_SHARD is
        # latency/launch-bound and the model still fits in HBM. Keep an explicit
        # (non-wrapping) auto policy so FSDP constructs HYBRID_SHARD's node-local
        # shard and inter-node replica process groups automatically.
        auto_wrap = root_only_auto_wrap_policy
    else:
        raise ValueError(
            f"unknown fsdp.wrap_policy={wrap_policy!r}; expected transformer or root"
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


def validate_fsdp_topology(model, accelerator, args, logger):
    """Verify HYBRID_SHARD means intra-node shards plus inter-node replicas."""
    from torch.distributed import get_world_size
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    world = accelerator.num_processes
    local = int(os.environ.get("LOCAL_WORLD_SIZE", min(world, 8)))
    strategy = args.get("fsdp", {}).get("sharding_strategy", "full_shard")
    roots = [m for m in model.modules() if isinstance(m, FSDP)]
    if not roots:
        if world == 1:
            logger.log_message("[fsdp] single-process probe: model is intentionally unwrapped")
            return
        raise RuntimeError("Accelerate returned no FSDP wrapper")
    root = roots[0]
    shard_size = get_world_size(root.process_group)
    inter_pg = getattr(root, "_inter_node_pg", None)
    replica_size = get_world_size(inter_pg) if inter_pg is not None else 1
    logger.log_message(
        f"[fsdp] topology: strategy={strategy}, world={world}, local={local}, "
        f"shard_group={shard_size}, replica_group={replica_size}"
    )
    if strategy == "hybrid_shard" and world > local:
        expected_replicas = world // local
        if shard_size != local or replica_size != expected_replicas:
            raise RuntimeError(
                "HYBRID_SHARD topology is not node-local: expected "
                f"shard_group={local}, replica_group={expected_replicas}; got "
                f"{shard_size} and {replica_size}"
            )


@hydra.main(config_path="configs", config_name="omni_text", version_base="1.1")
def main(args):
    torch.manual_seed(args.seed)

    resume_from = args.get("checkpoint", {}).get("resume_from", None)
    step_offset = 0
    if resume_from:
        with open(os.path.join(resume_from, "meta.json"), encoding="utf-8") as handle:
            resume_meta = json.load(handle)
            step_offset = int(resume_meta["step"])
        if not args.model.get("init_from", None):
            OmegaConf.update(args, "model.init_from", resume_from, force_add=True)
        if not args.logging.get("wandb_run_id", None):
            run_id = resume_meta.get("wandb_run_id")
            if run_id:
                OmegaConf.update(
                    args, "logging.wandb_run_id", run_id, force_add=True
                )

    if args.data.source == "continuation" and not args.model.get("init_from", None):
        raise SystemExit(
            "legacy/omni_aligned must set model.init_from to ablation 3's 90%-epoch checkpoint"
        )

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
        f"[fsdp] wrap_policy={args.get('fsdp', {}).get('wrap_policy', 'transformer')} "
        f"layer_classes={sorted(c.__name__ for c in layer_classes)}"
    )

    if float(args.optim.get("epochs", 1.0)) != 1.0:
        raise ValueError("the corpus rig currently supports exactly optim.epochs=1")
    dataloaders, capacities = get_dataloader(args, vocab)
    validation_loaders = get_validation_dataloaders(args)
    aligned_frac = (
        args.task.get("aligned_step_frac", None)
        if args.data.source == "continuation" else None
    )
    plan = build_training_plan(
        capacities,
        seed=args.seed,
        aligned_step_frac=aligned_frac,
        max_steps=args.optim.get("max_steps", None),
    )
    OmegaConf.update(
        args, "optim.resolved_total_steps", plan.total_steps, force_add=True
    )
    OmegaConf.update(
        args, "optim.resolved_step_counts", plan.step_counts, force_add=True
    )
    modality_sched = ModalityScheduler(plan.schedule)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.optim.lr,
        betas=tuple(args.optim.betas),
        weight_decay=args.optim.weight_decay,
    )
    # accelerator.prepare() wraps this in an AcceleratedScheduler, which advances the
    # underlying schedule once PER PROCESS on every .step() — the convention being that
    # step() is called per batch. We call it once per *optimizer* step, so on 8 GCDs it
    # consumed 8 scheduler steps each time and lapped the cosine curve repeatedly (the
    # logged lr jumped 1.5e-5 -> 3.8e-7 -> 2.2e-5 -> 2.7e-5 between steps instead of
    # decaying). Scaling the horizon by the world size cancels the multiplier exactly.
    world = max(accelerator.num_processes, 1)
    selected_fraction = float(args.data.get("epoch_end", 1.0)) - float(
        args.data.get("epoch_start", 0.0)
    )
    schedule_total_steps = int(args.optim.get("schedule_total_steps", 0) or 0)
    if not schedule_total_steps:
        if args.data.source in ("discrete_mixed", "discrete_mixed_asr"):
            tokens_per_step = (
                int(args.data.seq_len)
                * int(args.data.micro_batch_size)
                * int(args.optim.grad_acc)
                * world
            )
            schedule_total_steps = int(args.data.token_budget) // tokens_per_step
        else:
            schedule_total_steps = int(round(plan.total_steps / selected_fraction))
    OmegaConf.update(
        args, "optim.resolved_full_total_steps", schedule_total_steps, force_add=True
    )
    warmup_steps = int(args.optim.get("warmup_steps", 0) or 0)
    if not warmup_steps:
        warmup_steps = max(1, int(round(
            schedule_total_steps * float(args.optim.get("warmup_fraction", 0.05))
        )))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps * world,
        num_training_steps=schedule_total_steps * world,
    )

    # Datasets already shard by global rank. Preparing the loaders would cause
    # Accelerate to shard them a second time and silently train on only 1/world data.
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    validate_fsdp_topology(model, accelerator, args, logger)
    if resume_from:
        ckpt_utils.load_training_state(
            resume_from, accelerator, optimizer, lr_scheduler
        )
        logger.log_message(
            f"[resume] model/optimizer/scheduler from {resume_from} at step {step_offset}"
        )

    if args.model.get("compile", False):
        model = torch.compile(model)

    local_batches = {
        key: (
            int(args.data.audio_micro_batch_size)
            if key == "audio"
            else int(args.data.get("aligned_micro_batch_size", 0) or 0)
            if key == "aligned"
            else int(args.data.micro_batch_size)
        )
        for key in dataloaders
    }
    global_batches = {
        key: value * accelerator.num_processes * int(args.optim.grad_acc)
        for key, value in local_batches.items() if key in dataloaders
    }
    logger.log_run_banner(
        args, vocab, num_params, global_batches,
        plan.total_steps, plan.step_counts,
    )

    t0 = time.time()
    before_label = "resume" if step_offset else "base"
    selected_epoch_end = float(args.data.get("epoch_end", 1.0))
    after_label = "final" if selected_epoch_end >= 1.0 else "pilot"
    for stream, validation_loader in validation_loaders.items():
        evaluate_unit_ntp(
            model, validation_loader, accelerator, args, logger,
            label=f"{before_label}/{stream}", step=step_offset,
        )
    train(model, dataloaders, accelerator, optimizer, lr_scheduler, args=args,
          logger=logger, total_steps=plan.total_steps, scheduler=modality_sched,
          step_offset=step_offset)
    for stream, validation_loader in validation_loaders.items():
        evaluate_unit_ntp(
            model, validation_loader, accelerator, args, logger,
            label=f"{after_label}/{stream}", step=step_offset + plan.total_steps,
        )
    accelerator.wait_for_everyone()
    logger.log_message(f"TOTAL train time: {time.time() - t0:.1f}s "
                       f"for {plan.total_steps} steps")
    logger.finish()


if __name__ == "__main__":
    main()
