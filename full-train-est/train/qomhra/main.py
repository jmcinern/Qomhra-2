"""Entry point — 8-GCD FSDP timing run for the Qomhra full-train estimate.

Launched one-process-per-GCD by srun (see ../train.sh), which sets RANK /
LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT. Accelerate reads those for
torch.distributed env:// init; we build the FSDP plugin in code so no
`accelerate launch` / accelerate config file is needed.
"""
import functools
import time

import hydra
import torch
from accelerate import Accelerator, DataLoaderConfiguration, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import get_cosine_schedule_with_warmup
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from .data import get_dataloader
from .logging_utils import Logger
from .model import get_model
from .train_utils import train


def build_fsdp_plugin():
    # Full-shard (ZeRO-3 equivalent): shard params, grads and optimizer states
    # across the 8 GCDs so the 8B model + Adam states fit in 64 GB/GCD.
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3DecoderLayer},
    )
    return FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,   # fp32 grad reduction for stability
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,
    )


@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(args):
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.precision,
        fsdp_plugin=build_fsdp_plugin(),
        # dispatch_batches=False: each rank iterates its own SyntheticTokenDataset
        # (seeded by RANK) instead of rank 0 loading and scattering — correct for
        # data-parallel timing and avoids a needless scatter on an IterableDataset.
        dataloader_config=DataLoaderConfiguration(
            non_blocking=True, dispatch_batches=False
        ),
    )
    logger = Logger(args, accelerator)

    model, vocab = get_model(args)
    num_params = sum(p.numel() for p in model.parameters())
    dataloader = get_dataloader(args, vocab)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=tuple(args.optim.betas),
        weight_decay=args.optim.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.optim.warmup_steps,
        num_training_steps=args.optim.total_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    global_batch = (
        args.data.micro_batch_size * accelerator.num_processes * args.optim.grad_acc
    )
    logger.log_run_banner(args, vocab, num_params, global_batch)

    t0 = time.time()
    train(model, dataloader, accelerator, optimizer, lr_scheduler, args=args,
          logger=logger, global_batch=global_batch)
    accelerator.wait_for_everyone()
    logger.log_message(f"TOTAL train time: {time.time() - t0:.1f}s "
                       f"for {args.optim.total_steps} steps")
    logger.finish()


if __name__ == "__main__":
    main()
