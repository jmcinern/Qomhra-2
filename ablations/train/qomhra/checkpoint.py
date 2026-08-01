"""FSDP checkpoint save/load.

The rig previously saved nothing (it was a timing harness). The before/after
qualitative eval needs weights, so we gather a full (unsharded) state dict on rank 0
and write it out. `offload_to_cpu=True` is not optional: a 3B fp32 gather onto one
GCD's HBM alongside the live shards would OOM.
"""
import json
import os
import shutil

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from omegaconf import OmegaConf


def save(model, accelerator, args, step, ckpt_dir, optimizer=None, lr_scheduler=None):
    """Gather FULL_STATE_DICT on rank 0 and write {ckpt_dir}/step_{step}/.

    Every rank must enter the gather (it is a collective), but only rank 0 writes.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    out = os.path.join(ckpt_dir, f"step_{step:06d}")

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()

    if accelerator.is_main_process:
        os.makedirs(out, exist_ok=True)
        torch.save(state, os.path.join(out, "pytorch_model.bin"))
        OmegaConf.save(args, os.path.join(out, "config.yaml"))
        with open(os.path.join(out, "meta.json"), "w") as f:
            total = int(args.optim.get("resolved_full_total_steps", step))
            json.dump({
                "step": step,
                "total_steps": total,
                "epoch_progress": step / total,
                "base_model_id": args.model.base_model_id,
                "wandb_run_id": args.logging.get("wandb_run_id", None),
            }, f)
    del state
    accelerator.wait_for_everyone()
    if bool(args.get("checkpoint", {}).get("save_training_state", False)):
        if optimizer is None or lr_scheduler is None:
            raise RuntimeError("save_training_state requires optimizer and lr_scheduler")
        rank = int(accelerator.process_index)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            os.path.join(out, f"training_state_rank_{rank:05d}.pt"),
        )
        accelerator.wait_for_everyone()
    return out


def load_training_state(path, accelerator, optimizer, lr_scheduler):
    """Restore the same-world-size sharded optimizer/scheduler state."""
    rank = int(accelerator.process_index)
    state_path = os.path.join(path, f"training_state_rank_{rank:05d}.pt")
    if not os.path.isfile(state_path):
        raise RuntimeError(f"resume checkpoint lacks {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    accelerator.wait_for_everyone()


def prune(ckpt_dir, keep_last):
    """Keep only the newest `keep_last` step_* dirs (rank 0 only — call guarded)."""
    if not keep_last or not os.path.isdir(ckpt_dir):
        return
    steps = sorted(d for d in os.listdir(ckpt_dir) if d.startswith("step_"))
    for stale in steps[:-int(keep_last)]:
        shutil.rmtree(os.path.join(ckpt_dir, stale), ignore_errors=True)


def load_for_eval(ckpt_path, device="cuda", dtype=torch.bfloat16):
    """Rebuild OmniThinkerCPT on a single device (no FSDP) from a saved checkpoint.

    Used by eval/generate_compare.py. Returns (model, args).
    """
    from .model import get_model

    args = OmegaConf.load(os.path.join(ckpt_path, "config.yaml"))
    model, _ = get_model(args)
    state = torch.load(os.path.join(ckpt_path, "pytorch_model.bin"),
                       map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARN checkpoint missing {len(missing)} keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"WARN checkpoint unexpected {len(unexpected)} keys, e.g. {unexpected[:5]}")
    return model.to(device=device, dtype=dtype).eval(), args
