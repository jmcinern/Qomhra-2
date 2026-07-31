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


def save(model, accelerator, args, step, ckpt_dir):
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
            json.dump({"step": step, "base_model_id": args.model.base_model_id}, f)
    del state
    accelerator.wait_for_everyone()
    return out


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
