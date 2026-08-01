"""W&B + console logging — trimmed from nanoT5/utils/logging_utils.py.

Only the main process opens a wandb run (under 8-way FSDP every rank calling
wandb.init would otherwise produce 1 real + 7 phantom runs).
"""
import logging
import os

from accelerate.logging import get_logger
from omegaconf import OmegaConf
import wandb


class Logger:
    def __init__(self, args, accelerator):
        self.logger = get_logger("Main")
        self.accelerator = accelerator

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f"Working directory is {os.getcwd()}")

        self.setup_wandb(args)

    def setup_wandb(self, args):
        if args.logging.get("wandb", False) and self.accelerator.is_main_process:
            self.wandb_run = wandb.init(
                project=args.logging.get("wandb_project", "qomhra-train-est"),
                name=args.logging.get("wandb_run_name", None) or None,
                id=args.logging.get("wandb_run_id", None) or None,
                resume=args.logging.get("wandb_resume", "allow"),
                config=OmegaConf.to_container(args, resolve=True),
            )
        else:
            self.wandb_run = None

    def log_run_banner(self, args, vocab, num_params, global_batches,
                       total_steps, step_counts):
        wandb_url = self.wandb_run.url if self.wandb_run is not None else "(wandb off)"
        msg = [
            "",
            "================ RUN CONFIG ================",
            f"model         : {args.model.get('base_model_id', '?')} (pretrained)",
            f"ablation      : {args.task.modality}   (freeze: {args.freeze.mode})",
            f"params        : {num_params/1e9:.2f} B",
            f"vocab         : {vocab:,}",
            f"seq len       : {args.data.seq_len}",
            f"micro batch   : {args.data.micro_batch_size} / GCD   grad_acc {args.optim.grad_acc}",
            f"global batches: {global_batches}",
            f"epoch plan    : {step_counts} -> {total_steps:,} optimizer steps",
            f"wandb         : {wandb_url}",
            "============================================",
            "",
        ]
        self.log_message("\n".join(msg))

    def log_stats(self, stats, step, prefix=""):
        # A mixed run only reports the modality it ran this window, so keys come and
        # go between windows — W&B handles the gaps as separate series.
        stats = {k: v for k, v in stats.items() if v is not None}
        if self.wandb_run is not None:
            self.wandb_run.log({f"{prefix}{k}": v for k, v in stats.items()}, step=step)
        dict_msg = " | ".join(f"{k} --> {v:.4g}" for k, v in stats.items())
        self.log_message(f"[{prefix[:-1] or 'train'}] step {step} | {dict_msg}")

    def log_message(self, msg):
        self.logger.info(msg)

    def finish(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
