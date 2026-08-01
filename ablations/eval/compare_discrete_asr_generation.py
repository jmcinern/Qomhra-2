#!/usr/bin/env python3
"""Raw few-shot ASR comparison of the expanded base and trained model.

The same demonstrations and FLEURS queries are used for both states. Outputs
are scored exactly as decoded: no cleanup, truncation, or language filtering.
"""

import argparse
import copy
import gc
import json
import os
import sys

import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, "..", "train"))
sys.path.insert(0, TRAIN)

from audit_discrete_asr_prompt import (  # noqa: E402
    labelled_example,
    load_rows,
    generate,
    snapshot,
    validate_prompt_contract,
)
from asr_baseline import norm, wer  # noqa: F401,E402
from fleurs_eval import cer  # noqa: E402
from qomhra.checkpoint import load_for_eval  # noqa: E402
from qomhra.model import get_model  # noqa: E402


def initial_model(config_checkpoint):
    cfg = OmegaConf.load(os.path.join(config_checkpoint, "config.yaml"))
    # This reconstructs the expanded, untrained base rather than following an
    # init_from pointer embedded in an extension checkpoint.
    cfg.model.init_from = None
    torch.manual_seed(int(cfg.seed))
    model, _ = get_model(cfg)
    return model.to(device="cuda", dtype=torch.bfloat16).eval(), cfg


def score_state(model, tok, eos, demo_ids, rows, max_new, state, prompt_style):
    thinker = model.thinker
    thinker.config.use_cache = True
    outputs = []
    word_edits = word_count = char_edits = char_count = 0
    for index, row in enumerate(rows):
        query = labelled_example(row["input_ids"], tok, prompt_style)
        prompt_ids = demo_ids + query
        result = generate(thinker, tok, prompt_ids, eos, max_new)
        we, wc = wer(row["reference"], result["hypothesis"])
        ce, cc = cer(row["reference"], result["hypothesis"])
        result.update(
            id=row["id"],
            reference=row["reference"],
            prompt_tokens=len(prompt_ids),
            word_edits=we,
            reference_words=wc,
            char_edits=ce,
            reference_chars=cc,
        )
        outputs.append(result)
        word_edits += we
        word_count += wc
        char_edits += ce
        char_count += cc
        print(
            f"[{state} {index + 1}/{len(rows)}] id={row['id']} "
            f"prompt={len(prompt_ids)} stop={result['stop_reason']} "
            f"WER-edits={we}/{wc}\n"
            f"  REF: {row['reference']}\n"
            f"  HYP: {result['hypothesis']}",
            flush=True,
        )
    return {
        "wer": 100 * word_edits / max(word_count, 1),
        "cer": 100 * char_edits / max(char_count, 1),
        "eos_rate": sum(row["eos_emitted"] for row in outputs) / len(outputs),
        "max_token_cap_rate": sum(
            row["stop_reason"] == "max_new_tokens" for row in outputs
        ) / len(outputs),
        "mean_prompt_tokens": sum(row["prompt_tokens"] for row in outputs)
        / len(outputs),
        "rows": outputs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config-checkpoint", required=True)
    ap.add_argument("--trained-checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--shots", type=int, default=9)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--prompt-style", choices=["native", "qa", "language"], default="qa"
    )
    args = ap.parse_args()

    demos = load_rows(args.demos)[: args.shots]
    rows = load_rows(args.data)[: args.n]
    if len(demos) != args.shots:
        raise RuntimeError(f"requested {args.shots} demos, found {len(demos)}")

    base, base_cfg = initial_model(args.base_config_checkpoint)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot(base_cfg.model.base_model_id))
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    contract = validate_prompt_contract(demos, rows, eos)
    demo_ids = [
        token
        for demo in demos
        for token in labelled_example(demo["input_ids"], tok, args.prompt_style)
    ]
    contract.update(
        labelled_demo_tokens=len(demo_ids),
        prompt_style=args.prompt_style,
        scoring="raw decode; no hypothesis post-processing",
    )
    print(json.dumps(contract, indent=2), flush=True)

    results = {
        "base": score_state(
            base,
            tok,
            eos,
            demo_ids,
            rows,
            args.max_new_tokens,
            "base",
            args.prompt_style,
        )
    }
    del base
    gc.collect()
    torch.cuda.empty_cache()

    trained, _ = load_for_eval(
        args.trained_checkpoint, device="cuda", dtype=torch.bfloat16
    )
    results["trained_100pct"] = score_state(
        trained,
        tok,
        eos,
        demo_ids,
        rows,
        args.max_new_tokens,
        "trained_100pct",
        args.prompt_style,
    )

    report = {
        "base_definition": (
            "same expanded architecture reconstructed from the original run "
            "config and seed, with no ASR checkpoint loaded"
        ),
        "base_config_checkpoint": args.base_config_checkpoint,
        "trained_checkpoint": args.trained_checkpoint,
        "data": args.data,
        "n": len(rows),
        "shots": len(demos),
        "prompt_style": args.prompt_style,
        "prompt_contract": contract,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    for name, result in results.items():
        print(
            f"SUMMARY {name}: WER={result['wer']:.2f} "
            f"CER={result['cer']:.2f} EOS={result['eos_rate']:.0%} "
            f"cap={result['max_token_cap_rate']:.0%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
