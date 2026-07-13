#!/usr/bin/env python3
"""Before/after qualitative check: baseline Thinker vs a CPT checkpoint.

Runs greedy next-token generation from the same fixed Irish + English prompts on
(a) the stock Qwen2.5-Omni-3B Thinker and (b) a checkpoint written by the training
run, and writes them side by side. This is the qualitative half of the ablation
evidence; the quantitative half is the NTP loss curve in W&B.

Single GPU, no FSDP — the checkpoint is a full (unsharded) state dict.

  python -m eval.generate_compare \
      --checkpoint  .../output/<run>/checkpoints/step_000400 \
      --baseline    Qwen/Qwen2.5-Omni-3B \
      --prompts     eval/prompts.txt \
      --out         .../output/<run>/before_after

Note it loads the two models one at a time (a 3B pair does not need to co-reside),
so the run costs two loads but half the memory.
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch


def read_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.lstrip().startswith("#")]


@torch.no_grad()
def generate(thinker, tokenizer, prompts, device, max_new_tokens, temperature):
    outs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen = thinker.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature or None,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Strip the prompt: we want to read the continuation on its own.
        outs.append(tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))
    return outs


def load_baseline(model_id, device, dtype):
    from transformers import Qwen2_5OmniForConditionalGeneration
    full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa"
    )
    return full.thinker.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="step_NNNNNN dir from a run")
    ap.add_argument("--baseline", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__),
                                                      "prompts.txt"))
    ap.add_argument("--out", required=True, help="output prefix (.jsonl + .txt)")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (deterministic, so before/after is comparable)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    prompts = read_prompts(args.prompts)
    print(f"{len(prompts)} prompts | device={device}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.baseline)

    print("[before] loading baseline thinker...", flush=True)
    thinker = load_baseline(args.baseline, device, dtype)
    before = generate(thinker, tokenizer, prompts, device,
                      args.max_new_tokens, args.temperature)
    del thinker
    torch.cuda.empty_cache()

    print(f"[after] loading checkpoint {args.checkpoint} ...", flush=True)
    # Import late: pulls in the training package, which we don't need for the baseline.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from qomhra.checkpoint import load_for_eval
    model, _ = load_for_eval(args.checkpoint, device=device, dtype=dtype)
    after = generate(model.thinker, tokenizer, prompts, device,
                     args.max_new_tokens, args.temperature)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".jsonl", "w", encoding="utf-8") as f:
        for p, b, a in zip(prompts, before, after):
            f.write(json.dumps({"prompt": p, "before": b, "after": a},
                               ensure_ascii=False) + "\n")
    with open(args.out + ".txt", "w", encoding="utf-8") as f:
        f.write(f"baseline   : {args.baseline}\n"
                f"checkpoint : {args.checkpoint}\n"
                f"decoding   : {'greedy' if args.temperature == 0 else f'T={args.temperature}'}"
                f", max_new_tokens={args.max_new_tokens}\n")
        for p, b, a in zip(prompts, before, after):
            f.write("\n" + "=" * 78 + f"\nPROMPT: {p}\n"
                    + "-" * 78 + f"\nBEFORE: {b.strip()}\n"
                    + "-" * 78 + f"\nAFTER : {a.strip()}\n")
    print(f"wrote {args.out}.jsonl and {args.out}.txt", flush=True)


if __name__ == "__main__":
    main()
