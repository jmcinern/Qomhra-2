#!/usr/bin/env python3
"""Why does the chat column keep generating after it has answered?

Baseline probe showed 'What is the capital city of Ireland?' -> 'Dublin.\\nHuman:
What is the capital city of Ireland?\\nDublin\\nHuman: ...'. A correct answer followed
by a base-model-style continuation loop is the signature of generation not STOPPING:
the model emits <|im_end|>, nothing treats it as EOS, decoding rolls on, and
skip_special_tokens=True hides the marker so the log looks like the model rambled.

Prime suspect: we pass pad_token_id=tok.eos_token_id and let eos default. If the
thinker's eos is <|endoftext|> rather than <|im_end|>, the turn never terminates.

This prints the rendered template, the relevant special-token ids, and the raw
undecoded output so we can see the markers instead of inferring them.
"""
import argparse
import glob
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    if args.model is None:
        args.model = glob.glob("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                               "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import (AutoConfig, Qwen2_5OmniProcessor,
                              Qwen2_5OmniThinkerForConditionalGeneration)

    cfg = AutoConfig.from_pretrained(args.model)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, config=cfg.thinker_config, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(args.model)
    tok = proc.tokenizer

    print("=" * 72)
    print("TOKEN IDS")
    print("=" * 72)
    for name in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        print(f"  {name:16s} -> {tok.convert_tokens_to_ids(name)}")
    print(f"  tokenizer.eos_token   = {tok.eos_token!r} (id={tok.eos_token_id})")
    print(f"  tokenizer.pad_token   = {tok.pad_token!r} (id={tok.pad_token_id})")
    gc = thinker.generation_config
    print(f"  generation_config.eos_token_id = {gc.eos_token_id}")
    print(f"  generation_config.pad_token_id = {gc.pad_token_id}")

    conv = [{"role": "user", "content": [{"type": "text",
             "text": "What is the capital city of Ireland?"}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    print("\n" + "=" * 72)
    print("RENDERED TEMPLATE (repr, so markers/newlines are visible)")
    print("=" * 72)
    print(repr(text))

    ids = tok(text, return_tensors="pt").input_ids.to(device)
    print(f"\n  prompt tokens = {ids.shape[1]}")

    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    # A: exactly what baseline_probe.py did.
    with torch.no_grad():
        out_a = thinker.generate(input_ids=ids, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    new_a = out_a[0][ids.shape[1]:]

    # B: same, but stop on <|im_end|> explicitly.
    with torch.no_grad():
        out_b = thinker.generate(input_ids=ids, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok.eos_token_id, eos_token_id=im_end)
    new_b = out_b[0][ids.shape[1]:]

    print("\n" + "=" * 72)
    print("A) eos defaulted (what the probe ran)")
    print("=" * 72)
    print(f"  new tokens      : {new_a.shape[0]}")
    print(f"  contains im_end : {bool((new_a == im_end).any())}")
    print(f"  raw             : {tok.decode(new_a, skip_special_tokens=False)!r}")

    print("\n" + "=" * 72)
    print("B) eos_token_id=<|im_end|>")
    print("=" * 72)
    print(f"  new tokens      : {new_b.shape[0]}")
    print(f"  raw             : {tok.decode(new_b, skip_special_tokens=False)!r}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
