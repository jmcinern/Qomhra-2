#!/usr/bin/env python3
"""Sanity-check the UNTRAINED Qwen2.5-Omni thinker on Irish + English text.

Before building a before/after eval we need to know the "before" is worth measuring:
  - Is baseline Irish fluent, broken, or English-with-an-accent?
  - Does the raw-continuation format even work on an instruct-tuned checkpoint,
    or does it emit chat markers / immediately EOS?
  - Does the chat format answer in Irish when asked in Irish?

Prints raw and chat renderings side by side for the same prompt so the format
axis is directly comparable. Greedy, so reruns are identical.
"""
import argparse
import glob
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

# Continuations (base-model style) and instructions (chat style). Deliberately the
# same topics across the two languages so the language axis is not confounded.
RAW = [
    ("ga", "Is é an fáth a bhfuil an Ghaeilge tábhachtach ná"),
    ("ga", "Bhí an aimsir go hálainn inné, mar sin chuaigh mé"),
    ("ga", "Nuair a bhí mé óg, chaith mé an samhradh i gConamara ag"),
    ("en", "The Irish language is spoken today mainly in"),
]
CHAT = [
    ("ga", "Cad é príomhchathair na hÉireann?"),
    ("ga", "Inis dom faoi stair na Gaeilge in Éirinn."),
    ("en", "What is the capital city of Ireland?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    if args.model is None:
        snaps = glob.glob("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                          "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")
        if not snaps:
            raise SystemExit("no local Qwen2.5-Omni-3B snapshot")
        args.model = snaps[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import (AutoConfig, Qwen2_5OmniProcessor,
                              Qwen2_5OmniThinkerForConditionalGeneration)

    print(f"[load] {args.model}", flush=True)
    cfg = AutoConfig.from_pretrained(args.model)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, config=cfg.thinker_config, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    print("[load] OK", flush=True)

    # generation_config.eos_token_id is None on this checkpoint, so generate() has no
    # stop token and runs to max_new_tokens, decoding straight past <|im_end|> into a
    # self-primed continuation loop. tok.eos_token_id is correct; generate() just never
    # consults it. Pass it explicitly or every chat generation is garbage after the turn.
    eos_ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]

    def gen(text):
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = thinker.generate(**enc, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, eos_token_id=eos_ids,
                                   pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

    print("\n" + "=" * 72)
    print("RAW CONTINUATION (no chat template)")
    print("=" * 72)
    for lang, p in RAW:
        print(f"\n[{lang}] {p!r}\n  -> {gen(p)!r}", flush=True)

    print("\n" + "=" * 72)
    print("CHAT TEMPLATE")
    print("=" * 72)
    for lang, p in CHAT:
        conv = [{"role": "user", "content": [{"type": "text", "text": p}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        print(f"\n[{lang}] {p!r}\n  -> {gen(text)!r}", flush=True)
    print("\n" + "=" * 72, flush=True)


if __name__ == "__main__":
    main()
