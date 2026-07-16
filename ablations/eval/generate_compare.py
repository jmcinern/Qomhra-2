#!/usr/bin/env python3
"""Before/after: baseline Thinker vs a CPT checkpoint, over language x prompt-format.

Greedy generation from the same fixed prompts on (a) the stock Qwen2.5-Omni-3B Thinker
and (b) a checkpoint from a training run, written side by side. Qualitative half of the
ablation evidence; the quantitative half is the NTP loss curve in W&B.

Two axes (see prompts.txt):
  language ga|en   - English is the CONTROL, not a comparison. Same model, prompts and
                     decoding, so an Irish-only failure is Irish-specific, not the rig.
  format raw|chat  - Qwen2.5-Omni ships instruct-tuned and we continued-pretrain on RAW
                     text, so the chat rows measure whether CPT ate instruction-following.

Also reports two metrics that need no reference text, both at the floor for baseline
Irish and at ceiling for baseline English, so movement is unambiguous:
  terminated - did it emit <|im_end|> within budget? (baseline: ga 0/2, en 1/1)
  degenerate - is there a repeated n-gram tail? (baseline: ga 3/3 raw, en 0/1)

Single GPU, no FSDP — the checkpoint is a full (unsharded) state dict. The two models are
loaded one at a time: two loads, half the memory.

  python -m eval.generate_compare \
      --checkpoint  .../output/<run>/checkpoints/step_000400 \
      --out         .../output/<run>/before_after
"""
import argparse
import json
import os
import re
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

SNAPSHOT_GLOB = ("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")
LINE = re.compile(r"^(ga|en):(raw|chat):\s*(.+)$")


def read_prompts(path):
    """-> [(lang, fmt, text)]. Tags are mandatory: an untagged prompt is a silent
    mis-render (raw text through the chat template, or vice versa), which would look
    like a model result rather than a harness bug."""
    out = []
    with open(path, encoding="utf-8") as f:
        for n, ln in enumerate(f, 1):
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            m = LINE.match(s)
            if not m:
                raise SystemExit(f"{path}:{n}: expected '<ga|en>:<raw|chat>: text', got {s!r}")
            out.append((m.group(1), m.group(2), m.group(3).strip()))
    return out


def is_degenerate(text, n=4, times=3):
    """True if some n-gram repeats >= `times`. Baseline Irish loops like 'ina bheith
    agus ina bheith agus ina bheith'; English does not. Cheap, reference-free."""
    w = text.split()
    if len(w) < n * times:
        return False
    grams = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(grams.values()) >= times


@torch.no_grad()
def generate(thinker, proc, prompts, device, max_new_tokens, temperature):
    tok = proc.tokenizer
    # generation_config.eos_token_id is None on this checkpoint, so generate() has NO
    # stop token: it runs to max_new_tokens, decoding past <|im_end|> into a self-primed
    # continuation loop, and skip_special_tokens hides the marker so it reads as the
    # model rambling. tok.eos_token_id is correct (151645); generate never consults it.
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for lang, fmt, prompt in prompts:
        if fmt == "chat":
            conv = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(conv, add_generation_prompt=True,
                                            tokenize=False)
        else:
            text = prompt
        enc = tok(text, return_tensors="pt").to(device)
        gen = thinker.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature or None,
            eos_token_id=eos_id,
            pad_token_id=tok.pad_token_id,
        )
        new = gen[0][enc.input_ids.shape[1]:]
        body = tok.decode(new, skip_special_tokens=True)
        rows.append({
            "text": body,
            "n_tokens": int(new.shape[0]),
            "terminated": bool((new == eos_id).any()),
            "degenerate": is_degenerate(body),
        })
    return rows


def load_baseline(model_id, device, dtype):
    """Stock Thinker, loaded directly rather than via Qwen2_5OmniForConditionalGeneration
    (which also builds the Talker and torch.loads the TTS speaker dict — refused on the
    container's torch 2.5). See qomhra/model.py::_load_thinker."""
    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )
    cfg = AutoConfig.from_pretrained(model_id)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id, config=cfg.thinker_config, torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    return thinker.to(device).eval()


def summarise(f, prompts, before, after):
    """Per language x format: termination and degeneracy, before vs after."""
    cells = {}
    for (lang, fmt, _), b, a in zip(prompts, before, after):
        c = cells.setdefault((lang, fmt), {"n": 0, "tb": 0, "ta": 0, "db": 0, "da": 0})
        c["n"] += 1
        c["tb"] += b["terminated"]
        c["ta"] += a["terminated"]
        c["db"] += b["degenerate"]
        c["da"] += a["degenerate"]
    f.write("\n" + "=" * 78 + "\nSUMMARY  (terminated = emitted <|im_end|>; "
            "degenerate = repeated 4-gram)\n" + "=" * 78 + "\n")
    f.write(f"{'cell':10s} {'n':>3s} {'term before':>12s} {'term after':>11s} "
            f"{'degen before':>13s} {'degen after':>12s}\n")
    for (lang, fmt), c in sorted(cells.items()):
        f.write(f"{lang + ':' + fmt:10s} {c['n']:3d} {c['tb']:>8d}/{c['n']:<3d} "
                f"{c['ta']:>7d}/{c['n']:<3d} {c['db']:>9d}/{c['n']:<3d} "
                f"{c['da']:>8d}/{c['n']:<3d}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="step_NNNNNN dir from a run")
    ap.add_argument("--baseline", default=None,
                    help="default: the local Omni snapshot (a repo id sends the "
                         "tokenizer's is_base_mistral check to the unreachable HF API)")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__),
                                                      "prompts.txt"))
    ap.add_argument("--out", required=True, help="output prefix (.jsonl + .txt)")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (deterministic, so before/after is comparable)")
    args = ap.parse_args()

    if args.baseline is None:
        import glob
        snaps = glob.glob(SNAPSHOT_GLOB)
        if not snaps:
            raise SystemExit("no local Qwen2.5-Omni-3B snapshot; pass --baseline")
        args.baseline = snaps[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    prompts = read_prompts(args.prompts)
    print(f"{len(prompts)} prompts | device={device}", flush=True)

    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(args.baseline)

    print("[before] loading baseline thinker...", flush=True)
    thinker = load_baseline(args.baseline, device, dtype)
    before = generate(thinker, proc, prompts, device,
                      args.max_new_tokens, args.temperature)
    del thinker
    torch.cuda.empty_cache()

    print(f"[after] loading checkpoint {args.checkpoint} ...", flush=True)
    # Import late: pulls in the training package, unneeded for the baseline.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from qomhra.checkpoint import load_for_eval
    model, _ = load_for_eval(args.checkpoint, device=device, dtype=dtype)
    after = generate(model.thinker, proc, prompts, device,
                     args.max_new_tokens, args.temperature)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".jsonl", "w", encoding="utf-8") as f:
        for (lang, fmt, p), b, a in zip(prompts, before, after):
            f.write(json.dumps({"lang": lang, "fmt": fmt, "prompt": p,
                                "before": b, "after": a}, ensure_ascii=False) + "\n")

    with open(args.out + ".txt", "w", encoding="utf-8") as f:
        f.write(f"baseline   : {args.baseline}\n"
                f"checkpoint : {args.checkpoint}\n"
                f"decoding   : {'greedy' if args.temperature == 0 else f'T={args.temperature}'}"
                f", max_new_tokens={args.max_new_tokens}\n")
        # Grouped by cell so a whole column can be read at once.
        for cell in sorted({(l, fm) for l, fm, _ in prompts}):
            f.write("\n\n" + "#" * 78 + f"\n### {cell[0]} / {cell[1]}\n" + "#" * 78 + "\n")
            for (lang, fmt, p), b, a in zip(prompts, before, after):
                if (lang, fmt) != cell:
                    continue
                def tag(r):
                    return (f"[{r['n_tokens']:3d} tok"
                            f"{'' if r['terminated'] else ', NO-STOP'}"
                            f"{', DEGEN' if r['degenerate'] else ''}]")
                f.write("\n" + "=" * 78 + f"\nPROMPT: {p}\n"
                        + "-" * 78 + f"\nBEFORE {tag(b)}: {b['text'].strip()}\n"
                        + "-" * 78 + f"\nAFTER  {tag(a)}: {a['text'].strip()}\n")
        summarise(f, prompts, before, after)
    print(f"wrote {args.out}.jsonl and {args.out}.txt", flush=True)


if __name__ == "__main__":
    main()
