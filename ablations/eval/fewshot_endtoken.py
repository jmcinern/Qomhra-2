#!/usr/bin/env python3
"""Can the model be *shown* where an Irish turn ends, without instruction tuning?

Measured: ga:chat termination is 0/4 both before and after CPT — the model never emits
<|im_end|> in Irish while doing it reliably in English. Raw-text CPT contains no
<|im_end|> at all, so nothing in the training signal teaches where an Irish turn stops.

The question this answers: is that a missing *capability* or a missing *cue*? If a few
in-context examples of short Irish Q->A turns make it terminate, the ability is latent
and the ablations just never elicited it — which would mean the 0/4 says more about the
prompt than about the model, and the paper should report it that way.

Conditions, per language:
  zero  - the prompt as-is (what generate_compare measures)
  few   - the same prompt preceded by 3 complete Q->A turns, each properly terminated

English is the CONTROL: it already terminates at zero-shot, so if few-shot changes
English too, the effect is the harness, not Irish.

  python -m eval.fewshot_endtoken --checkpoints <ckpt> --labels text
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SNAPSHOT_GLOB = ("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")

# Short factual Q->A, so a correct answer is inherently brief and the model is never
# forced to stop mid-thought by the token budget (which would confound termination).
SHOTS = {
    "ga": [
        ("Cad é príomhchathair na Fraince?", "Is é Páras príomhchathair na Fraince."),
        ("Cén dath atá ar an spéir?", "Tá an spéir gorm."),
        ("Cé mhéad lá atá i seachtain?", "Tá seacht lá i seachtain."),
    ],
    "en": [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What colour is the sky?", "The sky is blue."),
        ("How many days are in a week?", "There are seven days in a week."),
    ],
}
TESTS = {
    "ga": ["Cad é príomhchathair na hÉireann?",
           "Cén dath atá ar an bhféar?",
           "Cé mhéad mí atá i mbliain?",
           "Cad is ainm don teanga a labhraítear in Éirinn?"],
    "en": ["What is the capital city of Ireland?",
           "What colour is grass?",
           "How many months are in a year?",
           "What language is spoken in Ireland?"],
}


def build(proc, lang, question, n_shots):
    conv = []
    for q, a in SHOTS[lang][:n_shots]:
        conv.append({"role": "user", "content": [{"type": "text", "text": q}]})
        conv.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    conv.append({"role": "user", "content": [{"type": "text", "text": question}]})
    return proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def run(thinker, proc, device, max_new_tokens=60):
    tok = proc.tokenizer
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    out = {}
    for lang in ("ga", "en"):
        for n_shots, cond in ((0, "zero"), (3, "few")):
            rows = []
            for q in TESTS[lang]:
                text = build(proc, lang, q, n_shots)
                enc = tok(text, return_tensors="pt").to(device)
                gen = thinker.generate(**enc, max_new_tokens=max_new_tokens,
                                       do_sample=False, eos_token_id=eos_id,
                                       pad_token_id=tok.pad_token_id)
                new = gen[0][enc.input_ids.shape[1]:]
                rows.append({
                    "q": q,
                    "text": tok.decode(new, skip_special_tokens=True),
                    "terminated": bool((new == eos_id).any()),
                    "n_tokens": int(new.shape[0]),
                })
            out[(lang, cond)] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()
    if args.baseline is None:
        args.baseline = glob.glob(SNAPSHOT_GLOB)[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(args.baseline)

    from generate_compare import load_baseline
    results = {}
    print("[baseline] loading...", flush=True)
    thinker = load_baseline(args.baseline, device, dtype)  # text only -> sdpa is safe
    results["baseline"] = run(thinker, proc, device)
    del thinker
    torch.cuda.empty_cache()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from qomhra.checkpoint import load_for_eval
    for i, ckpt in enumerate(args.checkpoints):
        label = args.labels[i] if i < len(args.labels) else os.path.basename(ckpt)
        print(f"[{label}] loading...", flush=True)
        model, _ = load_for_eval(ckpt, device=device, dtype=dtype)
        results[label] = run(model.thinker, proc, device)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("TERMINATION RATE  (emitted <|im_end|> within budget)")
    print("=" * 72)
    print(f"{'model':12s} {'ga zero':>9s} {'ga few':>8s} {'en zero':>9s} {'en few':>8s}")
    for label, r in results.items():
        def rate(lang, cond):
            rows = r[(lang, cond)]
            return f"{sum(x['terminated'] for x in rows)}/{len(rows)}"
        print(f"{label:12s} {rate('ga','zero'):>9s} {rate('ga','few'):>8s} "
              f"{rate('en','zero'):>9s} {rate('en','few'):>8s}")

    print("\n" + "=" * 72)
    print("IRISH SAMPLES (zero vs few)")
    print("=" * 72)
    for label, r in results.items():
        print(f"\n--- {label}")
        for cond in ("zero", "few"):
            row = r[("ga", cond)][0]
            print(f"  [{cond:4s}] term={row['terminated']!s:5s} {row['n_tokens']:3d}tok "
                  f"{row['text'][:110]!r}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
