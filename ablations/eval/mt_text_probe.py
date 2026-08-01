#!/usr/bin/env python3
"""Text-in translation probe — does the model translate at all, zero-shot?

IWSLT asks two things at once: can the model read speech units, and can it turn Irish
into English. When it returns Irish, those two failures are indistinguishable. This
probe removes the speech half. Irish text goes in, English is asked for, and whatever
comes out is scored against the parallel reference.

MEXA and FLEURS show the two languages already aligned in representation space without
any translation data in the mix, so the capability may be there even though the objective
was never trained. That is what this measures.

Zero-shot throughout: no demonstrations, and the raw decode is scored. A task cue is not
a demonstration -- the model is told what to do, never shown. Which wording works is the
whole question here, so several are run and reported side by side.

  python -m eval.mt_text_probe --data data/fleurs_parallel_test.parquet \
      --checkpoint <ckpt> --label text_ablation --n 50 --eos-token '<|endoftext|>'
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sacrebleu  # noqa: E402
import torch  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "train")))

from discrete_eval_common import (  # noqa: E402
    base_snapshot,
    decode,
    print_smoke,
    smoke_gate,
)

BLEU_SCORER = BLEU(tokenize="13a")
CHRFPP_SCORER = CHRF(char_order=6, word_order=2, beta=2)

# Each cue is (source label, target label, leading instruction). The bare cue is the
# minimal one already used by iwslt_st_units.py --task-cue english; the labelled and
# instructed variants add progressively more, and all three stay zero-shot.
CUES = {
    "bare": ("", "{tgt}:", ""),
    "labelled": ("{src}: ", "{tgt}:", ""),
    "instructed": ("{src}: ", "{tgt}:", "Aistrigh an abairt seo a leanas.\n"),
}
LANG_NAMES = {"ga": ("Gaeilge", "Irish"), "en": ("Béarla", "English")}


def build_prompt(tokenizer, source_text, cue, direction):
    """Zero-shot prompt: optional instruction, the source, then the target-language cue."""
    src_lang, tgt_lang = direction.split("2")
    # Labels are written in Irish, matching the language the continued pretraining saw.
    src_label = LANG_NAMES[src_lang][0]
    tgt_label = LANG_NAMES[tgt_lang][0]
    src_fmt, tgt_fmt, instruction = CUES[cue]
    text = (instruction
            + src_fmt.format(src=src_label) + source_text + "\n"
            + tgt_fmt.format(tgt=tgt_label))
    return tokenizer.encode(text, add_special_tokens=False)


def load_pairs(data_path, n, direction):
    import pyarrow.parquet as pq

    src_lang, tgt_lang = direction.split("2")
    records = pq.read_table(
        data_path, columns=["id", "ga_raw", "en_raw"]).to_pylist()
    records.sort(key=lambda r: r["id"])
    if n:
        records = records[:n]
    pairs = [{"item": str(r["id"]),
              "source": r[f"{src_lang}_raw"],
              "ref": r[f"{tgt_lang}_raw"]} for r in records]
    print(f"[data] {len(pairs)} {direction} pairs", flush=True)
    return pairs


def evaluate(thinker, tokenizer, pairs, eos_id, cue, args):
    rows, hyps, refs = [], [], []
    started = time.time()
    for index, pair in enumerate(pairs, start=1):
        prompt = build_prompt(tokenizer, pair["source"], cue, args.direction)
        result = decode(thinker, tokenizer, prompt, eos_id, args.max_new_tokens)
        rows.append({**pair, "cue": cue, **result})
        hyps.append(result["hypothesis"])
        refs.append(pair["ref"])
        if index % 25 == 0 or index == len(pairs):
            print(f"[{args.label}/{cue}] {index}/{len(pairs)} "
                  f"({(time.time() - started) / index:.2f}s/item)", flush=True)

    bleu = BLEU_SCORER.corpus_score(hyps, [refs])
    chrfpp = CHRFPP_SCORER.corpus_score(hyps, [refs])
    return {"cue": cue, "chrfpp": chrfpp.score, "bleu": bleu.score, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="fleurs_parallel_test.parquet")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--direction", choices=["ga2en", "en2ga"], default="ga2en")
    ap.add_argument("--cues", nargs="+", choices=sorted(CUES), default=sorted(CUES))
    ap.add_argument("--n", type=int, default=50, help="limit pairs (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--eos-token", default="<|im_end|>")
    ap.add_argument("--model-dir", default=None,
                    help="tokenizer source (default: local base snapshot)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = load_pairs(args.data, args.n, args.direction)

    from transformers import AutoTokenizer
    from qomhra.checkpoint import load_for_eval

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir or base_snapshot())
    eos_id = tokenizer.convert_tokens_to_ids(args.eos_token)
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        raise SystemExit(f"tokenizer does not know {args.eos_token!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[{args.label}] loading {args.checkpoint} ...", flush=True)
    model, _ = load_for_eval(args.checkpoint, device=device, dtype=dtype)

    results = []
    for cue in args.cues:
        result = evaluate(model.thinker, tokenizer, pairs, eos_id, cue, args)
        gate = smoke_gate(result["rows"])
        print(f"\n[{args.label}/{cue}] chrF++ {result['chrfpp']:.2f} "
              f"BLEU {result['bleu']:.2f}", flush=True)
        print_smoke(f"mt / {args.label} / {cue}", gate, result["rows"], limit=5)
        results.append({**result, "smoke": gate})

    print("\n" + "=" * 72)
    print(f"{args.direction}  {args.label}")
    print("=" * 72)
    print(f"{'cue':14s} {'chrF++':>8s} {'BLEU':>8s} {'clean':>8s}")
    for result in sorted(results, key=lambda r: -r["chrfpp"]):
        clean = result["smoke"]["items"] - result["smoke"]["hit_token_cap"]
        print(f"{result['cue']:14s} {result['chrfpp']:8.2f} {result['bleu']:8.2f} "
              f"{clean:5d}/{result['smoke']['items']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({
                "run": {
                    "eval": f"fleurs parallel text {args.direction}",
                    "label": args.label,
                    "checkpoint": os.path.abspath(args.checkpoint),
                    "data": os.path.abspath(args.data),
                    "direction": args.direction,
                    "shots": 0,
                    "post_processing": "none",
                    "speech_input": "none (text in)",
                    "eos_token": args.eos_token,
                    "eos_id": eos_id,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "sacrebleu_version": sacrebleu.__version__,
                    "chrfpp_signature": str(CHRFPP_SCORER.get_signature()),
                },
                "results": results,
            }, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
