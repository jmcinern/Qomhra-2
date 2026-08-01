#!/usr/bin/env python3
"""Greedy ASR evaluation from mHuBERT unit-token prompts on 50 FLEURS clips."""

import argparse
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, "..", "train"))
sys.path.insert(0, TRAIN)
from asr_baseline import norm, wer  # noqa: E402
from fleurs_eval import cer  # noqa: E402
from qomhra.checkpoint import load_for_eval  # noqa: E402

BASE_VOCAB = 151936


def local_snapshot(model_id):
    if os.path.isdir(model_id):
        return model_id
    pattern = os.path.join(
        os.environ.get("HF_HOME", ""),
        "hub",
        "models--" + model_id.replace("/", "--"),
        "snapshots",
        "*",
    )
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise RuntimeError(f"no local snapshot for {model_id}")
    return hits[0]


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--demos", help="JSONL of complete unit-to-transcript examples")
    ap.add_argument(
        "--rotate-audio",
        type=int,
        default=0,
        help="mismatch query audio by this many rows (semantic-conditioning control)",
    )
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    model, cfg = load_for_eval(args.checkpoint, device=device, dtype=dtype)
    thinker = model.thinker
    thinker.config.use_cache = True

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local_snapshot(cfg.model.base_model_id))
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    rows = load_rows(args.data)
    demos = load_rows(args.demos) if args.demos else []
    demo_ids = [token for demo in demos for token in demo["input_ids"]]
    query_ids = [row["input_ids"] for row in rows]
    if args.rotate_audio:
        shift = args.rotate_audio % len(rows)
        if shift == 0:
            raise ValueError("--rotate-audio must not be a multiple of the dataset size")
        query_ids = query_ids[shift:] + query_ids[:shift]
    results = []
    total_we = total_ww = total_ce = total_cc = 0
    text_cfg = getattr(thinker.config, "text_config", thinker.config)
    max_context = int(getattr(text_cfg, "max_position_embeddings", 32768))

    for index, (row, audio_ids) in enumerate(zip(rows, query_ids), 1):
        combined = demo_ids + audio_ids
        if len(combined) + args.max_new_tokens > max_context:
            raise RuntimeError(
                f"id={row['id']} prompt+decode exceeds context: "
                f"{len(combined)}+{args.max_new_tokens}"
            )
        prompt = torch.tensor([combined], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = thinker.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos,
                pad_token_id=eos,
                use_cache=True,
            )[0, prompt.shape[1]:].tolist()
        if eos in generated:
            generated = generated[: generated.index(eos)]
        # Unit IDs are not part of the stock tokenizer. If a weak model emits one,
        # omit it from the text decode but record the count as a diagnostic.
        emitted_units = sum(token >= BASE_VOCAB for token in generated)
        text_ids = [token for token in generated if token < len(tok)]
        hypothesis = tok.decode(text_ids, skip_special_tokens=True).strip()
        reference = row["reference"]
        we, ww = wer(reference, hypothesis)
        ce, cc = cer(reference, hypothesis)
        total_we += we
        total_ww += ww
        total_ce += ce
        total_cc += cc
        results.append(
            {
                "id": row["id"],
                "reference": reference,
                "hypothesis": hypothesis,
                "reference_norm": norm(reference),
                "hypothesis_norm": norm(hypothesis),
                "word_edits": we,
                "reference_words": ww,
                "char_edits": ce,
                "reference_chars": cc,
                "emitted_unit_tokens": emitted_units,
                "generation_tokens": len(generated),
            }
        )
        print(
            f"[{index:02d}/{len(rows)}] id={row['id']} "
            f"word_edits={we}/{ww} units_out={emitted_units}",
            flush=True,
        )

    report = {
        "label": args.label,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "demos": os.path.abspath(args.demos) if args.demos else None,
        "demo_count": len(demos),
        "rotate_audio": args.rotate_audio,
        "n": len(results),
        "wer": 100.0 * total_we / max(total_ww, 1),
        "cer": 100.0 * total_ce / max(total_cc, 1),
        "word_edits": total_we,
        "reference_words": total_ww,
        "char_edits": total_ce,
        "reference_chars": total_cc,
        "rows": results,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"{args.label}: WER={report['wer']:.2f} CER={report['cer']:.2f} "
        f"on n={len(results)}; wrote {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
