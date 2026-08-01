#!/usr/bin/env python3
"""FLEURS Irish transcription from mHuBERT units. Zero-shot, native contract, no cleanup.

The prompt IS the training contract, so there is no prompt engineering to get wrong:

    <|speech_start|> units <|speech_end|> <|transcript_start|>   ->   text <|im_end|>

Few-shot is deliberately not an option here. On the 250h model three-shot produced an
`identical_output_rate` of 0.96 between correct and mismatched audio -- the demonstrations
carried the answer and the audio was disconnected. A prompt that scores well while ignoring
its input measures the prompt.

Nothing rewrites a hypothesis. Tokens at or after the trained EOS are dropped, which is
where the model said it stopped, and `cut_at_special` ends the answer at the first structural
marker for checkpoints that emit <|endoftext|> instead. Both are recorded per row alongside
the untouched decode, so the smoke gate is checkable from the output file alone.

CER is the headline, not WER. Unbounded WER is not order-preserving on this task: on the 50h
run a silent model scored 96.89 against 97.28 for one that was really transcribing.

    python fleurs_units_asr.py --units <parquet> --data <fleurs parquet> \
        --checkpoint <ckpt> --label asr250h --n 10 --out out.json
"""
import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discrete_eval_common import (  # noqa: E402
    TRANSCRIPT_START, base_snapshot, decode, print_smoke, smoke_gate, speech_span)
from fleurs_eval import cer  # noqa: E402
# Same normalisation and same edit costs as every other suite in this directory, so a
# CER here and a CER there are the same measurement.
from asr_baseline import wer  # noqa: E402

# Any <|...|> run is a special token in this tokenizer. Matching the shape rather than a
# fixed list means a marker we did not anticipate still ends the answer. Defined here
# rather than imported so this script does not depend on which revision of fleurs_eval.py
# a given machine happens to be carrying.
SPECIAL = re.compile(r"<\|[^|>]{1,40}\|>")


def cut_at_special(text):
    """Everything from the first structural marker onwards is not the answer.

    Decoding stops on the stop token the contract asked for, but a continued-pretraining
    checkpoint trained on <|endoftext|>-separated documents may emit that instead: the
    generation is then correct up to the marker and runaway text after it. The tokenizer's
    skip_special_tokens removes the marker but keeps what follows, so the cut has to be
    made on the decoded string. Both forms are recorded per row.
    """
    match = SPECIAL.search(text)
    return text[:match.start()] if match else text


def load_rows(units_parquet, data_parquet, lang, n):
    """Join units to their FLEURS reference, in sentence-id order."""
    import pyarrow.parquet as pq
    refs = {r["id"]: r for r in pq.read_table(
        data_parquet, columns=["id", "split", f"{lang}_raw", f"{lang}_norm"]).to_pylist()}
    units = [r for r in pq.read_table(units_parquet).to_pylist() if r["lang"] == lang]
    units.sort(key=lambda r: r["id"])

    rows = []
    for r in units[:n] if n else units:
        ref = refs.get(r["id"])
        if ref is None:
            raise SystemExit(f"sentence {r['id']} has units but no reference row")
        if ref["split"] != "test":
            raise SystemExit(f"sentence {r['id']} is split={ref['split']}, expected test")
        rows.append({
            "id": r["id"],
            "units": np.asarray(r["units"], dtype=np.int64),
            # FLEURS' own normalised column, not a homegrown normaliser: Irish lenition,
            # eclipsis and the sineadh fada make homegrown normalisation a silent error source.
            "reference": ref[f"{lang}_norm"],
            "reference_raw": ref[f"{lang}_raw"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="fleurs_mexa_units.py attach output")
    ap.add_argument("--data", required=True, help="fleurs_parallel_test.parquet")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--lang", default="ga", choices=["ga", "en"])
    ap.add_argument("--n", type=int, default=10, help="0 = all")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(args.units, args.data, args.lang, args.n)
    print(f"[data] {len(rows)} {args.lang} test sentences, "
          f"ids {rows[0]['id']}..{rows[-1]['id']}", flush=True)

    from transformers import AutoTokenizer
    snapshot = base_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
    from qomhra.checkpoint import load_for_eval
    print(f"[{args.label}] loading {args.checkpoint} ...", flush=True)
    model, _ = load_for_eval(args.checkpoint, device="cuda", dtype=torch.bfloat16)
    thinker = model.thinker

    vocab_rows = thinker.get_input_embeddings().weight.shape[0]
    if vocab_rows <= TRANSCRIPT_START:
        raise SystemExit(f"[{args.label}] vocab has {vocab_rows} rows; this checkpoint was "
                         f"not trained with the expanded speech vocabulary")
    print(f"[{args.label}] vocab {vocab_rows} rows, eos <|im_end|>={eos_id}", flush=True)

    results, started = [], time.time()
    for index, row in enumerate(rows, 1):
        prompt = speech_span(row["units"]) + [TRANSCRIPT_START]
        report = decode(thinker, tokenizer, prompt, eos_id, args.max_new_tokens)
        # Recorded, not silently applied: both forms are kept so the effect is visible.
        cut = cut_at_special(report["hypothesis"])
        c_edits, c_n = cer(row["reference"], cut)
        w_edits, w_n = wer(row["reference"], cut)
        results.append({
            **row, "units": len(row["units"]),
            **report,
            "hypothesis_cut": cut,
            "cut_removed_chars": len(report["hypothesis"]) - len(cut),
            "cer_edits": c_edits, "cer_n": c_n,
            "wer_edits": w_edits, "wer_n": w_n,
        })
        print(f"[{args.label}] {index}/{len(rows)} "
              f"({(time.time() - started) / index:.1f}s each)", flush=True)

    gate = smoke_gate(results)
    corpus_cer = 100.0 * sum(r["cer_edits"] for r in results) / max(
        sum(r["cer_n"] for r in results), 1)
    corpus_wer = 100.0 * sum(r["wer_edits"] for r in results) / max(
        sum(r["wer_n"] for r in results), 1)

    print_smoke(f"{args.label} FLEURS {args.lang} ASR (units, zero-shot)", gate, results)
    print(f"\nCER {corpus_cer:.2f}   WER {corpus_wer:.2f}   (CER is the headline)")
    print("FOTHEIDIL, the in-house production ASR, scores WER 46.46 / CER 22.32 on FLEURS.")

    for r in results:
        print(f"\n--- id {r['id']}  ({r['units']} units, {r['generated_tokens']} generated, "
              f"stop={r['stop_reason']})")
        print(f"  REF {r['reference']}")
        print(f"  HYP {r['hypothesis']!r}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "label": args.label,
            "checkpoint": os.path.abspath(args.checkpoint),
            "lang": args.lang,
            "n": len(results),
            "prompt_contract": "<|speech_start|> units <|speech_end|> <|transcript_start|>",
            "shots": 0,
            "post_processing": "none; EOS truncation and cut_at_special recorded per row",
            "decoding": "greedy, batch 1, bf16",
            "max_new_tokens": args.max_new_tokens,
            "reference_column": f"{args.lang}_norm",
            "smoke_gate": gate,
            "corpus_cer": round(corpus_cer, 2),
            "corpus_wer": round(corpus_wer, 2),
            "rows": results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
