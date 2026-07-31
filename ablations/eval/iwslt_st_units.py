#!/usr/bin/env python3
"""IWSLT2023 ga->eng speech translation — zero-shot, unit-in, no post-processing.

Rebuild of `iwslt_st.py` under the rerun contract, changed the same three ways as
`cluas_qa_units.py`: zero-shot (was 3-shot), unit-in (was the untrained Qwen audio
tower), and scored on the raw decode (`strip_boilerplate()` and the extra
RAW_BACKUP_STOPS are gone -- only the trained EOS stops generation).

No windowing here. The dev set is 1,120 utterances, median 3.0 s and max 8.0 s, which is
~110 units at the median after dedup: comfortably inside 1024, and close to the 3.8 s
median the discrete-ASR runs trained on.

The task cue is the open question, not the length. Training only ever taught
<|transcript_start|> = "transcribe this Irish speech", so under a strictly native prompt
an ASR checkpoint is expected to emit Irish, and chrF++ against an English reference will
be low. That is a real measurement of the contract, not a broken harness. `--task-cue
english` adds a minimal English label as a declared, recorded departure.

  python -m eval.iwslt_st_units --data data/iwslt2023_dev.parquet \
      --units data/iwslt2023_dev.units.parquet --checkpoint <ckpt> --label asr250h \
      --n 10 --out output/iwslt_units_asr250h.json
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import sacrebleu  # noqa: E402
import torch  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "train")))

from discrete_eval_common import (  # noqa: E402
    AUDIO_MARKERS,
    SPEECH_END_TEXT,
    SPEECH_START_TEXT,
    TRANSCRIPT_START,
    TRANSCRIPT_START_TEXT,
    base_snapshot,
    decode,
    decode_prepared,
    print_smoke,
    reference_token_budget,
    smoke_gate,
    speech_span,
)

BLEU_SCORER = BLEU(tokenize="13a")
CHRFPP_SCORER = CHRF(char_order=6, word_order=2, beta=2)
ENGLISH_CUE = "\nEnglish:"
INSTRUCTED_PREFIX = (
    "Translate the following spoken Irish into English. Output only the English "
    "translation.\nIrish: "
)
INSTRUCTED_SUFFIX = "\nEnglish:"
NATIVE_INSTRUCTION = (
    "Translate the spoken Irish in the audio into English. "
    "Output only the English translation and nothing else."
)
BASE_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text "
    "and speech."
)
TEXT_EOS = "<|endoftext|>"
CHAT_EOS = "<|im_end|>"


def normalize_for_overlap(text):
    """Case/punctuation-neutral text used only for the language diagnostic."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def load_fotheidil_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        row["utterance_id"]: row["fotheidil_transcript"]
        for row in rows
    }
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate utterance IDs in {path}")
    return mapping


def build_prompt(tokenizer, units, task_cue):
    speech = speech_span(units)
    if task_cue == "instructed":
        return (
            tokenizer.encode(INSTRUCTED_PREFIX, add_special_tokens=False)
            + speech
            # Preserve the learned speech->text boundary. The instruction and
            # language cue specify translation, but they must not replace the
            # marker every paired-ASR example used before text generation.
            + [TRANSCRIPT_START]
            + tokenizer.encode(INSTRUCTED_SUFFIX, add_special_tokens=False)
        )
    prefix = speech
    if task_cue == "english":
        return prefix + tokenizer.encode(ENGLISH_CUE, add_special_tokens=False)
    return prefix + [TRANSCRIPT_START]


def build_tower_text(task_cue):
    """The ablations' raw prompt, with the waveform occupying the unit slot.

    Mirrors build_prompt() branch for branch. Base has no embedding for the three
    sentinels, so they go in as literal text and the tower's audio markers sit
    exactly where speech_span() puts the units.
    """
    speech = SPEECH_START_TEXT + AUDIO_MARKERS + SPEECH_END_TEXT
    if task_cue == "instructed":
        return (INSTRUCTED_PREFIX + speech + TRANSCRIPT_START_TEXT
                + INSTRUCTED_SUFFIX)
    if task_cue == "english":
        return speech + ENGLISH_CUE
    return speech + TRANSCRIPT_START_TEXT


def build_tower_chat(processor):
    conversation = [
        {"role": "system",
         "content": [{"type": "text", "text": BASE_SYSTEM}]},
        {"role": "user",
         "content": [
             {"type": "text", "text": NATIVE_INSTRUCTION},
             {"type": "audio", "audio": "c"},
         ]},
    ]
    return processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)


def load_split(data_path, units_path, n, selection_seed=2137, with_audio=False,
               fotheidil=None):
    import pyarrow.parquet as pq

    columns = ["name", "eng", "duration_s"]
    if with_audio:
        columns.append("audio")
    records = pq.read_table(data_path, columns=columns).to_pylist()
    records.sort(key=lambda r: str(r["name"]))

    units_table = pq.read_table(units_path, columns=["key", "units"])
    store = {k.as_py(): np.asarray(u.as_py(), dtype=np.int64)
             for k, u in zip(units_table["key"], units_table["units"])}

    tests = []
    for record in records:
        key = str(record["name"])
        if key not in store:
            raise SystemExit(f"no units for {key}; run prepare_eval_units.py")
        row = {"name": key, "eng": record["eng"],
               "duration_s": record["duration_s"], "units": store[key]}
        if fotheidil is not None:
            if key not in fotheidil:
                raise SystemExit(f"no FOTHEIDIL transcript for {key}")
            row["ga"] = fotheidil[key]
        if with_audio:
            from iwslt_st import decode_audio
            row["wav"] = decode_audio(record["audio"])
        tests.append(row)
    if n:
        tests = sorted(
            tests,
            key=lambda row: hashlib.sha256(
                f"{selection_seed}:{row['name']}".encode()).digest(),
        )[:n]
        tests.sort(key=lambda row: row["name"])
    lengths = [len(t["units"]) for t in tests]
    print(f"[data] {len(tests)} utterances; dedup units min {min(lengths)} "
          f"median {int(np.median(lengths))} max {max(lengths)}", flush=True)
    return tests


def evaluate(thinker, tokenizer, tests, eos_id, args, processor=None):
    rows, hyps, refs, ga_refs = [], [], [], []
    started = time.time()
    for index, test in enumerate(tests, start=1):
        reference_tokens, generation_budget = reference_token_budget(
            tokenizer,
            test["eng"],
            multiplier=args.reference_budget_multiplier,
            minimum=args.min_new_tokens,
            hard_max=args.max_new_tokens,
        )
        if processor is None:
            prompt = build_prompt(tokenizer, test["units"], args.task_cue)
            result = decode(
                thinker, tokenizer, prompt, eos_id, generation_budget)
        else:
            text = (build_tower_text(args.task_cue)
                    if args.prompt_format == "raw"
                    else build_tower_chat(processor))
            inputs = processor(
                text=text, audio=[test["wav"]], sampling_rate=16000,
                return_tensors="pt", padding=True).to(thinker.device)
            result = decode_prepared(
                thinker, tokenizer, inputs, eos_id, generation_budget)
        row = {"item": test["name"], "ref": test["eng"],
                     "fotheidil_ref": test.get("ga"),
                     "duration_s": test["duration_s"],
                     "units": len(test["units"]),
                     "reference_tokens": reference_tokens,
                     "generation_budget": generation_budget,
                     **result}
        rows.append(row)
        hyps.append(result["hypothesis"])
        refs.append(test["eng"])
        if test.get("ga") is not None:
            ga_refs.append(test["ga"])
        if index % 25 == 0 or index == len(tests):
            print(f"[{args.label}] {index}/{len(tests)} "
                  f"({(time.time() - started) / index:.2f}s/utt)", flush=True)

    bleu = BLEU_SCORER.corpus_score(hyps, [refs])
    chrfpp = CHRFPP_SCORER.corpus_score(hyps, [refs])
    language_diagnostic = None
    if ga_refs:
        hypotheses_normalized = [normalize_for_overlap(text) for text in hyps]
        english_normalized = [normalize_for_overlap(text) for text in refs]
        irish_normalized = [normalize_for_overlap(text) for text in ga_refs]
        en_score = CHRFPP_SCORER.corpus_score(
            hypotheses_normalized, [english_normalized]).score
        ga_score = CHRFPP_SCORER.corpus_score(
            hypotheses_normalized, [irish_normalized]).score
        counts = {"translation_like": 0, "transcription_like": 0, "tie": 0}
        for row, hypothesis, english, irish in zip(
                rows, hypotheses_normalized, english_normalized, irish_normalized):
            sentence_en = CHRFPP_SCORER.sentence_score(
                hypothesis, [english]).score
            sentence_ga = CHRFPP_SCORER.sentence_score(
                hypothesis, [irish]).score
            label = ("translation_like" if sentence_en > sentence_ga else
                     "transcription_like" if sentence_ga > sentence_en else "tie")
            counts[label] += 1
            row["sentence_chrfpp_normalized_vs_english"] = sentence_en
            row["sentence_chrfpp_normalized_vs_fotheidil"] = sentence_ga
            row["translation_minus_transcription"] = sentence_en - sentence_ga
            row["output_language_diagnostic"] = label
        language_diagnostic = {
            "chrfpp_normalized_vs_english": en_score,
            "chrfpp_normalized_vs_fotheidil": ga_score,
            "translation_minus_transcription": en_score - ga_score,
            "utterance_counts": counts,
        }
    return {
        "chrfpp": chrfpp.score,
        "bleu": bleu.score,
        "language_diagnostic": language_diagnostic,
        "detail": {
            "sacrebleu_version": sacrebleu.__version__,
            "bleu_signature": str(BLEU_SCORER.get_signature()),
            "chrfpp_signature": str(CHRFPP_SCORER.get_signature()),
            "bp": bleu.bp,
            "hyp_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
        },
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="iwslt dev parquet")
    ap.add_argument("--units", required=True, help="prepare_eval_units.py pack output")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=0, help="limit utterances (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--reference-budget-multiplier", type=float, default=1.25)
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--task-cue", choices=["none", "english", "instructed"],
                    default="instructed",
                    help="instructed mirrors the validated text ga->en cue")
    ap.add_argument("--eos-token", default="<|im_end|>")
    ap.add_argument("--additional-eos-token", action="append",
                    default=[TEXT_EOS],
                    help="also accept this trained document boundary (repeatable)")
    ap.add_argument("--baseline", action="store_true",
                    help="stock Qwen native audio tower")
    ap.add_argument("--prompt-format", choices=["chat", "raw"], default="chat",
                    help="baseline only: chat is base's own contract, raw is the "
                         "ablations' prompt with the waveform in the unit slot")
    ap.add_argument("--selection-seed", type=int, default=2137)
    ap.add_argument(
        "--fotheidil-csv",
        default=os.path.join(
            HERE, "output", "iwslt_fotheidil", "analysis",
            "iwslt_fotheidil_transcripts.csv"),
        help="consolidated Irish ASR references; empty string disables diagnostic")
    ap.add_argument("--model-dir", default=None, help="tokenizer source (default: local base snapshot)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fotheidil = (load_fotheidil_csv(args.fotheidil_csv)
                 if args.fotheidil_csv else None)
    tests = load_split(
        args.data, args.units, args.n, args.selection_seed,
        with_audio=args.baseline, fotheidil=fotheidil)

    from transformers import AutoTokenizer

    snapshot = args.model_dir or base_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    # Both boundaries occur legitimately across the raw/chat contracts.  Accepting
    # either stops generation but preserves the exact emitted token in every row.
    # Keyed on the prompt contract, not on which model it is: a baseline run on the
    # ablations' raw prompt must accept the ablations' boundary, not the chat one.
    chat_contract = args.baseline and args.prompt_format == "chat"
    eos_tokens = [CHAT_EOS if chat_contract else args.eos_token]
    eos_tokens += args.additional_eos_token
    eos_tokens = list(dict.fromkeys(eos_tokens))
    eos_id = [tokenizer.convert_tokens_to_ids(token) for token in eos_tokens]
    if any(value is None or value == tokenizer.unk_token_id for value in eos_id):
        raise SystemExit(f"tokenizer does not know one of {eos_tokens!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[{args.label}] loading {args.checkpoint} ...", flush=True)
    processor = None
    if args.baseline:
        from transformers import Qwen2_5OmniProcessor
        from generate_compare import load_baseline
        processor = Qwen2_5OmniProcessor.from_pretrained(snapshot)
        thinker = load_baseline(args.checkpoint, device, dtype, attn="eager")
    else:
        from qomhra.checkpoint import load_for_eval
        model, _ = load_for_eval(args.checkpoint, device=device, dtype=dtype)
        thinker = model.thinker

    result = evaluate(
        thinker, tokenizer, tests, eos_id, args, processor=processor)
    result["run"] = {
        "eval": "iwslt2023 ga->eng",
        "label": args.label,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "units": os.path.abspath(args.units),
        "shots": 0,
        "post_processing": "none",
        "speech_input": ("native Qwen audio tower" if args.baseline else
                         "mhubert units in expanded vocab"),
        "prompt_format": ("chat" if chat_contract else
                          f"raw {args.task_cue}"),
        "task_cue": "native_chat" if chat_contract else args.task_cue,
        "accepted_eos_tokens": eos_tokens,
        "accepted_eos_ids": eos_id,
        "selection_seed": args.selection_seed,
        "fotheidil_csv": (os.path.abspath(args.fotheidil_csv)
                           if args.fotheidil_csv else None),
        "max_new_tokens": args.max_new_tokens,
        "generation_budget": (
            "per row: ceil(reference tokens * "
            f"{args.reference_budget_multiplier}) + 1, minimum "
            f"{args.min_new_tokens}, hard ceiling {args.max_new_tokens}"
        ),
        "do_sample": False,
        "device": device,
        "dtype": str(dtype),
    }

    gate = smoke_gate(result["rows"])
    print(f"\n[{args.label}] chrF++ {result['chrfpp']:.2f}  BLEU {result['bleu']:.2f}",
          flush=True)
    if result["language_diagnostic"]:
        diag = result["language_diagnostic"]
        print(f"[{args.label}] normalized chrF++ English "
              f"{diag['chrfpp_normalized_vs_english']:.2f} / FOTHEIDIL "
              f"{diag['chrfpp_normalized_vs_fotheidil']:.2f}; "
              f"translation-minus-transcription "
              f"{diag['translation_minus_transcription']:+.2f}; "
              f"{diag['utterance_counts']}", flush=True)
    print_smoke(f"iwslt / {args.label}", gate, result["rows"])

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({**result, "smoke": gate}, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
