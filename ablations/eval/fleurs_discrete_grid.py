#!/usr/bin/env python3
"""Watertight FLEURS pre-eval for the discrete-speech ablations.

The six headings match the unordered MEXA comparison grid.  FLEURS can measure
five of them behaviourally; speech_ga~speech_en is retained explicitly as N/A
because these models are not evaluated as speech generators.  The text/text
heading contains both translation directions.

Discrete checkpoints receive mHuBERT unit IDs as ordinary vocabulary items.
The stock base model receives the original waveform through its native audio
tower.  Every decode is greedy, batch one, keeps its untouched generated IDs
and raw text, and stops *during decoding* on either structural EOS used during
training.  There is no regex trimming or other post-generation cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

torch.backends.cudnn.enabled = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from asr_baseline import wer  # noqa: E402
from discrete_eval_common import (  # noqa: E402
    AUDIO_MARKERS,
    BASE_VOCAB,
    SPEECH_END_TEXT,
    SPEECH_START_TEXT,
    TRANSCRIPT_START,
    TRANSCRIPT_START_TEXT,
    base_snapshot,
    decode,
    decode_prepared,
    reference_token_budget,
    speech_span,
)
from fleurs_eval import cer  # noqa: E402
from iwslt_st import BLEU_SCORER, CHRFPP_SCORER, TARGET_SR, decode_audio  # noqa: E402


END_OF_TEXT = 151643
IM_END = 151645
EOS_IDS = (END_OF_TEXT, IM_END)
EOS_NAMES = {END_OF_TEXT: "<|endoftext|>", IM_END: "<|im_end|>"}
MAX_SEQUENCE_LENGTH = 1024

# Six MEXA headings, but six executable directions: text/text is bidirectional
# while speech/speech is deliberately N/A.
PAIR_GRID = {
    "speech_ga~text_ga": ("asr_ga",),
    "speech_en~text_en": ("asr_en",),
    "text_ga~text_en": ("text_ga2en", "text_en2ga"),
    "speech_ga~speech_en": (),
    "speech_ga~text_en": ("st_ga2en",),
    "speech_en~text_ga": ("st_en2ga",),
}

CONDITIONS = {
    "asr_ga": {
        "pair": "speech_ga~text_ga",
        "input_modality": "speech",
        "source_lang": "ga",
        "target_lang": "ga",
        "metric": "wer",
        # The unit ASR sequence itself is the instruction learned in training.
        "instruction": "",
        "cue": "",
    },
    "asr_en": {
        "pair": "speech_en~text_en",
        "input_modality": "speech",
        "source_lang": "en",
        "target_lang": "en",
        "metric": "wer",
        "instruction": "",
        "cue": "",
    },
    "text_ga2en": {
        "pair": "text_ga~text_en",
        "input_modality": "text",
        "source_lang": "ga",
        "target_lang": "en",
        "metric": "chrf",
        "instruction": "Translate the following Irish text into English.",
        "source_label": "Irish",
        "cue": "English:",
    },
    "text_en2ga": {
        "pair": "text_ga~text_en",
        "input_modality": "text",
        "source_lang": "en",
        "target_lang": "ga",
        "metric": "chrf",
        "instruction": "Translate the following English text into Irish.",
        "source_label": "English",
        "cue": "Irish:",
    },
    "st_ga2en": {
        "pair": "speech_ga~text_en",
        "input_modality": "speech",
        "source_lang": "ga",
        "target_lang": "en",
        "metric": "chrf",
        "instruction": "Translate the spoken Irish into English.",
        "cue": "English:",
    },
    "st_en2ga": {
        "pair": "speech_en~text_ga",
        "input_modality": "speech",
        "source_lang": "en",
        "target_lang": "ga",
        "metric": "chrf",
        "instruction": "Translate the spoken English into Irish.",
        "cue": "Irish:",
    },
}


def text_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _target(row, lang: str, metric: str) -> str:
    # FLEURS-normalised text for transcription; raw human translation for chrF.
    return row[f"{lang}_{'norm' if metric == 'wer' else 'raw'}"]


def choose_ids(unit_rows, n: int, demo_pool_size: int, max_source_units: int):
    """Choose one parallel, deterministic subset for every condition.

    Three shortest bilingual items form a fixed demonstration pool even for a
    zero-shot run, so comparing 0/1/3 shots never changes the scored sentences.
    Scored IDs are then the first sorted IDs whose *both* speech versions fit the
    declared source budget.  This prevents condition-specific dropping.
    """
    lengths: dict[int, dict[str, int]] = {}
    for row in unit_rows:
        lengths.setdefault(int(row["id"]), {})[row["lang"]] = len(row["units"])
    parallel = {
        item_id: langs for item_id, langs in lengths.items()
        if set(langs) == {"ga", "en"}
    }
    demos = sorted(
        parallel, key=lambda item_id: (max(parallel[item_id].values()), item_id)
    )[:demo_pool_size]
    eligible = [
        item_id for item_id in sorted(parallel)
        if item_id not in set(demos)
        and max(parallel[item_id].values()) <= max_source_units
    ]
    if len(eligible) < n:
        raise ValueError(
            f"only {len(eligible)} parallel IDs fit <= {max_source_units} units; "
            f"{n} requested"
        )
    return demos, eligible[:n], parallel


def build_discrete_item(tokenizer, spec, row, units, include_answer=False):
    """Build one plain-token discrete example in the model's learned interface."""
    if spec["input_modality"] == "speech":
        prefix = []
        if spec["instruction"]:
            prefix += text_ids(tokenizer, spec["instruction"] + "\n")
        prefix += speech_span(units) + [TRANSCRIPT_START]
        if spec["cue"]:
            prefix += text_ids(tokenizer, "\n" + spec["cue"] + " ")
        demo_eos = IM_END
    else:
        prompt = (
            f"{spec['instruction']}\n"
            f"{spec['source_label']}: {row[spec['source_lang'] + '_raw']}\n"
            f"{spec['cue']} "
        )
        prefix = text_ids(tokenizer, prompt)
        demo_eos = END_OF_TEXT
    if include_answer:
        prefix += text_ids(
            tokenizer, _target(row, spec["target_lang"], spec["metric"])
        )
        prefix.append(demo_eos)
    return prefix


def build_discrete_prompt(tokenizer, spec, row, units, demos):
    prompt = []
    for demo_row, demo_units in demos:
        prompt += build_discrete_item(
            tokenizer, spec, demo_row, demo_units, include_answer=True
        )
    prompt += build_discrete_item(
        tokenizer, spec, row, units, include_answer=False
    )
    return prompt


def prompt_continuation_stops(spec, shots):
    """Markers that unambiguously begin another example in this exact prompt.

    These are decoding boundaries, not post-hoc cleaners.  They are enabled only
    when demonstrations make continuation possible; untouched IDs/text and the
    matched marker remain in the row diagnostics.
    """
    if not shots:
        return None
    if spec["input_modality"] == "text":
        return [f"\n{spec['source_label']}:"]
    if spec["instruction"]:
        return [spec["instruction"]]
    return None


def _base_instruction(spec):
    if spec["metric"] == "wer":
        language = "Irish" if spec["target_lang"] == "ga" else "English"
        return (
            f"Transcribe the {language} audio exactly. "
            f"Output only the {language} transcription."
        )
    if spec["input_modality"] == "speech":
        source = "Irish" if spec["source_lang"] == "ga" else "English"
        target = "English" if spec["target_lang"] == "en" else "Irish"
        return (
            f"Translate the spoken {source} into {target}. "
            f"Output only the {target} translation."
        )
    return spec["instruction"] + " Output only the translation."


def build_base_inputs(processor, spec, row, demos, device):
    """Build native stock-Qwen chat input, including waveform audio where needed."""
    conversation = [{
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    }]
    audios = []

    def user_turn(item):
        content = [{"type": "text", "text": _base_instruction(spec)}]
        if spec["input_modality"] == "speech":
            content.append({"type": "audio", "audio": "x"})
            audios.append(item[f"{spec['source_lang']}_audio_decoded"])
        else:
            content.append({
                "type": "text",
                "text": item[f"{spec['source_lang']}_raw"],
            })
        return {"role": "user", "content": content}

    for demo_row, _ in demos:
        conversation.append(user_turn(demo_row))
        conversation.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": _target(demo_row, spec["target_lang"], spec["metric"]),
            }],
        })
    conversation.append(user_turn(row))
    rendered = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    return processor(
        text=rendered,
        audio=(audios or None),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    ).to(device)


def build_base_raw_item(spec, row, include_answer=False):
    """build_discrete_item() as text, for a model without the expanded vocabulary.

    Same clauses in the same order; the only substitution is inside the speech span,
    where the tower's audio markers take the place of the unit ids and the sentinels
    are spelled out rather than emitted as ids base has no embedding for.
    """
    if spec["input_modality"] == "speech":
        prompt = spec["instruction"] + "\n" if spec["instruction"] else ""
        prompt += SPEECH_START_TEXT + AUDIO_MARKERS + SPEECH_END_TEXT
        prompt += TRANSCRIPT_START_TEXT
        if spec["cue"]:
            prompt += "\n" + spec["cue"] + " "
        demo_eos = EOS_NAMES[IM_END]
    else:
        prompt = (
            f"{spec['instruction']}\n"
            f"{spec['source_label']}: {row[spec['source_lang'] + '_raw']}\n"
            f"{spec['cue']} "
        )
        demo_eos = EOS_NAMES[END_OF_TEXT]
    if include_answer:
        prompt += _target(row, spec["target_lang"], spec["metric"]) + demo_eos
    return prompt


def build_base_raw_inputs(processor, spec, row, demos, device):
    """The ablations' prompt, waveform in the unit slot, through the processor."""
    text = "".join(
        build_base_raw_item(spec, demo_row, include_answer=True)
        for demo_row, _ in demos
    )
    text += build_base_raw_item(spec, row)
    audios = None
    if spec["input_modality"] == "speech":
        key = f"{spec['source_lang']}_audio_decoded"
        audios = [demo_row[key] for demo_row, _ in demos] + [row[key]]
    return processor(
        text=text,
        audio=audios,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    ).to(device)


def _wrong_modality(report):
    # Unit ids for the expanded-vocab arms; echoed sentinel text for base_raw, which
    # has no unit ids to emit. Both mean the model answered with prompt scaffolding
    # instead of the requested text.
    return (int(report.get("unit_ids_in_answer", 0)) > 0
            or int(report.get("sentinel_text_in_answer", 0)) > 0)


def score_rows(spec, rows):
    refs = [row["reference"] for row in rows]
    hyps = [row["hypothesis"] for row in rows]
    if spec["metric"] == "wer":
        w_edits = w_n = c_edits = c_n = 0
        for ref, hyp in zip(refs, hyps):
            edits, length = wer(ref, hyp)
            w_edits += edits
            w_n += length
            edits, length = cer(ref, hyp)
            c_edits += edits
            c_n += length
        scores = {
            "wer": 100.0 * w_edits / max(w_n, 1),
            "cer": 100.0 * c_edits / max(c_n, 1),
        }
    else:
        scores = {
            "chrfpp": CHRFPP_SCORER.corpus_score(hyps, [refs]).score,
            "bleu": BLEU_SCORER.corpus_score(hyps, [refs]).score,
        }
    stops = Counter(row["stop_reason"] for row in rows)
    scores["diagnostics"] = {
        "empty": sum(not row["hypothesis"].strip() for row in rows),
        "cap_hits": stops["max_new_tokens"],
        "loops": sum(bool(row["looping"]) for row in rows),
        "wrong_modality": sum(bool(row["wrong_modality"]) for row in rows),
        "unit_ids_in_answers": sum(row["unit_ids_in_answer"] for row in rows),
        "stop_reasons": dict(stops),
        "mismatch_identical": sum(
            bool(row.get("mismatch_identical")) for row in rows
        ),
        "mismatch_items": sum("mismatch_identical" in row for row in rows),
    }
    return scores


def load_tables(data_path, units_path):
    import pyarrow.parquet as pq

    unit_rows = pq.read_table(units_path).to_pylist()
    data_rows = pq.read_table(data_path).to_pylist()
    by_id = {int(row["id"]): row for row in data_rows}
    units = {
        (int(row["id"]), row["lang"]): np.asarray(row["units"], dtype=np.int64)
        for row in unit_rows
    }
    return by_id, units, unit_rows


def attach_audio(rows, languages):
    for row in rows:
        for lang in languages:
            key = f"{lang}_audio_decoded"
            if key not in row:
                row[key] = decode_audio(row[f"{lang}_audio"])


def run_condition(
    thinker,
    tokenizer,
    processor,
    spec,
    rows,
    units_by_key,
    demos,
    baseline,
    max_new_tokens,
    mismatch_controls,
    device,
    shots,
    reference_budget_multiplier,
    min_new_tokens,
    base_prompt_format="chat",
):
    build_base = (build_base_raw_inputs if base_prompt_format == "raw"
                  else build_base_inputs)
    results = []
    started = time.time()
    for index, row in enumerate(rows):
        lang = spec["source_lang"]
        source_units = units_by_key[(int(row["id"]), lang)]
        reference = _target(row, spec["target_lang"], spec["metric"])
        reference_tokens, generation_budget = reference_token_budget(
            tokenizer,
            reference,
            multiplier=reference_budget_multiplier,
            minimum=min_new_tokens,
            hard_max=max_new_tokens,
        )
        if baseline:
            inputs = build_base(processor, spec, row, demos, device)
            report = decode_prepared(
                thinker, tokenizer, inputs, EOS_IDS, generation_budget,
                stop_strings=prompt_continuation_stops(spec, shots),
            )
        else:
            demo_units = [
                (demo, units_by_key[(int(demo["id"]), lang)])
                for demo, _ in demos
            ]
            prompt = build_discrete_prompt(
                tokenizer, spec, row, source_units, demo_units
            )
            if len(prompt) + generation_budget > MAX_SEQUENCE_LENGTH:
                raise ValueError(
                    f"{row['id']} {spec['pair']} prompt {len(prompt)} + "
                    f"answer {generation_budget} exceeds {MAX_SEQUENCE_LENGTH}"
                )
            report = decode(
                thinker, tokenizer, prompt, EOS_IDS, generation_budget,
                stop_strings=prompt_continuation_stops(spec, shots),
            )
        # The shared decoder preserves every generated ID, including the terminal
        # structural token.  Preserve its literal rendering too; `hypothesis` is
        # only the answer IDs before that terminal token, with no cleanup.
        report["raw_output_text"] = tokenizer.decode(
            report["raw_output_ids"], skip_special_tokens=False
        )
        result = {
            "id": int(row["id"]),
            "reference": reference,
            "reference_tokens": reference_tokens,
            "generation_budget": generation_budget,
            **report,
            "wrong_modality": _wrong_modality(report),
            "source_units": len(source_units),
        }

        if mismatch_controls and spec["input_modality"] == "speech":
            mismatch_row = rows[(index + 1) % len(rows)]
            if baseline:
                attach_audio([mismatch_row], [lang])
                mismatch_inputs = build_base(
                    processor, spec, mismatch_row, demos, device
                )
                mismatch = decode_prepared(
                    thinker, tokenizer, mismatch_inputs, EOS_IDS,
                    generation_budget,
                    stop_strings=prompt_continuation_stops(spec, shots),
                )
            else:
                mismatch_units = units_by_key[(int(mismatch_row["id"]), lang)]
                mismatch_prompt = build_discrete_prompt(
                    tokenizer, spec, row, mismatch_units, demo_units
                )
                mismatch = decode(
                    thinker, tokenizer, mismatch_prompt, EOS_IDS,
                    generation_budget,
                    stop_strings=prompt_continuation_stops(spec, shots),
                )
            mismatch["raw_output_text"] = tokenizer.decode(
                mismatch["raw_output_ids"], skip_special_tokens=False
            )
            result.update({
                "mismatch_source_id": int(mismatch_row["id"]),
                "mismatch_hypothesis": mismatch["hypothesis"],
                "mismatch_raw_output_ids": mismatch["raw_output_ids"],
                "mismatch_raw_output_text": mismatch["raw_output_text"],
                "mismatch_stop_reason": mismatch["stop_reason"],
                "mismatch_identical": (
                    mismatch["answer_ids"] == report["answer_ids"]
                ),
            })
        results.append(result)
        print(
            f"[{spec['pair']}] {index + 1}/{len(rows)} "
            f"{(time.time() - started) / (index + 1):.2f}s/item",
            flush=True,
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--checkpoint", help="omit for stock native-audio base")
    parser.add_argument("--baseline", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument(
        "--base-prompt-format", choices=["chat", "raw"], default="chat",
        help="stock base only: chat is its own contract, raw is the ablations' "
             "prompt with the waveform occupying the unit slot")
    parser.add_argument("--label", required=True)
    parser.add_argument("--n", type=int, default=34)
    parser.add_argument("--shots", type=int, choices=[0, 1, 3], default=0)
    parser.add_argument("--demo-pool-size", type=int, default=3)
    parser.add_argument("--max-source-units", type=int, default=760)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--reference-budget-multiplier", type=float, default=1.25)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument(
        "--mismatch-controls", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    unknown = set(args.conditions) - set(CONDITIONS)
    if unknown:
        parser.error(f"unknown conditions: {sorted(unknown)}")
    if args.mismatch_controls and args.n < 2:
        parser.error("--mismatch-controls requires --n >= 2 so the control uses "
                     "a genuinely different utterance")

    by_id, units_by_key, unit_rows = load_tables(args.data, args.units)
    demo_ids, scored_ids, lengths = choose_ids(
        unit_rows, args.n, args.demo_pool_size, args.max_source_units
    )
    missing = [item_id for item_id in demo_ids + scored_ids if item_id not in by_id]
    if missing:
        raise SystemExit(f"unit IDs absent from FLEURS parquet: {missing}")
    rows = [by_id[item_id] for item_id in scored_ids]
    demo_rows = [by_id[item_id] for item_id in demo_ids[:args.shots]]
    demos = [(row, lengths[int(row["id"])]) for row in demo_rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    snapshot = base_snapshot(args.baseline)

    if args.checkpoint:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        sys.path.insert(0, str(HERE.parent / "train"))
        from qomhra.checkpoint import load_for_eval

        print(f"[{args.label}] loading discrete checkpoint {args.checkpoint}", flush=True)
        model, _ = load_for_eval(args.checkpoint, device=device, dtype=dtype)
        thinker = model.thinker
        processor = None
        baseline = False
    else:
        from transformers import Qwen2_5OmniProcessor
        from generate_compare import load_baseline

        print(f"[{args.label}] loading stock base through native tower", flush=True)
        processor = Qwen2_5OmniProcessor.from_pretrained(snapshot)
        tokenizer = processor.tokenizer
        thinker = load_baseline(snapshot, device, dtype, attn="eager")
        baseline = True
        speech_langs = {
            CONDITIONS[name]["source_lang"] for name in args.conditions
            if CONDITIONS[name]["input_modality"] == "speech"
        }
        attach_audio(rows + demo_rows, speech_langs)

    output = {
        "run": {
            "label": args.label,
            "checkpoint": args.checkpoint,
            "baseline_native_audio_tower": baseline,
            "prompt_format": (args.base_prompt_format if baseline else "raw"),
            "n": len(rows),
            "scored_ids": scored_ids,
            "fixed_demo_pool_ids": demo_ids,
            "active_demo_ids": demo_ids[:args.shots],
            "shots": args.shots,
            "max_source_units": args.max_source_units,
            "max_new_tokens": args.max_new_tokens,
            "generation_budget": (
                "per row: ceil(reference tokens * "
                f"{args.reference_budget_multiplier}) + 1, minimum "
                f"{args.min_new_tokens}, hard ceiling {args.max_new_tokens}"
            ),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "eos_ids": list(EOS_IDS),
            "eos_names": EOS_NAMES,
            "generation": "greedy, batch 1, raw IDs/text retained",
            "post_processing": "none",
            "mismatch_controls": args.mismatch_controls,
            "prompt_continuation_stops": {
                name: prompt_continuation_stops(CONDITIONS[name], args.shots)
                for name in args.conditions
            },
            "pair_grid": PAIR_GRID,
            "speech_ga~speech_en": (
                "N/A: representational-only; no defensible FLEURS speech-generation task"
            ),
        },
        "conditions": {},
    }

    for name in args.conditions:
        spec = CONDITIONS[name]
        condition_rows = run_condition(
            thinker, tokenizer, processor, spec, rows, units_by_key, demos,
            baseline, args.max_new_tokens, args.mismatch_controls, device,
            args.shots, args.reference_budget_multiplier, args.min_new_tokens,
            base_prompt_format=args.base_prompt_format,
        )
        output["conditions"][name] = {
            "spec": spec,
            "summary": score_rows(spec, condition_rows),
            "rows": condition_rows,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
