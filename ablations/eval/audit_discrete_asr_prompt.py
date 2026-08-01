#!/usr/bin/env python3
"""Gate the 50-row run with a raw 5-row prompt/effectiveness audit.

One model load evaluates the same rows under:
  zero_shot          query unit prefix only
  three_shot         three complete unit->transcript->EOS demonstrations
  three_shot_mismatch the same prompt, but another row's query audio

No hypothesis truncation or cleanup is performed.  The report retains raw output
IDs, explicit EOS status, repetition diagnostics and cross-condition equality.
"""

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

SPEECH_START = 152936
SPEECH_END = 152937
TRANSCRIPT_START = 152938
EXPANDED_VOCAB = 152939


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def snapshot(model_id):
    if os.path.isdir(model_id):
        return model_id
    pattern = os.path.join(
        os.environ["HF_HOME"],
        "hub",
        "models--" + model_id.replace("/", "--"),
        "snapshots",
        "*",
    )
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise RuntimeError(f"no local snapshot for {model_id}")
    return hits[0]


def validate_prompt_contract(demos, queries, eos):
    for index, demo in enumerate(demos):
        ids = demo["input_ids"]
        assert ids[0] == SPEECH_START, f"demo {index}: wrong start"
        assert ids.count(SPEECH_START) == ids.count(SPEECH_END) == 1
        assert ids.count(TRANSCRIPT_START) == 1
        assert ids[-1] == eos, f"demo {index}: does not end in trained EOS {eos}"
        assert max(ids) < EXPANDED_VOCAB
    for row in queries:
        ids = row["input_ids"]
        assert ids[0] == SPEECH_START
        assert ids[-2:] == [SPEECH_END, TRANSCRIPT_START]
        assert eos not in ids
        assert max(ids) < EXPANDED_VOCAB
    return {
        "demo_count": len(demos),
        "demo_tokens": sum(len(d["input_ids"]) for d in demos),
        "eos_id": eos,
        "query_contract": "ends at transcript_start; contains no reference text",
    }


def labelled_example(ids, tok, style):
    """Render the trained unit sequence with optional explicit text role labels.

    The learned speech/transcript sentinel tokens are always retained.  The text
    labels test the prompt-format finding from the conventional FLEURS harness
    without changing the acoustic representation or answer boundary.
    """
    if style == "native":
        return ids
    speech_end = ids.index(SPEECH_END)
    transcript_start = ids.index(TRANSCRIPT_START)
    if speech_end >= transcript_start:
        raise ValueError("speech end must precede transcript start")
    if style == "qa":
        input_label, output_label = "Q: ", "\nA: "
    elif style == "language":
        input_label, output_label = "Irish speech: ", "\nIrish transcript: "
    else:
        raise ValueError(f"unknown prompt style: {style}")
    return (
        tok.encode(input_label, add_special_tokens=False)
        + ids[: speech_end + 1]
        + tok.encode(output_label, add_special_tokens=False)
        + ids[transcript_start:]
    )


def generate(thinker, tok, prompt_ids, eos, max_new):
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        raw = thinker.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=max_new,
            eos_token_id=eos,
            pad_token_id=eos,
            use_cache=True,
        )[0, prompt.shape[1]:].tolist()
    eos_emitted = eos in raw
    answer_ids = raw[: raw.index(eos)] if eos_emitted else raw
    unknown = [token for token in answer_ids if token >= len(tok)]
    text_ids = [token for token in answer_ids if token < len(tok)]
    text = tok.decode(text_ids, skip_special_tokens=True).strip()
    return {
        "hypothesis": text,
        "raw_output_ids": raw,
        "answer_ids": answer_ids,
        "eos_emitted": eos_emitted,
        "stop_reason": "eos" if eos_emitted else "max_new_tokens",
        "unknown_or_unit_ids": unknown,
        "unique_token_ratio": len(set(answer_ids)) / max(len(answer_ids), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--prompt-style",
        choices=["native", "qa", "language"],
        default="native",
        help="retain the trained sentinels, optionally adding explicit role labels",
    )
    args = ap.parse_args()

    model, cfg = load_for_eval(args.checkpoint, device="cuda", dtype=torch.bfloat16)
    thinker = model.thinker
    thinker.config.use_cache = True
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot(cfg.model.base_model_id))
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    demos = load_rows(args.demos)
    rows = load_rows(args.data)[: args.n]
    contract = validate_prompt_contract(demos, rows, eos)
    demo_ids = [
        token
        for demo in demos
        for token in labelled_example(demo["input_ids"], tok, args.prompt_style)
    ]

    conditions = {}
    for condition in ("zero_shot", "three_shot", "three_shot_mismatch"):
        outputs = []
        word_edits = word_count = char_edits = char_count = 0
        for index, row in enumerate(rows):
            query = (
                rows[(index + 1) % len(rows)]["input_ids"]
                if condition == "three_shot_mismatch"
                else row["input_ids"]
            )
            query = labelled_example(query, tok, args.prompt_style)
            prefix = [] if condition == "zero_shot" else demo_ids
            result = generate(
                thinker, tok, prefix + query, eos, args.max_new_tokens
            )
            we, wc = wer(row["reference"], result["hypothesis"])
            ce, cc = cer(row["reference"], result["hypothesis"])
            result.update(
                id=row["id"],
                reference=row["reference"],
                word_edits=we,
                reference_words=wc,
                char_edits=ce,
                reference_chars=cc,
            )
            outputs.append(result)
            word_edits += we
            word_count += wc
            char_edits += ce
            char_count += cc
            print(
                f"[{condition} {index + 1}/{len(rows)}] id={row['id']} "
                f"stop={result['stop_reason']} edits={we}/{wc}",
                flush=True,
            )
        conditions[condition] = {
            "wer": 100 * word_edits / max(word_count, 1),
            "cer": 100 * char_edits / max(char_count, 1),
            "eos_rate": sum(row["eos_emitted"] for row in outputs) / len(outputs),
            "mean_unique_token_ratio": sum(
                row["unique_token_ratio"] for row in outputs
            ) / len(outputs),
            "rows": outputs,
        }

    correct = conditions["three_shot"]["rows"]
    mismatch = conditions["three_shot_mismatch"]["rows"]
    identical = sum(
        a["answer_ids"] == b["answer_ids"] for a, b in zip(correct, mismatch)
    )
    report = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "n": len(rows),
        "prompt_style": args.prompt_style,
        "prompt_contract": contract,
        "conditions": conditions,
        "audio_control": {
            "identical_output_rate": identical / len(rows),
            "wer_delta_mismatch_minus_correct": (
                conditions["three_shot_mismatch"]["wer"]
                - conditions["three_shot"]["wer"]
            ),
        },
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    for name, result in conditions.items():
        print(
            f"{name}: WER={result['wer']:.2f} CER={result['cer']:.2f} "
            f"EOS={result['eos_rate']:.0%} "
            f"unique={result['mean_unique_token_ratio']:.3f}",
            flush=True,
        )
    print(json.dumps(report["audio_control"], indent=2), flush=True)


if __name__ == "__main__":
    main()
