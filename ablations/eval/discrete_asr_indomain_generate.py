#!/usr/bin/env python3
"""Zero-shot native-format generation on the held-out rows retrieval was scored on.

Every previous generation evaluation of the discrete-ASR checkpoints changed
three things at once relative to training: FLEURS audio (median 15.2 s against a
3.8 s training median), nine one-second in-context demonstrations, and English
role labels.  None of those occurred in training, where each example is exactly

    <|speech_start|> units <|speech_end|> <|transcript_start|> text <|im_end|>

for a single utterance.  This script decodes that format and nothing else, on the
same validation rows of the same Parquet that `discrete_asr_retrieval.py` ranks,
selected identically (sorted by filename, first N).  The retrieval and generation
numbers therefore describe one setting instead of two disjoint ones.

Three conditions run per model state:

    correct     the row's own speech prefix
    shuffled    another row's speech prefix, same reference (audio-dependence control)
    unconditional  the prefix reduced to the bare sentinels, no units at all

`shuffled` and `unconditional` bound how much of any apparent success is the Irish
language model rather than the acoustic mapping.  Scores are computed on the raw
decode; `truncate=True` variants are reported alongside as secondary diagnostics
only, never in place of the plain score.

Results are broken out by prefix-length bucket because the training duration
distribution is heavily short-skewed and an aggregate hides that.
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, "..", "train"))
sys.path.insert(0, TRAIN)

from audit_discrete_asr_prompt import (  # noqa: E402
    SPEECH_END,
    SPEECH_START,
    TRANSCRIPT_START,
    generate,
    snapshot,
)
from asr_baseline import norm, wer  # noqa: E402
from fleurs_eval import cer  # noqa: E402


def cut_to_reference(ref, hyp, unit):
    """Trim the hypothesis to the reference length, as a secondary diagnostic.

    asr_baseline.wer grew a truncate= flag locally, but the deployed module here
    still takes (ref, hyp) only, and it is imported by other agents' running
    jobs -- so the trimming is done here rather than by editing shared code.
    norm() is idempotent, so re-normalising inside wer/cer is a no-op.
    """
    if unit == "word":
        reference, hypothesis = norm(ref).split(), norm(hyp).split()
        return " ".join(hypothesis[: len(reference)])
    return norm(hyp)[: len(norm(ref))]
from qomhra.model import get_model  # noqa: E402

BUCKETS = ((0, 200), (200, 400), (400, 700), (700, 1 << 30))


def bucket_name(prefix_len):
    for low, high in BUCKETS:
        if low <= prefix_len < high:
            return f"{low}-{high}" if high < (1 << 30) else f"{low}+"
    raise AssertionError(prefix_len)


def heldout_examples(path, n, tok):
    """Same rows, same order, as discrete_asr_retrieval.heldout_examples.

    Scan scalar metadata first, then materialise token lists only for the chosen
    rows. This avoids corpus-wide Python and Arrow representations of the two
    1024-token list columns.
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    validation_filenames = []
    for batch in parquet.iter_batches(
        columns=["split", "filename"], batch_size=65_536
    ):
        splits, filenames = batch.column(0), batch.column(1)
        for row_index in range(batch.num_rows):
            if splits[row_index].as_py() == "validation":
                validation_filenames.append(filenames[row_index].as_py())

    heldout_total = len(validation_filenames)
    if heldout_total < n:
        raise RuntimeError(f"only {heldout_total} validation rows, requested {n}")
    wanted = set(sorted(validation_filenames)[:n])

    rows = []
    for batch in parquet.iter_batches(
        columns=["input_ids", "labels", "split", "filename"], batch_size=512
    ):
        ids_col, labels_col, split_col, filename_col = (
            batch.column(index) for index in range(4)
        )
        for row_index in range(batch.num_rows):
            if split_col[row_index].as_py() != "validation":
                continue
            filename = filename_col[row_index].as_py()
            if filename not in wanted:
                continue
            labels = np.asarray(labels_col[row_index].as_py(), dtype=np.int64)
            target_positions = np.flatnonzero(labels != -100)
            if not len(target_positions):
                raise RuntimeError(
                    f"held-out row {filename} has no transcript labels"
                )
            start = int(target_positions[0])
            ids = list(map(int, ids_col[row_index].as_py()))
            prefix = ids[:start]
            if prefix[0] != SPEECH_START or prefix[-1] != TRANSCRIPT_START:
                raise RuntimeError(f"{filename}: prefix is not the trained contract")
            reference = tok.decode(
                [t for t in ids[start:] if t < len(tok)],
                skip_special_tokens=True,
            ).strip()
            rows.append(
                {
                    "filename": filename,
                    "prefix": prefix,
                    "reference": reference,
                    "prefix_tokens": len(prefix),
                    "reference_words": len(reference.split()),
                }
            )
    rows.sort(key=lambda row: row["filename"])
    if len(rows) != n:
        raise RuntimeError(f"selected {n} validation rows but loaded {len(rows)}")
    return rows, heldout_total


def bare_prefix(prefix):
    """The trained sentinels with every unit removed."""
    speech_end = prefix.index(SPEECH_END)
    return [prefix[0]] + prefix[speech_end:]


def aggregate(outputs):
    def rate(key):
        return sum(row[key] for row in outputs) / max(len(outputs), 1)

    totals = {key: sum(row[key] for row in outputs) for key in
              ("word_edits", "reference_words", "char_edits", "reference_chars",
               "word_edits_trunc", "char_edits_trunc")}
    return {
        "n": len(outputs),
        "wer": 100 * totals["word_edits"] / max(totals["reference_words"], 1),
        "cer": 100 * totals["char_edits"] / max(totals["reference_chars"], 1),
        "wer_truncated_diagnostic": (
            100 * totals["word_edits_trunc"] / max(totals["reference_words"], 1)
        ),
        "cer_truncated_diagnostic": (
            100 * totals["char_edits_trunc"] / max(totals["reference_chars"], 1)
        ),
        "eos_rate": rate("eos_emitted"),
        "max_token_cap_rate": sum(
            row["stop_reason"] == "max_new_tokens" for row in outputs
        ) / max(len(outputs), 1),
        "mean_hypothesis_words": rate("hypothesis_words"),
        "mean_reference_words": rate("reference_words"),
        "mean_unique_token_ratio": rate("unique_token_ratio"),
        "empty_hypothesis_rate": sum(
            not row["hypothesis"].strip() for row in outputs
        ) / max(len(outputs), 1),
    }


def score_state(model, tok, eos, rows, max_new, state, condition):
    thinker = model.thinker
    thinker.config.use_cache = True
    outputs = []
    for index, row in enumerate(rows):
        if condition == "correct":
            prefix = row["prefix"]
        elif condition == "shuffled":
            prefix = rows[(index + 1) % len(rows)]["prefix"]
        elif condition == "unconditional":
            prefix = bare_prefix(row["prefix"])
        else:
            raise ValueError(condition)
        result = generate(thinker, tok, prefix, eos, max_new)
        we, wc = wer(row["reference"], result["hypothesis"])
        ce, cc = cer(row["reference"], result["hypothesis"])
        wet, _ = wer(
            row["reference"], cut_to_reference(row["reference"], result["hypothesis"], "word")
        )
        cet, _ = cer(
            row["reference"], cut_to_reference(row["reference"], result["hypothesis"], "char")
        )
        result.update(
            filename=row["filename"],
            reference=row["reference"],
            prompt_tokens=len(prefix),
            prefix_tokens=row["prefix_tokens"],
            length_bucket=bucket_name(row["prefix_tokens"]),
            hypothesis_words=len(result["hypothesis"].split()),
            word_edits=we,
            reference_words=wc,
            char_edits=ce,
            reference_chars=cc,
            word_edits_trunc=wet,
            char_edits_trunc=cet,
        )
        outputs.append(result)
        print(
            f"[{state}/{condition} {index + 1}/{len(rows)}] {row['filename']} "
            f"prefix={len(prefix)} stop={result['stop_reason']} "
            f"WER-edits={we}/{wc}\n"
            f"  REF: {row['reference']}\n"
            f"  HYP: {result['hypothesis']}",
            flush=True,
        )
    report = aggregate(outputs)
    report["by_length_bucket"] = {
        name: aggregate([r for r in outputs if r["length_bucket"] == name])
        for name in sorted({r["length_bucket"] for r in outputs})
    }
    report["rows"] = outputs
    return report


def run_conditions(model, tok, eos, rows, max_new, state, conditions):
    return {
        condition: score_state(model, tok, eos, rows, max_new, state, condition)
        for condition in conditions
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--comparison-checkpoint",
        help=(
            "optional second compatible checkpoint to load after --checkpoint; "
            "used for before/after domain-adaptation comparisons"
        ),
    )
    ap.add_argument("--comparison-label", default="comparison")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--conditions",
        default="correct,shuffled,unconditional",
        help="comma-separated subset of correct,shuffled,unconditional",
    )
    ap.add_argument(
        "--skip-base",
        action="store_true",
        help="decode only the trained checkpoint, not the expanded initial state",
    )
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    cfg = OmegaConf.load(os.path.join(args.checkpoint, "config.yaml"))
    # Without this the extension checkpoint's embedded init_from pointer is
    # followed and the "initial" state is silently the previous checkpoint
    # rather than the expanded, untrained base.
    init_from = cfg.model.init_from
    cfg.model.init_from = None
    torch.manual_seed(int(cfg.seed))
    model, _ = get_model(cfg)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot(cfg.model.base_model_id))
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    rows, heldout_total = heldout_examples(args.data, args.n, tok)

    results = {}
    if not args.skip_base:
        # Expanded architecture with no ASR weights of any kind loaded.
        results["expanded_base"] = run_conditions(
            model, tok, eos, rows, args.max_new_tokens, "expanded_base", conditions
        )

    state = torch.load(
        os.path.join(args.checkpoint, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"trained checkpoint mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    del state
    gc.collect()
    torch.cuda.empty_cache()
    results["trained"] = run_conditions(
        model, tok, eos, rows, args.max_new_tokens, "trained", conditions
    )
    if args.comparison_checkpoint:
        state = torch.load(
            os.path.join(args.comparison_checkpoint, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"comparison checkpoint mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected"
            )
        del state
        gc.collect()
        torch.cuda.empty_cache()
        results[args.comparison_label] = run_conditions(
            model,
            tok,
            eos,
            rows,
            args.max_new_tokens,
            args.comparison_label,
            conditions,
        )

    report = {
        "label": args.label,
        "checkpoint": os.path.abspath(args.checkpoint),
        "comparison_checkpoint": (
            os.path.abspath(args.comparison_checkpoint)
            if args.comparison_checkpoint
            else None
        ),
        "data": os.path.abspath(args.data),
        "heldout_total": heldout_total,
        "n": len(rows),
        "expanded_base_definition": (
            "expanded architecture rebuilt from this run's config and seed with "
            "init_from cleared, so no ASR weights are loaded; the config's own "
            f"init_from pointer ({init_from}) is deliberately not followed"
        ),
        "max_new_tokens": args.max_new_tokens,
        "prompt": (
            "native trained contract, zero-shot: no demonstrations, no text role "
            "labels, no chat template"
        ),
        "selection": "first N validation rows by filename; identical to discrete_asr_retrieval.py",
        "scoring": "raw decode; truncated variants are secondary diagnostics only",
        "conditions": {
            "correct": "the row's own speech prefix",
            "shuffled": "the next row's speech prefix against this reference",
            "unconditional": "sentinels only, all units removed",
        },
        "prefix_token_percentiles": {
            str(p): float(np.percentile([r["prefix_tokens"] for r in rows], p))
            for p in (5, 25, 50, 75, 95)
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    for state_name, conds in results.items():
        for condition, result in conds.items():
            print(
                f"SUMMARY {state_name}/{condition}: WER={result['wer']:.2f} "
                f"CER={result['cer']:.2f} EOS={result['eos_rate']:.0%} "
                f"cap={result['max_token_cap_rate']:.0%} "
                f"words={result['mean_hypothesis_words']:.1f} "
                f"(ref {result['mean_reference_words']:.1f})",
                flush=True,
            )
    if "correct" in conditions and "shuffled" in conditions:
        trained = results["trained"]
        print(
            "AUDIO DEPENDENCE (trained): "
            f"CER correct={trained['correct']['cer']:.2f} "
            f"shuffled={trained['shuffled']['cer']:.2f} "
            f"delta={trained['shuffled']['cer'] - trained['correct']['cer']:+.2f}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
