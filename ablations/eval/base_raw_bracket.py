#!/usr/bin/env python3
"""Bracket table: base (own contract) / base_raw (matched prompt) / ab1-ab4.

Why a bracket and not a single base row. The four ablations were scored on a raw
document prompt with no instruction and no chat roles; base was scored on its own
chat template with an explicit English instruction. base_raw removes that difference
by running the same model on the ablations' prompt, with the waveform occupying the
slot the unit block occupies.

It does not remove the other difference and nothing can: base reads waveforms through
its audio tower, the ablations read discrete units, and the discrete checkpoints have
no audio tower to run (model.py sets it to None under discrete_only) while base has
no embedding for the unit ids. So base and base_raw bound the same model under two
prompts, and the ablations are read against that interval:

  - an ablation above `base`      -> the result does not depend on the prompt
  - an ablation above `base_raw` only -> the result is prompt-conditional, say so

`sentinel_echo` is reported next to every base_raw score. Base sees the sentinels as
ordinary subwords ("<", "|", "speech", "_start", ...) and sometimes writes them back
out; those characters are scored like any other output because the contract forbids
post-processing, so the count has to travel with the number it distorts.

Usage:
  python base_raw_bracket.py \
    --fleurs-final ablations/eval/output/review_csv/fleurs_numbers_final.csv \
    --base-raw-glob 'ablations/eval/output/fleurs_grid_p10_base_raw_*_mult2.5.json' \
    --out ablations/eval/output/review_visualisations/base_raw_bracket.md
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

MODEL_ORDER = ["base", "base_raw", "ab1", "ab2", "ab3", "ab4"]
MODEL_LABEL = {
    "base": "base · chat (own contract)",
    "base_raw": "base_raw · matched prompt",
    "ab1": "AB1 · text",
    "ab2": "AB2 · speech",
    "ab3": "AB3 · mixed",
    "ab4": "AB4 · aligned",
}
# (condition, metric column in fleurs_numbers_final.csv, lower_is_better)
FLEURS_COLUMNS = [
    ("asr_ga", "asr_ga_wer", True),
    ("asr_en", "asr_en_wer", True),
    ("text_ga2en", "text_ga2en_chrfpp", False),
    ("text_en2ga", "text_en2ga_chrfpp", False),
    ("st_ga2en", "st_ga2en_chrfpp", False),
    ("st_en2ga", "st_en2ga_chrfpp", False),
]


def load_base_raw(pattern):
    """One row per condition from the per-condition base_raw result JSONs."""
    scores, echoes, eos_counts, n_rows = {}, {}, {}, {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for name, block in payload["conditions"].items():
            summary, rows = block["summary"], block["rows"]
            value = summary.get("wer", summary.get("chrfpp"))
            if value is None:
                continue
            scores[name] = float(value)
            echoes[name] = sum(r.get("sentinel_text_in_answer", 0) for r in rows)
            eos_counts[name] = sum(1 for r in rows if r["stop_reason"] == "eos")
            n_rows[name] = len(rows)
    return scores, echoes, eos_counts, n_rows


def build(fleurs_final, base_raw_glob):
    final = pd.read_csv(fleurs_final).set_index("model")
    scores, echoes, eos_counts, n_rows = load_base_raw(base_raw_glob)

    header = ["model"] + [name for name, _, _ in FLEURS_COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
    ]
    for model in MODEL_ORDER:
        cells = [MODEL_LABEL[model]]
        for name, column, _ in FLEURS_COLUMNS:
            if model == "base_raw":
                value = scores.get(name)
            elif model in final.index and column in final.columns:
                value = final.loc[model, column]
            else:
                value = None
            cells.append("—" if value is None else f"{float(value):.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("WER for asr_*, lower is better; chrF++ elsewhere, higher is better.")
    lines.append("")
    lines.append("base_raw generation diagnostics (n, natural EOS, echoed sentinels):")
    lines.append("")
    diag = ["| condition | n | natural EOS | sentinel echo |",
            "|---|---:|---:|---:|"]
    for name, _, _ in FLEURS_COLUMNS:
        if name in scores:
            diag.append(
                f"| {name} | {n_rows[name]} | {eos_counts[name]} | {echoes[name]} |"
            )
    lines.extend(diag)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleurs-final", required=True)
    parser.add_argument("--base-raw-glob", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    table = build(args.fleurs_final, args.base_raw_glob)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
