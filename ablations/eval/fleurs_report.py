#!/usr/bin/env python3
"""Turn the FLEURS result JSONs into the paper's spreadsheet and summary tables.

Reads every per-condition result file produced by fleurs_eval.py and writes:

  fleurs_long.csv      one row per model x checkpoint x condition x sentence — the
                       error-analysis file (equivalent to cluas_long.csv)
  fleurs_summary.csv   one row per model x checkpoint x condition, corpus scores
  fleurs_tables.md     the summary tables, one table per question

Table rules followed here (agreed with the user):
  * columns run in derivation order and END on the number that answers the question, so the
    reader finishes the row on the payoff rather than a distractor;
  * rows are sorted by that final column, so the ranking is free;
  * one table shows one thing — the other columns are its workings;
  * WER-based and chrF-based conditions never share a table, and no gap is ever computed
    across the two metric families.

  python fleurs_report.py --results output --out-dir ../../Qomhra2-Paper/evals
"""
import argparse
import csv
import glob
import json
import os

# Display order and plain-English names. Nothing here decides a metric — the metric comes
# from the result file, which got it from the CONDITIONS table in fleurs_eval.py.
CONDITION_LABELS = [
    ("text_ga2en", "Irish text -> English text"),
    ("text_en2ga", "English text -> Irish text"),
    ("asr_ga", "Irish speech -> Irish text"),
    ("asr_en", "English speech -> English text"),
    ("st_ga2en", "Irish speech -> English text"),
    ("st_en2ga", "English speech -> Irish text"),
    ("copy_ga", "Irish text -> Irish text (format control)"),
]
MODEL_ORDER = ["base_base", "text_pct10", "text_final", "speech_pct10", "speech_final",
               "both_pct10", "both_final", "aligned_pct10", "aligned_final"]


def load_results(results_dir, min_n):
    """{label: {condition: {"summary": ..., "rows": [...]}}} over every full-run JSON.

    Anything scored on fewer than min_n sentences is skipped. Smoke probes and targeted
    checks share the naming scheme of full runs, so this is the guard that stops a ten-row
    or three-row file from silently becoming a reported number.
    """
    out, skipped = {}, []
    for path in sorted(glob.glob(os.path.join(results_dir, "fleurs_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        run = data["run"]
        for name, payload in data["conditions"].items():
            if payload["summary"]["n"] < min_n:
                skipped.append((os.path.basename(path), name, payload["summary"]["n"]))
                continue
            out.setdefault(run["label"], {"run": run, "conditions": {}})
            out[run["label"]]["conditions"][name] = payload
    if skipped:
        print(f"[load] skipped {len(skipped)} partial result(s) (< {min_n} sentences): "
              + ", ".join(f"{f}:{c}(n={n})" for f, c, n in skipped[:6])
              + (" ..." if len(skipped) > 6 else ""))
    return out


def model_sort_key(label):
    return (MODEL_ORDER.index(label) if label in MODEL_ORDER else len(MODEL_ORDER), label)


def write_long(results, path):
    # hypothesis is what the tokeniser hands back with special tokens deleted;
    # hypothesis_bounded is the same decode cut at the first special token, which is the text
    # wer_trunc/cer_trunc are measured on. Both are kept so the correction can be inspected.
    fields = ["model", "condition", "metric", "sentence_id", "reference", "hypothesis",
              "hypothesis_bounded", "score", "wer", "cer", "wer_trunc", "cer_trunc", "chrfpp",
              "generated_tokens", "stop_reason", "ga_gender", "en_gender", "ga_duration_s",
              "en_duration_s", "raw_generation"]
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for label in sorted(results, key=model_sort_key):
            for name, _ in CONDITION_LABELS:
                payload = results[label]["conditions"].get(name)
                if not payload:
                    continue
                metric = payload["summary"]["metric"]
                for row in payload["rows"]:
                    writer.writerow(dict(
                        row, model=label, metric=metric,
                        sentence_id=row["id"], reference=row["ref"], hypothesis=row["hyp"],
                        hypothesis_bounded=row.get("hyp_bounded"),
                        raw_generation=row["raw"],
                        score=(row.get("wer_trunc") if metric == "wer"
                               else row.get("chrfpp"))))
                    n += 1
    print(f"wrote {path} ({n} rows)")


def write_summary(results, path):
    fields = ["model", "condition", "metric", "n", "wer", "cer", "wer_trunc", "cer_trunc",
              "chrfpp", "bleu", "cap_hit_pct", "mean_generated_tokens"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for label in sorted(results, key=model_sort_key):
            for name, _ in CONDITION_LABELS:
                payload = results[label]["conditions"].get(name)
                if not payload:
                    continue
                s = payload["summary"]
                writer.writerow(dict(s, model=label, condition=name,
                                     cap_hit_pct=100.0 * s["cap_hit_rate"]))
    print(f"wrote {path}")


def score_of(results, label, condition, key):
    payload = results[label]["conditions"].get(condition)
    return payload["summary"].get(key) if payload else None


def fmt(value, nd=2):
    return "—" if value is None else f"{value:.{nd}f}"


def table(title, note, header, rows, sort_col, descending=True):
    """One question, one table. Rows sort on the final (payoff) column, which is bolded."""
    ranked = sorted((r for r in rows if r[sort_col] is not None),
                    key=lambda r: r[sort_col], reverse=descending)
    ranked += [r for r in rows if r[sort_col] is None]
    lines = [f"### {title}", "", note, "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in ranked:
        cells = [r["model"]] + [fmt(r[k]) for k in r["cols"][:-1]]
        cells.append(f"**{fmt(r[r['cols'][-1]])}**")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_tables(results):
    labels = sorted(results, key=model_sort_key)
    out = ["# FLEURS results — Irish/English, text vs speech",
           "",
           "Scores only. Transcription is word error rate (WER) and character error rate "
           "(CER), where **lower is better**. Both are measured on a bounded hypothesis: "
           "the raw decode is cut at the first special token, then cut to the reference "
           "length before aligning, which bounds them at 100%. These checkpoints often fail "
           "to emit the end-of-turn token the prompt format asked for and run to the "
           "decoding limit, and counting the insertions that follow ranks a model that stays "
           "silent above one that transcribes. The unbounded figures stay in "
           "`fleurs_summary.csv` as `wer` and `cer`. "
           "Translation is chrF++ and BLEU, where **higher is better**. "
           "The two families are never compared or subtracted across.",
           ""]

    # --- Modality gap, ga->en: the same sentences by text vs by speech, one metric. ---
    rows = []
    for label in labels:
        text = score_of(results, label, "text_ga2en", "chrfpp")
        speech = score_of(results, label, "st_ga2en", "chrfpp")
        gap = None if text is None or speech is None else text - speech
        rows.append({"model": label, "text": text, "speech": speech, "gap": gap,
                     "cols": ["text", "speech", "gap"]})
    out.append(table(
        "Modality gap, Irish → English (chrF++)",
        "What it costs to route the same content through audio, with translation ability "
        "held constant. The text column is also the ceiling: the speech column cannot beat "
        "it. Gap = text − speech; larger means the audio pathway loses more.",
        ["model", "Irish text → English", "Irish speech → English", "gap"], rows, "gap"))

    # --- Modality gap, en->ga. ---
    rows = []
    for label in labels:
        text = score_of(results, label, "text_en2ga", "chrfpp")
        speech = score_of(results, label, "st_en2ga", "chrfpp")
        gap = None if text is None or speech is None else text - speech
        rows.append({"model": label, "text": text, "speech": speech, "gap": gap,
                     "cols": ["text", "speech", "gap"]})
    out.append(table(
        "Modality gap, English → Irish (chrF++)",
        "The reverse direction, same construction.",
        ["model", "English text → Irish", "English speech → Irish", "gap"], rows, "gap"))

    # --- Language gap in the audio pathway: English control vs Irish, WER only. ---
    rows = []
    for label in labels:
        english = score_of(results, label, "asr_en", "wer_trunc")
        irish = score_of(results, label, "asr_ga", "wer_trunc")
        gap = None if english is None or irish is None else irish - english
        rows.append({"model": label, "english": english, "irish": irish, "gap": gap,
                     "cols": ["english", "irish", "gap"]})
    out.append(table(
        "Language gap in the audio pathway (WER %)",
        "Transcription only, so modality is held constant and the difference is the "
        "Irish-specific deficit. English is the control. Gap = Irish − English; larger "
        "means Irish costs the audio pathway more.",
        ["model", "English speech → English (WER)", "Irish speech → Irish (WER)", "gap"],
        rows, "gap"))

    # --- All translation conditions side by side (same metric, so this is legitimate). ---
    lines = ["### Translation conditions (chrF++, higher is better)", "",
             "All four translation cells on the same sentences and the same metric.", "",
             "| model | Irish text → English | English text → Irish | Irish speech → English "
             "| English speech → Irish |", "|---|---|---|---|---|"]
    for label in labels:
        cells = [fmt(score_of(results, label, c, "chrfpp"))
                 for c in ("text_ga2en", "text_en2ga", "st_ga2en", "st_en2ga")]
        lines.append("| " + " | ".join([label] + cells) + " |")
    out.append("\n".join(lines) + "\n")

    # --- Transcription and the format control (WER/CER, higher is worse). ---
    lines = ["### Transcription and format control (WER % / CER %, lower is better)", "",
             "The copy control asks the model to repeat Irish text back unchanged. It "
             "separates a model that cannot do the task from one that cannot follow any "
             "instruction.", "",
             "| model | Irish speech → Irish | English speech → English | Irish text → Irish "
             "(copy) |", "|---|---|---|---|"]
    for label in labels:
        cells = []
        for c in ("asr_ga", "asr_en", "copy_ga"):
            cells.append(f"{fmt(score_of(results, label, c, 'wer_trunc'))} / "
                         f"{fmt(score_of(results, label, c, 'cer_trunc'))}")
        lines.append("| " + " | ".join([label] + cells) + " |")
    out.append("\n".join(lines) + "\n")

    # --- Diagnostics: how often decoding ran to the cap instead of stopping. ---
    lines = ["### Decoding diagnostics — share of answers that ran to the token limit (%)",
             "",
             "A high figure means the model did not stop on its own. Inspection of the base "
             "run showed these are repetition loops, not sentences cut off mid-way, so the "
             "limit is not suppressing real content.", "",
             "| model | " + " | ".join(n for n, _ in CONDITION_LABELS) + " |",
             "|" + "---|" * (len(CONDITION_LABELS) + 1)]
    for label in labels:
        cells = []
        for name, _ in CONDITION_LABELS:
            rate = score_of(results, label, name, "cap_hit_rate")
            cells.append("—" if rate is None else f"{100 * rate:.0f}")
        lines.append("| " + " | ".join([label] + cells) + " |")
    out.append("\n".join(lines) + "\n")

    # --- Completeness: the design depends on identical row counts across conditions. ---
    lines = ["### Row counts (the parallel design depends on these being identical)", "",
             "| model | " + " | ".join(n for n, _ in CONDITION_LABELS) + " |",
             "|" + "---|" * (len(CONDITION_LABELS) + 1)]
    for label in labels:
        cells = [str(score_of(results, label, n, "n") or "—") for n, _ in CONDITION_LABELS]
        lines.append("| " + " | ".join([label] + cells) + " |")
    out.append("\n".join(lines) + "\n")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="output", help="directory of fleurs_*.json")
    ap.add_argument("--out-dir", required=True, help="e.g. Qomhra2-Paper/evals")
    ap.add_argument("--min-n", type=int, default=340,
                    help="ignore results scored on fewer sentences than this")
    args = ap.parse_args()

    results = load_results(args.results, args.min_n)
    if not results:
        raise SystemExit(f"no result JSONs in {args.results}")
    print(f"[load] {len(results)} models: {sorted(results, key=model_sort_key)}")
    os.makedirs(args.out_dir, exist_ok=True)
    write_long(results, os.path.join(args.out_dir, "fleurs_long.csv"))
    write_summary(results, os.path.join(args.out_dir, "fleurs_summary.csv"))
    tables_path = os.path.join(args.out_dir, "fleurs_tables.md")
    with open(tables_path, "w", encoding="utf-8") as f:
        f.write(build_tables(results))
    print(f"wrote {tables_path}")


if __name__ == "__main__":
    main()
