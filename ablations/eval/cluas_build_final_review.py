#!/usr/bin/env python3
"""Build cluas_raw_outputs_final.csv / cluas_numbers_final.csv from mult2x units_rows.json
plus the Sonnet verdicts already written by cluas_sonnet_judge.py.

Standalone equivalent of cluas_prepare_review.py's CSV-writing path, without the
judge-cache dependency: every row here is freshly marked by cluas_sonnet_judge.py
(cluas_sonnet_scored_final/*.json), so there is no cache to fall back to.
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import pyarrow.parquet as pq

CONDITIONS = ("just_audio", "no_context", "just_transcript")


def selected_rows(rows):
    best = {}
    for row in rows:
        key = (row["condition"], row["question_id"])
        score = (
            row["mean_logprob"]
            if row["hypothesis"].strip() and row["mean_logprob"] is not None
            else float("-inf")
        )
        if key not in best or score > best[key][0]:
            best[key] = (score, row)
    return [pair[1] for pair in best.values()]


def load_question_meta(parquet_path):
    table = pq.read_table(parquet_path, columns=["questions_json"])
    meta = {}
    for value in table.column("questions_json").to_pylist():
        for question in json.loads(value):
            meta[question["question_id"]] = question
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--sonnet-scored-dir", required=True)
    ap.add_argument("--numbers-out", required=True)
    ap.add_argument("--raw-out", required=True)
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(glob.glob(pattern))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("no CLUAS row JSON inputs matched")

    question_meta = load_question_meta(args.data_parquet)
    sonnet_verdicts = {}
    for path in glob.glob(os.path.join(args.sonnet_scored_dir, "*.json")):
        sonnet_verdicts[os.path.splitext(os.path.basename(path))[0]] = json.load(
            open(path, encoding="utf-8"))

    all_raw = []
    grouped = defaultdict(list)
    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        model = payload["run"]["label"]
        for row in selected_rows(payload["rows"]):
            meta = question_meta.get(row["question_id"], {})
            sonnet = (
                sonnet_verdicts.get(model, {})
                .get(row["question_id"], {})
                .get(row["condition"])
            )
            empty_answer = not row["hypothesis"].strip()
            official_marks = (
                sonnet.get("marks") if sonnet
                else 0 if empty_answer
                else None
            )
            official_reason = (
                sonnet.get("reason", "") if sonnet
                else "no answer given" if empty_answer
                else None
            )
            item = {
                "model": model,
                "condition": row["condition"],
                "question_id": row["question_id"],
                "question": row["question"],
                "official_marking_scheme": meta.get("rubric_ga"),
                "marks_available": meta.get("marks_available"),
                "blanks_required": meta.get("blanks_required"),
                "hypothesis": row["hypothesis"],
                "official_marks": official_marks,
                "official_reason": official_reason,
                "judge_status": (
                    "sonnet" if sonnet
                    else "automatic_zero_empty" if empty_answer
                    else "pending"
                ),
                "stop_reason": row["stop_reason"],
                "emitted_stop_token": row.get("emitted_stop_token"),
                "generation_budget": row.get("generation_budget"),
                "generated_tokens": row["generated_tokens"],
                "selected_window": row["window"],
                "n_windows": row["n_windows"],
                "mismatch_hypothesis": row.get("mismatch_hypothesis"),
                "mismatch_identical": row.get("mismatch_identical"),
                "raw_output_ids": json.dumps(row["raw_output_ids"]),
            }
            all_raw.append(item)
            grouped[(model, row["condition"])].append(item)

    summaries = []
    for (model, condition), rows in sorted(grouped.items()):
        stop_counts = Counter(row["stop_reason"] for row in rows)
        judged = [row for row in rows if row["official_marks"] is not None]
        summaries.append({
            "model": model,
            "condition": condition,
            "questions": len(rows),
            "official_marks": (
                sum(int(row["official_marks"]) for row in judged)
                if judged else None
            ),
            "marks_available": sum(
                int(row["marks_available"]) for row in rows
                if row["marks_available"] is not None
            ),
            "official_judged_questions": len(judged),
            "official_pending_questions": len(rows) - len(judged),
            "nonempty_n": sum(bool(row["hypothesis"].strip()) for row in rows),
            "natural_eos_n": stop_counts["eos"],
            "prompt_boundary_stop_n": stop_counts["stop_string"],
            "cap_n": stop_counts["max_new_tokens"],
            "mismatch_different_n": sum(
                row["mismatch_identical"] is False for row in rows
            ),
            "mismatch_compared_n": sum(
                row["mismatch_identical"] is not None for row in rows
            ),
        })

    summary_lookup = {(row["model"], row["condition"]): row for row in summaries}
    for row in summaries:
        available = row["marks_available"]
        row["official_percent"] = (
            100.0 * row["official_marks"] / available
            if row["official_marks"] is not None and available else None
        )
        audio = summary_lookup.get((row["model"], "just_audio"), {})
        blind = summary_lookup.get((row["model"], "no_context"), {})
        row["listening_gain_marks"] = (
            audio.get("official_marks") - blind.get("official_marks")
            if audio.get("official_marks") is not None
            and blind.get("official_marks") is not None
            else None
        )

    for path, rows in [(args.numbers_out, summaries), (args.raw_out, all_raw)]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
