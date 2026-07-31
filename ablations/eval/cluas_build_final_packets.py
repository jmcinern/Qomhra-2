#!/usr/bin/env python3
"""Build consolidated Sonnet-judge packets straight from mult2x units_rows.json.

Standalone equivalent of cluas_prepare_review.py's --consolidated-packet-dir path,
without the judge-cache dependency: these are fresh generations that were never
marked before, so there is nothing to look up in a cache. One packet per model,
matching the shape cluas_sonnet_judge.py expects.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

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
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(glob.glob(pattern))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("no CLUAS row JSON inputs matched")

    question_meta = load_question_meta(args.data_parquet)
    os.makedirs(args.out_dir, exist_ok=True)

    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        model = payload["run"]["label"]
        by_question = defaultdict(dict)
        for row in selected_rows(payload["rows"]):
            by_question[row["question_id"]][row["condition"]] = row["hypothesis"]
        items = []
        for question_id, answers in sorted(by_question.items()):
            meta = question_meta[question_id]
            items.append({
                "question_id": question_id,
                "question": meta["text_ga"],
                "marking_scheme": meta["rubric_ga"],
                "marks_available": int(meta["marks_available"]),
                "blanks_required": int(meta["blanks_required"]),
                "answers": {
                    condition: answers.get(condition, "")
                    for condition in CONDITIONS
                },
            })
        out_path = os.path.join(args.out_dir, f"{model}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(items, handle, ensure_ascii=False, indent=2)
        print(f"wrote {len(items)} questions to {out_path}")


if __name__ == "__main__":
    main()
