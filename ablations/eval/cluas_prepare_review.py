#!/usr/bin/env python3
"""Build CLUAS judge packets plus numeric/raw review CSVs.

Official marks are populated only from the LC-Aural-Bench judge cache. Missing
verdicts remain blank: this script never replaces the official judge with an
overlap heuristic.
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq


CONDITIONS = ("just_audio", "no_context", "just_transcript")


def load_judge_module(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("lc_scoring_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    """Load the official question/rubric fields without materialising audio."""
    table = pq.read_table(parquet_path, columns=["questions_json"])
    meta = {}
    for value in table.column("questions_json").to_pylist():
        for question in json.loads(value):
            meta[question["question_id"]] = question
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--judge-script", required=True)
    ap.add_argument("--packet-dir", required=True)
    ap.add_argument("--data-parquet")
    ap.add_argument("--consolidated-packet-dir")
    ap.add_argument("--sonnet-scored-dir")
    ap.add_argument("--numbers-out", required=True)
    ap.add_argument("--raw-out", required=True)
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(glob.glob(pattern))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("no CLUAS row JSON inputs matched")

    judge = load_judge_module(args.judge_script)
    question_meta = (
        load_question_meta(args.data_parquet) if args.data_parquet else {}
    )
    sonnet_verdicts = {}
    if args.sonnet_scored_dir:
        for path in glob.glob(os.path.join(args.sonnet_scored_dir, "*.json")):
            sonnet_verdicts[os.path.splitext(os.path.basename(path))[0]] = (
                json.load(open(path, encoding="utf-8"))
            )
    os.makedirs(args.packet_dir, exist_ok=True)
    all_raw = []
    grouped = defaultdict(list)
    uncached = []

    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        model = payload["run"]["label"].removesuffix("_p10")
        for row in selected_rows(payload["rows"]):
            year = int(row["question_id"].split("-", 1)[0])
            cache_path = (
                judge.ROOT / "data" / str(year) /
                "cluastuiscint" / "judgments.json"
            )
            cache = (
                json.loads(cache_path.read_text(encoding="utf-8"))
                if cache_path.exists() else {}
            )
            key = judge.cache_key(row["question_id"], row["hypothesis"])
            verdict = cache.get(key)
            empty_answer = not row["hypothesis"].strip()
            sonnet = (
                sonnet_verdicts.get(model, {})
                .get(row["question_id"], {})
                .get(row["condition"])
            )
            official_marks = (
                sonnet.get("marks") if sonnet
                else verdict.get("marks") if verdict
                else 0 if empty_answer
                else None
            )
            official_reason = (
                sonnet.get("reason", "") if sonnet
                else verdict.get("reasoning", verdict.get("reason", ""))
                if verdict
                else "no answer given" if empty_answer
                else None
            )
            meta = question_meta.get(row["question_id"], {})
            item = {
                "model": model,
                "condition": row["condition"],
                "year": year,
                "question_id": row["question_id"],
                "question": row["question"],
                "official_marking_scheme": meta.get("rubric_ga"),
                "marks_available": meta.get("marks_available"),
                "blanks_required": meta.get("blanks_required"),
                "accepted_rubric_answers": " || ".join(
                    row.get("rubric_answer_candidates") or []
                ),
                "hypothesis": row["hypothesis"],
                "official_marks": official_marks,
                "official_reason": official_reason,
                "judge_cached": bool(verdict),
                "judge_status": (
                    "sonnet" if sonnet
                    else "cached" if verdict
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
            if not sonnet and not verdict and not empty_answer:
                uncached.append(item)

    # One sparse answer JSON per model/condition/year, directly consumable by
    # scoring_judge.py. Missing questions are deliberately absent.
    packets = defaultdict(dict)
    for row in all_raw:
        packets[(row["model"], row["condition"], row["year"])][
            row["question_id"]
        ] = row["hypothesis"]
    for (model, condition, year), answers in packets.items():
        name = f"{model}__{condition}__{year}.json"
        with open(os.path.join(args.packet_dir, name), "w",
                  encoding="utf-8") as handle:
            json.dump(answers, handle, ensure_ascii=False, indent=2)

    # This is the packet shape used by the established Claude Sonnet judge:
    # one item per question, with all three conditions graded independently.
    if args.consolidated_packet_dir:
        if not question_meta:
            raise SystemExit(
                "--data-parquet is required with --consolidated-packet-dir"
            )
        os.makedirs(args.consolidated_packet_dir, exist_ok=True)
        by_model_question = defaultdict(dict)
        for row in all_raw:
            by_model_question[(row["model"], row["question_id"])][
                row["condition"]
            ] = row["hypothesis"]
        consolidated = defaultdict(list)
        for (model, question_id), answers in sorted(
            by_model_question.items()
        ):
            meta = question_meta[question_id]
            consolidated[model].append({
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
        for model, items in consolidated.items():
            path = os.path.join(args.consolidated_packet_dir, f"{model}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(items, handle, ensure_ascii=False, indent=2)
        print(
            f"wrote {len(consolidated)} consolidated Sonnet packets to "
            f"{args.consolidated_packet_dir}"
        )

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

    summary_lookup = {
        (row["model"], row["condition"]): row for row in summaries
    }
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
    print(f"wrote {len(packets)} judge packets to {args.packet_dir}")
    cached_n = sum(row["judge_status"] == "cached" for row in all_raw)
    sonnet_n = sum(row["judge_status"] == "sonnet" for row in all_raw)
    automatic_n = sum(
        row["judge_status"] == "automatic_zero_empty" for row in all_raw
    )
    print(f"judging: {sonnet_n} Sonnet row(s), {cached_n} cached row(s), "
          f"{automatic_n} automatic empty-answer zero(s), "
          f"{len(uncached)} non-empty pending row(s)")


if __name__ == "__main__":
    main()
