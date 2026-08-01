#!/usr/bin/env python3
"""Join Banba FOTHEIDIL JSON responses to the IWSLT review and few-shot manifests."""

import argparse
import csv
import json
from pathlib import Path


def load_transcription(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    segments = document.get("transcripts")
    if not isinstance(segments, list):
        raise ValueError(f"{path} has no transcripts list")
    transcript = " ".join(
        segment.get("text", "").strip() for segment in segments
        if segment.get("text", "").strip()
    )
    if not transcript:
        raise ValueError(f"{path} has an empty transcript")
    speakers = sorted({
        segment.get("speaker", "") for segment in segments
        if segment.get("speaker", "")
    })
    return {
        "fotheidil_transcript": transcript,
        "fotheidil_speakers": "|".join(speakers),
        "fotheidil_start_s": min(segment["startTimeSeconds"] for segment in segments),
        "fotheidil_end_s": max(segment["endTimeSeconds"] for segment in segments),
        "fotheidil_checked": all(segment.get("checked") is True for segment in segments),
        "fotheidil_segments_json": json.dumps(segments, ensure_ascii=False),
        "irish_transcript_reviewed": "",
        "irish_transcript_review_notes": "",
    }


def write_review(source_csv, json_dir, output_csv):
    with source_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        source_fields = reader.fieldnames
    if not source_fields:
        raise ValueError(f"{source_csv} has no header")

    extra_fields = [
        "fotheidil_transcript",
        "fotheidil_speakers",
        "fotheidil_start_s",
        "fotheidil_end_s",
        "fotheidil_checked",
        "fotheidil_segments_json",
        "irish_transcript_reviewed",
        "irish_transcript_review_notes",
    ]
    insert_at = source_fields.index("english_reference") + 1
    fields = source_fields[:insert_at] + extra_fields + source_fields[insert_at:]

    for row in rows:
        result_path = json_dir / f"{Path(row['utterance_id']).stem}.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        row.update(load_transcription(result_path))

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_fewshots(source_manifest, json_dir, output_csv):
    rows = json.loads(source_manifest.read_text(encoding="utf-8"))
    fields = [
        "fewshot_no",
        "utterance_id",
        "duration_s",
        "english_reference",
        "fotheidil_transcript",
        "fotheidil_speakers",
        "fotheidil_start_s",
        "fotheidil_end_s",
        "fotheidil_checked",
        "fotheidil_segments_json",
        "irish_transcript_reviewed",
        "irish_transcript_review_notes",
    ]
    output_rows = []
    for index, row in enumerate(rows, start=1):
        result_path = json_dir / f"{Path(row['name']).stem}.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        output_rows.append({
            "fewshot_no": index,
            "utterance_id": row["name"],
            "duration_s": row["duration_s"],
            "english_reference": row["eng"],
            **load_transcription(result_path),
        })

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--review-json-dir", type=Path, required=True)
    parser.add_argument("--fewshot-manifest", type=Path, required=True)
    parser.add_argument("--fewshot-json-dir", type=Path, required=True)
    parser.add_argument("--review-out", type=Path, required=True)
    parser.add_argument("--fewshot-out", type=Path, required=True)
    args = parser.parse_args()

    review_count = write_review(
        args.review_csv, args.review_json_dir, args.review_out)
    fewshot_count = write_fewshots(
        args.fewshot_manifest, args.fewshot_json_dir, args.fewshot_out)
    print(f"wrote {review_count} review rows and {fewshot_count} few-shot rows")


if __name__ == "__main__":
    main()
