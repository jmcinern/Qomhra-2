#!/usr/bin/env python3
"""Score IWSLT model outputs against production FOTHEIDIL transcripts."""

import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from sacrebleu.metrics import CHRF


RESULT_FILES = [
    ("base", "iwslt_st_base_base_chat_mnt64.json"),
    ("text_10pct", "iwslt_st_text_pct10_chat_mnt64.json"),
    ("text_final", "iwslt_st_text_final_raw_mnt64.json"),
    ("speech_10pct", "iwslt_st_speech_pct10_chat_mnt64.json"),
    ("speech_final", "iwslt_st_speech_final_raw_mnt64.json"),
    ("both_10pct", "iwslt_st_both_pct10_chat_mnt64.json"),
    ("both_final", "iwslt_st_both_final_raw_mnt64.json"),
    ("aligned_10pct", "iwslt_st_aligned_pct10_chat_mnt64.json"),
    ("aligned_final", "iwslt_st_aligned_final_raw_mnt64.json"),
]


def norm(text):
    """Lowercase, strip punctuation, collapse whitespace; retain accented letters."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def load_asr(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    segments = document.get("transcripts")
    if not isinstance(segments, list):
        raise ValueError(f"{path} has no transcripts list")
    return " ".join(
        segment.get("text", "").strip()
        for segment in segments
        if segment.get("text", "").strip()
    )


def load_model_rows(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if len(document) != 1:
        raise ValueError(f"{path} must contain exactly one model result")
    result = next(iter(document.values()))
    rows = result["rows"]
    mapping = {row["name"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"{path} contains duplicate utterance IDs")
    return mapping


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--asr-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = pq.read_table(
        args.data, columns=["name", "eng", "duration_s"]).to_pylist()
    if len(records) != 1120:
        raise ValueError(f"expected 1120 packed records, found {len(records)}")

    transcripts = {}
    transcript_rows = []
    for index, record in enumerate(records):
        name = record["name"]
        asr_path = args.asr_dir / f"{Path(name).stem}.json"
        if not asr_path.is_file():
            raise FileNotFoundError(asr_path)
        text = load_asr(asr_path)
        if not text:
            raise ValueError(f"empty raw ASR transcript: {asr_path}")
        transcripts[name] = text
        transcript_rows.append({
            "role": "fewshot" if index < args.shots else "scored",
            "utterance_id": name,
            "duration_s": record["duration_s"],
            "english_reference": record["eng"],
            "fotheidil_transcript": text,
            "fotheidil_transcript_normalized": norm(text),
        })

    write_csv(
        args.out_dir / "iwslt_fotheidil_transcripts.csv",
        transcript_rows,
        list(transcript_rows[0]),
    )

    tests = records[args.shots:]
    test_names = [record["name"] for record in tests]
    refs = {record["name"]: record["eng"] for record in tests}
    chrf = CHRF(char_order=6, word_order=2, beta=2)
    summary_rows = []
    per_rows = []

    for variant, filename in RESULT_FILES:
        model_rows = load_model_rows(args.results_dir / filename)
        if set(model_rows) != set(test_names):
            raise ValueError(f"{variant} utterance set differs from scored data")

        hyps = [model_rows[name]["hyp"] for name in test_names]
        asr = [transcripts[name] for name in test_names]
        eng = [refs[name] for name in test_names]
        hyps_norm = [norm(text) for text in hyps]
        asr_norm = [norm(text) for text in asr]
        eng_norm = [norm(text) for text in eng]

        exact_fotheidil_score = chrf.corpus_score(hyps, [asr]).score
        norm_asr_score = chrf.corpus_score(hyps_norm, [asr_norm]).score
        norm_eng_score = chrf.corpus_score(hyps_norm, [eng_norm]).score
        sentence_asr_scores = [
            chrf.sentence_score(hyp, [reference]).score
            for hyp, reference in zip(hyps_norm, asr_norm)
        ]
        sentence_eng_scores = [
            chrf.sentence_score(hyp, [reference]).score
            for hyp, reference in zip(hyps_norm, eng_norm)
        ]
        sentence_deltas = [
            fotheidil_score - english_score
            for fotheidil_score, english_score
            in zip(sentence_asr_scores, sentence_eng_scores)
        ]
        summary_rows.append({
            "variant": variant,
            "rows": len(test_names),
            "chrfpp_exact_hyp_vs_fotheidil": exact_fotheidil_score,
            "chrfpp_normalized_hyp_vs_fotheidil": norm_asr_score,
            "chrfpp_normalized_hyp_vs_english_reference": norm_eng_score,
            "normalized_fotheidil_minus_english": norm_asr_score - norm_eng_score,
            "mean_sentence_chrfpp_vs_fotheidil": (
                sum(sentence_asr_scores) / len(sentence_asr_scores)),
            "mean_sentence_chrfpp_vs_english": (
                sum(sentence_eng_scores) / len(sentence_eng_scores)),
            "mean_sentence_fotheidil_minus_english": (
                sum(sentence_deltas) / len(sentence_deltas)),
            "median_sentence_fotheidil_minus_english": statistics.median(
                sentence_deltas),
            "pct_utterances_fotheidil_gt_english": (
                100 * sum(delta > 0 for delta in sentence_deltas)
                / len(sentence_deltas)),
        })
        for index, name in enumerate(test_names):
            per_rows.append({
                "variant": variant,
                "utterance_id": name,
                "fotheidil_transcript": asr[index],
                "english_reference": eng[index],
                "model_hypothesis": hyps[index],
                "sentence_chrfpp_normalized_vs_fotheidil": sentence_asr_scores[index],
                "sentence_chrfpp_normalized_vs_english": sentence_eng_scores[index],
                "fotheidil_minus_english": (
                    sentence_asr_scores[index] - sentence_eng_scores[index]),
            })

    write_csv(
        args.out_dir / "iwslt_fotheidil_chrf_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_csv(
        args.out_dir / "iwslt_fotheidil_chrf_per_utterance.csv",
        per_rows,
        list(per_rows[0]),
    )
    print(
        f"wrote {len(transcript_rows)} transcripts, "
        f"{len(summary_rows)} summary rows, and {len(per_rows)} per-model rows")


if __name__ == "__main__":
    main()
