#!/usr/bin/env python3
"""Route IWSLT outputs as translation/transcription using fastText language ID.

This does not rewrite a hypothesis.  It records the language classifier's top
label/confidence, scores every untouched output against both references, and
routes English to the human translation and Irish to the FOTHEIDIL transcript.
Other languages and empty outputs stay explicitly unclassified.
"""
import argparse
import csv
import glob
import json
import os
import statistics

import fasttext
from sacrebleu.metrics import CHRF


CHRFPP = CHRF(char_order=6, word_order=2, beta=2)


def clean_for_lid(text):
    return " ".join((text or "").replace("\n", " ").split())


def sentence_chrf(hypothesis, reference):
    return CHRFPP.sentence_score(hypothesis, [reference or ""]).score


def corpus_chrf(rows, reference_key):
    if not rows:
        return None
    return CHRFPP.corpus_score(
        [row["hypothesis"] for row in rows],
        [[row[reference_key] for row in rows]],
    ).score


def classify(model, hypothesis):
    text = clean_for_lid(hypothesis)
    if not text:
        return "empty", None, "empty"
    labels, probabilities = model.predict(text, k=1)
    language = labels[0].removeprefix("__label__")
    route = (
        "translation" if language == "en"
        else "transcription" if language == "ga"
        else "other"
    )
    return language, float(probabilities[0]), route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--lid-model", required=True)
    ap.add_argument("--numbers-out", required=True)
    ap.add_argument("--raw-out", required=True)
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(glob.glob(pattern))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("no IWSLT JSON inputs matched")

    lid = fasttext.load_model(args.lid_model)
    raw_rows = []
    summaries = []
    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        model_label = payload["run"]["label"].removesuffix("_p10")
        routed = []
        for row in payload["rows"]:
            language, confidence, route = classify(lid, row["hypothesis"])
            english_score = sentence_chrf(row["hypothesis"], row["ref"])
            irish_score = sentence_chrf(
                row["hypothesis"], row.get("fotheidil_ref") or ""
            )
            routed_score = (
                english_score if route == "translation"
                else irish_score if route == "transcription"
                else None
            )
            item = {
                "model": model_label,
                "item": row["item"],
                "reference_english": row["ref"],
                "reference_fotheidil_irish": row.get("fotheidil_ref"),
                "hypothesis": row["hypothesis"],
                "lid_language": language,
                "lid_confidence": confidence,
                "route": route,
                "chrfpp_vs_english": english_score,
                "chrfpp_vs_fotheidil": irish_score,
                "routed_chrfpp": routed_score,
                "stop_reason": row["stop_reason"],
                "emitted_stop_token": row.get("emitted_stop_token"),
                "reference_tokens": row.get("reference_tokens"),
                "generation_budget": row.get("generation_budget"),
                "generated_tokens": row["generated_tokens"],
                "raw_output_ids": json.dumps(row["raw_output_ids"]),
            }
            raw_rows.append(item)
            routed.append(item)

        translations = [row for row in routed if row["route"] == "translation"]
        transcriptions = [
            row for row in routed if row["route"] == "transcription"
        ]
        other = [row for row in routed if row["route"] == "other"]
        empty = [row for row in routed if row["route"] == "empty"]
        scored = [row["routed_chrfpp"] for row in routed
                  if row["routed_chrfpp"] is not None]
        summaries.append({
            "model": model_label,
            "n": len(routed),
            "translation_n": len(translations),
            "translation_pct": 100 * len(translations) / len(routed),
            "transcription_n": len(transcriptions),
            "transcription_pct": 100 * len(transcriptions) / len(routed),
            "other_n": len(other),
            "empty_n": len(empty),
            "original_translation_chrfpp_all": payload["chrfpp"],
            "translation_chrfpp_on_lid_en": corpus_chrf(
                translations, "reference_english"
            ),
            "transcription_chrfpp_on_lid_ga": corpus_chrf(
                transcriptions, "reference_fotheidil_irish"
            ),
            "mean_sentence_routed_chrfpp": (
                statistics.mean(scored) if scored else None
            ),
            "natural_eos_n": sum(
                row["stop_reason"] == "eos" for row in routed
            ),
            "cap_n": sum(
                row["stop_reason"] == "max_new_tokens" for row in routed
            ),
        })

    for path, rows in [(args.numbers_out, summaries), (args.raw_out, raw_rows)]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
