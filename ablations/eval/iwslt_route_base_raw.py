#!/usr/bin/env python3
"""Route base_raw's IWSLT generations the same way iwslt_language_route.py
routes every other model, so the transcription-bias table has a base row
on the matched (plain-document) prompt template instead of the old
chat-template numbers.

Input is the per-example JSON from the base_raw rerun
(base_raw_p10_mult2.5_iwslt.json), not covered by the original script's
glob pattern for the ab1-4/base(chat) files.
"""
import csv
import json
import os
import statistics

import fasttext
from sacrebleu.metrics import CHRF

CHRFPP = CHRF(char_order=6, word_order=2, beta=2)

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(
    HERE, "output", "partial10_pre_ablations", "base_raw_p10_mult2.5_iwslt.json"
)
LID_MODEL = os.path.join(HERE, "lid.176.ftz")
NUMBERS_OUT = os.path.join(HERE, "output", "review_csv", "iwslt_numbers_base_raw.csv")
RAW_OUT = os.path.join(HERE, "output", "review_csv", "iwslt_raw_outputs_base_raw.csv")


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
    language = labels[0].replace("__label__", "")
    route = (
        "translation" if language == "en"
        else "transcription" if language == "ga"
        else "other"
    )
    return language, float(probabilities[0]), route


def main():
    lid = fasttext.load_model(LID_MODEL)
    payload = json.load(open(INPUT_JSON, encoding="utf-8"))

    routed = []
    for row in payload["rows"]:
        language, confidence, route = classify(lid, row["hypothesis"])
        english_score = sentence_chrf(row["hypothesis"], row["ref"])
        irish_score = sentence_chrf(row["hypothesis"], row.get("fotheidil_ref") or "")
        routed_score = (
            english_score if route == "translation"
            else irish_score if route == "transcription"
            else None
        )
        routed.append({
            "model": "base_raw",
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
        })

    translations = [r for r in routed if r["route"] == "translation"]
    transcriptions = [r for r in routed if r["route"] == "transcription"]
    other = [r for r in routed if r["route"] == "other"]
    empty = [r for r in routed if r["route"] == "empty"]
    scored = [r["routed_chrfpp"] for r in routed if r["routed_chrfpp"] is not None]

    summary = {
        "model": "base_raw",
        "n": len(routed),
        "translation_n": len(translations),
        "translation_pct": 100 * len(translations) / len(routed),
        "transcription_n": len(transcriptions),
        "transcription_pct": 100 * len(transcriptions) / len(routed),
        "other_n": len(other),
        "empty_n": len(empty),
        "original_translation_chrfpp_all": payload["chrfpp"],
        "translation_chrfpp_on_lid_en": corpus_chrf(translations, "reference_english"),
        "transcription_chrfpp_on_lid_ga": corpus_chrf(
            transcriptions, "reference_fotheidil_irish"
        ),
        "mean_sentence_routed_chrfpp": statistics.mean(scored) if scored else None,
        "natural_eos_n": sum(r["stop_reason"] == "eos" for r in routed),
        "cap_n": sum(r["stop_reason"] == "max_new_tokens" for r in routed),
    }

    os.makedirs(os.path.dirname(NUMBERS_OUT), exist_ok=True)
    with open(NUMBERS_OUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    with open(RAW_OUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(routed[0]))
        writer.writeheader()
        writer.writerows(routed)

    print(f"wrote {NUMBERS_OUT}")
    print(f"wrote {RAW_OUT}")
    print(summary)


if __name__ == "__main__":
    main()
