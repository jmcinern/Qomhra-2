#!/usr/bin/env python3
"""Add bounded FLEURS WER/CER to a completed discrete-ASR report."""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from asr_baseline import norm  # noqa: E402


def edits(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, 1):
        cur = [i]
        for j, hyp_item in enumerate(hyp, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (ref_item != hyp_item),
                )
            )
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    args = ap.parse_args()
    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)

    word_edits = char_edits = 0
    for row in report["rows"]:
        ref_text = norm(row["reference"])
        hyp_text = norm(row["hypothesis"])
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()[: len(ref_words)]
        we = edits(ref_words, hyp_words)
        ce = edits(ref_text, hyp_text[: len(ref_text)])
        row["word_edits_truncated"] = we
        row["char_edits_truncated"] = ce
        word_edits += we
        char_edits += ce
    report["wer_truncated"] = 100.0 * word_edits / max(report["reference_words"], 1)
    report["cer_truncated"] = 100.0 * char_edits / max(report["reference_chars"], 1)
    report["word_edits_truncated"] = word_edits
    report["char_edits_truncated"] = char_edits
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"{report['label']}: raw WER={report['wer']:.2f} CER={report['cer']:.2f}; "
        f"bounded WER={report['wer_truncated']:.2f} "
        f"CER={report['cer_truncated']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
