#!/usr/bin/env python3
"""Corpus WER/CER for FOTHEIDIL on FLEURS Irish.

FOTHEIDIL is the in-house production service: diarization -> NeMo Irish ASR ->
Marian capitalisation/punctuation.  It returns one entry per diarized segment, so
the hypothesis for a clip is every segment concatenated in start-time order.

Scoring reuses asr_baseline.norm/wer and fleurs_eval.cer, the same functions the
discrete-ASR evaluations use, so the numbers are directly comparable.  norm()
lowercases and strips punctuation, which removes the casing/punctuation that
Marian restores -- FLEURS references carry neither.
"""

import argparse
import glob
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def norm(s):
    """Copy of asr_baseline.norm -- importing it pulls in torch, unavailable here."""
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _edits(r, h):
    if not r:
        return 0, 0
    prev = list(range(len(h) + 1))
    for i, rt in enumerate(r, 1):
        cur = [i]
        for j, ht in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rt != ht)))
        prev = cur
    return prev[len(h)], len(r)


def wer(ref, hyp):
    return _edits(norm(ref).split(), norm(hyp).split())


def cer(ref, hyp):
    return _edits(norm(ref), norm(hyp))


def hypothesis(payload):
    segs = sorted(payload.get("transcripts", []),
                  key=lambda s: (s.get("startTimeSeconds") or 0.0))
    return " ".join((s.get("text") or "").strip() for s in segs).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="dir of FOTHEIDIL *.json")
    ap.add_argument("--manifest", required=True, help="FLEURS manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    by_stem = {os.path.splitext(m["file"])[0]: m for m in manifest}

    rows = []
    we = wc = ce = cc = 0
    for path in sorted(glob.glob(os.path.join(args.responses, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        meta = by_stem.get(stem)
        if meta is None:
            raise RuntimeError(f"no manifest entry for {stem}")
        payload = json.load(open(path, encoding="utf-8"))
        hyp = hypothesis(payload)
        # FLEURS ships a normalised reference; fall back to the cased one.
        ref = meta.get("normalised") or meta["transcript"]
        a, b = wer(ref, hyp)
        c, d = cer(ref, hyp)
        we += a; wc += b; ce += c; cc += d
        rows.append({
            "file": meta["file"], "id": meta.get("id"), "gender": meta.get("gender"),
            "segments": len(payload.get("transcripts", [])),
            "reference": ref, "hypothesis": hyp,
            "word_edits": a, "reference_words": b,
            "char_edits": c, "reference_chars": d,
            "wer": 100 * a / max(b, 1), "cer": 100 * c / max(d, 1),
        })

    report = {
        "system": "FOTHEIDIL production (diarization -> NeMo Irish ASR -> Marian capt)",
        "data": "FLEURS ga_ie dev",
        "n": len(rows),
        "scoring": "corpus WER/CER; lowercased, punctuation stripped, whitespace collapsed",
        "corpus_wer": 100 * we / max(wc, 1),
        "corpus_cer": 100 * ce / max(cc, 1),
        "total_reference_words": wc,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"n={len(rows)}  corpus WER={report['corpus_wer']:.2f}  "
          f"CER={report['corpus_cer']:.2f}  ({wc} reference words)")
    print()
    for r in sorted(rows, key=lambda x: x["wer"]):
        print(f"  WER {r['wer']:6.1f}  CER {r['cer']:6.1f}  seg={r['segments']}  {r['file']}")
        print(f"    REF: {r['reference'][:100]}")
        print(f"    HYP: {r['hypothesis'][:100]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
