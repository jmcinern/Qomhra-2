#!/usr/bin/env python3
"""Pack CLUAS (LC Aural) exam years into a single parquet for the LUMI eval.

CLUAS = Higher-Level Irish Leaving-Cert listening comprehension (An Chluastuiscint).
Each exam year has 8 question-bearing audio clips (fógraí, comhrá míreanna, píosaí) and
~30 free-text questions answered in Irish, marked against the official scheme.

Everything needed is already in the local LC-Aural-Bench repo:
  data/<year>/cluastuiscint/benchmark.json  parsed questions + verbatim rubrics + marks
  data/<year>/snippets/<snippet_id>.wav     the question-bearing clips (16k mono 16-bit)
  data/<year>/snippets/gold.json            per-clip gold transcript
`export_jsonl.rows(year)` flattens these into one row per question with the correct
snippet->clip mapping (a comhrá splits into mir1/mir2), so we import and reuse it.

We fold ALL requested years into ONE parquet (audio bytes live once per clip; questions
grouped under their clip as a JSON string). One big file, not thousands of loose wavs, so
the cluster filesystem is happy AND each model can be loaded once to answer every year.
Snippet ids are year-prefixed (e.g. 2013-A-fogra1) so they stay unique across years.

Few-shot demos are text-only (Ceist -> Freagra) and DELIBERATELY SYNTHETIC (generic
announcement/conversation phrasing, not from any exam paper), so scoring every real year
leaks nothing. Written alongside as <out>.demos.json.

Usage (laptop, then scp both outputs to /scratch/.../eval/data/):
  # all 13 years into one file:
  python cluas_to_parquet.py --repo D:/VS-code-projects/LC-Aural-Bench \
    --out D:/VS-code-projects/Qomhra-2/ablations/eval/data/cluas_all.parquet
  # or a subset:
  python cluas_to_parquet.py --years 2013 --repo ... --out .../cluas_2013.parquet
"""
import argparse
import json
import os
import sys
import wave
from collections import OrderedDict

# Generic Irish listening-comprehension Q->A pairs. Invented, not from any exam paper, so
# using them as few-shot examples cannot leak an answer into any scored year.
SYNTHETIC_DEMOS = [
    {"question": "Cén t-am a thosaíonn an ócáid?", "answer": "A hocht a chlog san oíche."},
    {"question": "Cá mbeidh an cruinniú ar siúl?", "answer": "I halla an bhaile."},
    {"question": "Cé mhéad a chosnaíonn an ticéad?", "answer": "Deich euro."},
]


def wav_meta(path):
    with wave.open(path) as w:
        return w.getframerate(), w.getnframes() / float(w.getframerate())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*",
                    default=list(range(2013, 2026)), help="exam years to pack")
    ap.add_argument("--repo", required=True, help="local LC-Aural-Bench checkout")
    ap.add_argument("--out", required=True, help="output .parquet")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "scripts"))
    import benchmark_common
    benchmark_common.ROOT = __import__("pathlib").Path(args.repo)  # point rows() at --repo
    from export_jsonl import rows  # noqa: E402  (reuses the snippet->clip mapping)
    import pyarrow as pa
    import pyarrow.parquet as pq

    # ---- group every year's questions under their audio clip --------------
    clips = OrderedDict()
    for year in args.years:
        for r in rows(year):
            sid = r["snippet_id"]
            if r["audio_path"] is None:
                raise SystemExit(f"{r['unique_id']}: snippet {sid} has no wav in {args.repo}")
            c = clips.setdefault(sid, {
                "snippet_id": sid, "year": year, "section": r["section"],
                "item_id": r["item_id"], "gold_transcript": r["gold_transcript"] or "",
                "questions": [],
            })
            c["questions"].append({
                "question_id": r["unique_id"],
                "question_number": r["question_number"],
                "text_ga": r["question"],
                "rubric_ga": r["marking_scheme"],
                "marks_available": r["marks_available"],
                "blanks_required": r["blanks_required"],
                "canuint": r["canuint"],  # dialect: na Mumhan | Chonnacht | Uladh
            })

    sids, yrs, audio, sr_col, dur_col = [], [], [], [], []
    sect, item, qjson, gtrans = [], [], [], []
    n_q = n_marks = 0
    for sid, c in clips.items():
        wav = os.path.join(args.repo, "data", str(c["year"]), "snippets", f"{sid}.wav")
        sr, dur = wav_meta(wav)
        with open(wav, "rb") as f:
            audio.append(f.read())
        sids.append(sid); yrs.append(c["year"]); sr_col.append(sr); dur_col.append(dur)
        sect.append(c["section"]); item.append(c["item_id"])
        gtrans.append(c["gold_transcript"])
        qjson.append(json.dumps(c["questions"], ensure_ascii=False))
        n_q += len(c["questions"])
        n_marks += sum(q["marks_available"] for q in c["questions"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pq.write_table(pa.table({
        "snippet_id": sids, "year": yrs, "section": sect, "item_id": item,
        "audio": audio, "sr": sr_col, "duration_s": dur_col,
        "gold_transcript": gtrans, "questions_json": qjson,
    }), args.out)
    print(f"wrote {args.out}: {len(args.years)} years, {len(sids)} clips, "
          f"{n_q} questions, {n_marks} marks")

    # ---- synthetic text-only few-shot demos (no exam leakage) -------------
    demo_path = args.out + ".demos.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(SYNTHETIC_DEMOS, f, ensure_ascii=False, indent=2)
    print(f"wrote {demo_path}: {len(SYNTHETIC_DEMOS)} synthetic demos")


if __name__ == "__main__":
    main()
