#!/usr/bin/env python3
"""Turn the sonnet-judge marks into the CSV tables the paper needs.

Reads the merged judge verdicts (output/cluas_judge/judge_scored{,_10pct}/<model>.json)
and the judge packets (the answers that were marked), joins every question to its
metadata from the source LC-Aural-Bench repo, and writes five CSVs:

  cluas_long.csv        one row per model x checkpoint x question x condition
                        (marks, judge reason, the model's actual answer) - this is the
                        file to do error analysis from; every other table is a sum of it
  cluas_overall.csv     model x checkpoint, the three conditions + listening gain
  cluas_by_task.csv     the same, split by exam task (Fogra / Comhra / Piosa)
  cluas_by_dialect.csv  the same, split by dialect (Munster / Connacht / Ulster)
  cluas_by_year.csv     the same, split by exam year

Usage:
  python cluas_report.py --repo D:/VS-code-projects/LC-Aural-Bench \
    --out-dir D:/VS-code-projects/Qomhra-2/Qomhra2-Paper/evals
"""
import argparse
import csv
import json
import os
import pathlib
import sys
from collections import defaultdict

MODELS = ["baseline", "text", "speech", "both", "aligned"]
CONDS = ["just_audio", "no_context", "just_transcript"]
DIALECT = {"na Mumhan": "Munster", "Chonnacht": "Connacht", "Uladh": "Ulster"}
DIALECTS = ["Munster", "Connacht", "Ulster"]
# item_type in the source data -> the exam task, in Irish with the English gloss
TASK = {"fogra": "Fogra", "comhra": "Comhra", "piosa": "Piosa"}
TASKS = ["Fogra", "Comhra", "Piosa"]
TASK_EN = {"Fogra": "announcement", "Comhra": "conversation", "Piosa": "passage"}
# the base model has no continued pretraining, so it has a single checkpoint
CHECKPOINTS = ["final", "10pct"]


def load_meta(repo, years):
    """question_id -> everything we slice by."""
    sys.path.insert(0, os.path.join(repo, "scripts"))
    import benchmark_common
    benchmark_common.ROOT = pathlib.Path(repo)
    from export_jsonl import rows
    meta = {}
    for y in years:
        for r in rows(y):
            meta[r["unique_id"]] = {
                "year": r["year"],
                "section": r["section"],
                "item_id": r["item_id"],
                "task": TASK.get(r["item_type"], r["item_type"]),
                "dialect": DIALECT.get(r["canuint"], r["canuint"]),
                "marks_available": r["marks_available"],
                "question": r["question"],
                "marking_scheme": r["marking_scheme"],
            }
    return meta


def read_variant(judge_root, model, checkpoint):
    """(verdicts, packet) for one model/checkpoint, or (None, None) if absent."""
    sub = "judge_scored" if checkpoint == "final" else "judge_scored_10pct"
    tag = "" if checkpoint == "final" else "_10pct"
    vp = os.path.join(judge_root, sub, f"{model}.json")
    pp = os.path.join(judge_root, "judge_packets", f"{model}{tag}.json")
    if not os.path.exists(vp):
        return None, None
    verdicts = json.load(open(vp, encoding="utf-8"))
    packet = json.load(open(pp, encoding="utf-8")) if os.path.exists(pp) else {}
    return verdicts, packet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="D:/VS-code-projects/LC-Aural-Bench")
    ap.add_argument("--judge-root", default="output/cluas_judge")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2013, 2026)))
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    judge_root = os.path.join(here, args.judge_root)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    meta = load_meta(args.repo, args.years)

    # marks available per slice, for the denominators
    avail = {"overall": 0}
    for key in ("task", "dialect", "year", "section"):
        avail[key] = defaultdict(int)
    for m in meta.values():
        avail["overall"] += m["marks_available"]
        for key in ("task", "dialect", "year", "section"):
            avail[key][m[key]] += m["marks_available"]

    # ---- long table + accumulators --------------------------------------
    long_rows = []
    # acc[(model, checkpoint)][slice_key][slice_value][cond] = marks
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    n_q = {}

    for model in MODELS:
        for ckpt in CHECKPOINTS:
            # baseline is checkpoint-free: it is the same model in both scoreboards,
            # so we record it once and label it "base"
            if model == "baseline" and ckpt == "10pct":
                continue
            verdicts, packet = read_variant(judge_root, model, ckpt)
            if verdicts is None:
                print(f"[skip] {model}/{ckpt}: no verdict file")
                continue
            label = "base" if model == "baseline" else ckpt
            seen = 0
            for qid, verd in verdicts.items():
                if qid not in meta:
                    print(f"[warn] {model}/{ckpt}: {qid} not in source meta")
                    continue
                seen += 1
                info = meta[qid]
                answers = (packet.get(qid) or {}).get("answers", {})
                for c in CONDS:
                    v = verd.get(c) or {}
                    mk = max(0, min(int(v.get("marks", 0)), info["marks_available"]))
                    long_rows.append({
                        "model": model,
                        "checkpoint": label,
                        "question_id": qid,
                        "year": info["year"],
                        "section": info["section"],
                        "task": info["task"],
                        "task_en": TASK_EN.get(info["task"], ""),
                        "item_id": info["item_id"],
                        "dialect": info["dialect"],
                        "condition": c,
                        "marks": mk,
                        "marks_available": info["marks_available"],
                        "judge_reason": (v.get("reason") or "").replace("\n", " "),
                        "answer": (answers.get(c) or "").replace("\n", " ").strip(),
                        "question": info["question"].replace("\n", " "),
                        "marking_scheme": info["marking_scheme"].replace("\n", " | "),
                    })
                    a = acc[(model, label)]
                    a["overall"]["all"][c] += mk
                    for key in ("task", "dialect", "year", "section"):
                        a[key][info[key]][c] += mk
            n_q[(model, label)] = seen

    def pct(num, den):
        return round(100.0 * num / den, 2) if den else 0.0

    def summary_rows(slice_key, order, avail_map, colname):
        out = []
        for (model, label), a in acc.items():
            for val in order:
                d = a[slice_key].get(val, {})
                av = avail_map[val] if slice_key != "overall" else avail_map
                audio = d.get("just_audio", 0)
                blind = d.get("no_context", 0)
                trans = d.get("just_transcript", 0)
                row = {"model": model, "checkpoint": label}
                if colname:
                    row[colname] = val
                    if slice_key == "task":
                        row["task_en"] = TASK_EN.get(val, "")
                row.update({
                    "marks_available": av,
                    "just_audio": audio,
                    "no_context": blind,
                    "just_transcript": trans,
                    "listening_gain": audio - blind,
                    "just_audio_pct": pct(audio, av),
                    "no_context_pct": pct(blind, av),
                    "just_transcript_pct": pct(trans, av),
                    "listening_gain_pct": pct(audio - blind, av),
                })
                out.append(row)
        return out

    order_model = {m: i for i, m in enumerate(MODELS)}
    order_ckpt = {"base": 0, "10pct": 1, "final": 2}

    def write_csv(name, rows, sort_extra=None):
        if not rows:
            return
        rows = sorted(rows, key=lambda r: (order_model[r["model"]],
                                           order_ckpt.get(r["checkpoint"], 9),
                                           sort_extra(r) if sort_extra else 0))
        p = os.path.join(out_dir, name)
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p}  ({len(rows)} rows)")

    write_csv("cluas_overall.csv",
              summary_rows("overall", ["all"], avail["overall"], None))
    write_csv("cluas_by_task.csv",
              summary_rows("task", TASKS, avail["task"], "task"),
              lambda r: TASKS.index(r["task"]))
    write_csv("cluas_by_dialect.csv",
              summary_rows("dialect", DIALECTS, avail["dialect"], "dialect"),
              lambda r: DIALECTS.index(r["dialect"]))
    years = sorted(avail["year"])
    write_csv("cluas_by_year.csv",
              summary_rows("year", years, avail["year"], "year"),
              lambda r: r["year"])

    # long table last (biggest)
    p = os.path.join(out_dir, "cluas_long.csv")
    long_rows.sort(key=lambda r: (order_model[r["model"]],
                                  order_ckpt.get(r["checkpoint"], 9),
                                  r["question_id"], CONDS.index(r["condition"])))
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)
    print(f"wrote {p}  ({len(long_rows)} rows)")

    # ---- console tables --------------------------------------------------
    def table(title, slice_key, order, avail_map):
        print(f"\n{title}")
        hdr = f"  {'model':10s}{'ckpt':>7}" + "".join(
            f"{str(v)[:9]:>11}" for v in order)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        print(f"  {'[avail]':10s}{'':>7}" + "".join(
            f"{avail_map[v]:>11}" for v in order))
        for model in MODELS:
            for label in ("base", "10pct", "final"):
                a = acc.get((model, label))
                if not a:
                    continue
                cells = []
                for v in order:
                    d = a[slice_key].get(v, {})
                    cells.append(d.get("just_audio", 0) - d.get("no_context", 0))
                print(f"  {model:10s}{label:>7}" + "".join(f"{c:>11}" for c in cells))

    print(f"\nCLUAS listening gain (audio - blind), /{avail['overall']} marks total")
    table("by TASK", "task", TASKS, avail["task"])
    table("by DIALECT", "dialect", DIALECTS, avail["dialect"])

    print("\nquestions marked per variant:",
          {f"{m}/{c}": n for (m, c), n in sorted(n_q.items())})


if __name__ == "__main__":
    main()
