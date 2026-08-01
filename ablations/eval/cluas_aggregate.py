#!/usr/bin/env python3
"""Aggregate the sonnet-judge verdicts (output/judge_out/<model>.json) into the CLUAS
scoreboard.

For each model we report marks for the three conditions (just_audio / no_context /
just_transcript), the listening_gain = just_audio - no_context, and break every one of
those down THREE ways: overall, per exam year, and per DIALECT (canúint: Munster / Connacht
/ Ulster). Every question is tagged with exactly one dialect, so the dialect columns of a
condition sum back to that condition's overall total.

Marks-available, year and dialect for each question are read straight from the source
LC-Aural-Bench repo (keyed by question_id), so this does not rely on those fields being
present in the parquet the models answered from.

Usage (laptop):
  python cluas_aggregate.py --repo D:/VS-code-projects/LC-Aural-Bench \
    --judge-dir output/judge_out --out output/cluas_scoreboard.json
"""
import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

MODELS = ["baseline", "text", "speech", "both", "aligned"]
CONDS = ["just_audio", "no_context", "just_transcript"]
# canúint value in the source data -> friendly dialect label
DIALECT = {"na Mumhan": "Munster", "Chonnacht": "Connacht", "Uladh": "Ulster"}
DIALECTS = ["Munster", "Connacht", "Ulster"]


def load_question_meta(repo, years):
    """question_id -> {marks, year, dialect} from the source repo."""
    sys.path.insert(0, os.path.join(repo, "scripts"))
    import benchmark_common
    benchmark_common.ROOT = pathlib.Path(repo)
    from export_jsonl import rows
    meta = {}
    for y in years:
        for r in rows(y):
            meta[r["unique_id"]] = {
                "marks": r["marks_available"],
                "year": r["year"],
                "dialect": DIALECT.get(r["canuint"], r["canuint"]),
            }
    return meta


def blank_split():
    """Nested zeroed accumulator: cond -> {overall, per-year, per-dialect}."""
    return {c: {"overall": 0,
                "year": defaultdict(int),
                "dialect": defaultdict(int)} for c in CONDS}


def gain(split):
    """listening_gain for each slice = just_audio - no_context."""
    a, b = split["just_audio"], split["no_context"]
    g = {"overall": a["overall"] - b["overall"], "year": {}, "dialect": {}}
    for y in set(a["year"]) | set(b["year"]):
        g["year"][y] = a["year"].get(y, 0) - b["year"].get(y, 0)
    for d in DIALECTS:
        g["dialect"][d] = a["dialect"].get(d, 0) - b["dialect"].get(d, 0)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="local LC-Aural-Bench checkout")
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2013, 2026)))
    ap.add_argument("--judge-dir", default="output/judge_out")
    ap.add_argument("--out", default="output/cluas_scoreboard.json")
    ap.add_argument("--tag", default="", help="optional suffix filter on model files")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    meta = load_question_meta(args.repo, args.years)

    # marks available, sliced the same three ways, for the denominators
    avail = {"overall": 0, "year": defaultdict(int), "dialect": defaultdict(int)}
    for m in meta.values():
        avail["overall"] += m["marks"]
        avail["year"][m["year"]] += m["marks"]
        avail["dialect"][m["dialect"]] += m["marks"]

    board = {}
    for model in MODELS:
        p = os.path.join(here, args.judge_dir, f"{model}.json")
        if not os.path.exists(p):
            print(f"[skip] {model}: no verdict file at {p}")
            continue
        verdicts = json.load(open(p, encoding="utf-8"))
        split = blank_split()
        seen = 0
        for qid, verd in verdicts.items():
            if qid not in meta:
                print(f"[warn] {model}: {qid} not in source meta, skipping")
                continue
            seen += 1
            info = meta[qid]
            for c in CONDS:
                mk = int(verd.get(c, {}).get("marks", 0))
                mk = max(0, min(mk, info["marks"]))  # clamp to marks available
                split[c]["overall"] += mk
                split[c]["year"][info["year"]] += mk
                split[c]["dialect"][info["dialect"]] += mk
        board[model] = {"split": split, "gain": gain(split), "n_questions": seen}

    # ---- print: overall table -------------------------------------------
    def pline(name, s):
        g = s["just_audio"]["overall"] - s["no_context"]["overall"]
        print(f"{name:10s}{s['just_audio']['overall']:>8}"
              f"{s['no_context']['overall']:>8}{g:>7}"
              f"{s['just_transcript']['overall']:>12}")

    print(f"\nCLUAS — sonnet judge — /{avail['overall']} marks, "
          f"{len(args.years)} years\n")
    hdr = f"{'model':10s}{'audio':>8}{'blind':>8}{'gain':>7}{'transcript':>12}"
    print(hdr); print("-" * len(hdr))
    for model in MODELS:
        if model in board:
            pline(model, board[model]["split"])

    # ---- print: listening gain by dialect -------------------------------
    print(f"\nlistening gain (audio - blind) by dialect  "
          f"[avail M{avail['dialect']['Munster']} "
          f"C{avail['dialect']['Connacht']} U{avail['dialect']['Ulster']}]:")
    dh = f"  {'model':10s}" + "".join(f"{d:>10}" for d in DIALECTS)
    print(dh); print("  " + "-" * (len(dh) - 2))
    for model in MODELS:
        if model not in board:
            continue
        g = board[model]["gain"]["dialect"]
        print(f"  {model:10s}" + "".join(f"{g[d]:>10}" for d in DIALECTS))

    out = os.path.join(here, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # defaultdicts -> plain dicts for json
    def plain(s):
        return {c: {"overall": s[c]["overall"],
                    "year": dict(s[c]["year"]),
                    "dialect": dict(s[c]["dialect"])} for c in CONDS}
    dump = {"max_marks": avail["overall"],
            "avail": {"overall": avail["overall"],
                      "year": dict(avail["year"]),
                      "dialect": dict(avail["dialect"])},
            "board": {m: {"split": plain(b["split"]),
                          "gain": b["gain"],
                          "n_questions": b["n_questions"]}
                      for m, b in board.items()}}
    json.dump(dump, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
