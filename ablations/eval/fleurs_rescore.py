#!/usr/bin/env python3
"""Re-score the FLEURS results from the saved generations, after cutting off runaway output.

Nothing is regenerated: every result file stores the full raw generation, so this is pure
re-measurement on the same text and costs no GPU.

WHY THIS EXISTS. Decoding stopped only on <|im_end|>, the chat-format end marker. The
continued-pretraining checkpoints were trained on plain documents separated by <|endoftext|>,
so that is the token they emit when they are finished — and because it was not in the stop
list they were forced to carry on generating past their own answer until they hit the token
limit. On `text_final` Irish->English text this happened on 340 of 340 sentences: the correct
translation was produced, then thrown back into the scorer alongside several hundred tokens of
unrelated continuation. That is a harness artifact, not a property of the model.

THE THREE CUTS, in order. Each is applied identically to every model and every condition,
including the base model, so no variant gets a rule of its own.

  1. END MARKER. Cut at the first special token of any kind (<|endoftext|>, <|im_start|>,
     <|im_end|>, <|audio_bos|>, <|AUDIO|>, ...). These are structural markers the model can
     only have learned as boundaries; none can be part of a legitimate answer.

  2. WORD-LEVEL LOOP. If the text ends in a block of up to 40 words repeated 3+ times, keep
     the first copy and drop the rest. Catches both "duine duine duine..." and whole repeated
     sentences ("I am not sure if I can help you with that." x30).

  3. CHARACTER-LEVEL LOOP. The same test inside a single very long unbroken word (60+
     characters, no spaces). This is the only way to touch the `speech` final checkpoint,
     which emits no spaces at all ("...wellwellwellondwellwell...") and therefore has no word
     boundaries for rule 2 to work with.

WHAT THIS DOES NOT DO. It does not repair, reorder, pick the best of several attempts, or
choose between candidate answers. It only removes text that comes AFTER the model has either
signalled it was finished or started looping. A model that produced a wrong answer keeps its
wrong answer and its bad score.

BOTH NUMBERS ARE KEPT. The original scores stay in place and the comparison table prints them
side by side, because how much a variant gains from these cuts is itself a measurement.

  python fleurs_rescore.py --results output --out output/rescored \
      --compare ../../Qomhra2-Paper/evals/fleurs_rescore_comparison.md
"""
import argparse
import copy
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_baseline import strip_boilerplate, wer  # noqa: E402
from fleurs_eval import CONDITIONS, SPECIAL, cer  # noqa: E402
from iwslt_st import BLEU_SCORER, CHRFPP_SCORER  # noqa: E402

# SPECIAL comes from fleurs_eval so transcription and translation cut in exactly the same
# place; only this file needs to know whether the cut fired.

MAX_BLOCK_WORDS = 40   # longest repeating unit rule 2 will look for
MIN_REPEATS = 3        # a block must repeat this many times before it counts as a loop
MIN_RUN_CHARS = 60     # a "word" this long with no spaces is what rule 3 looks inside


def cut_at_special(text):
    """Rule 1: everything from the first structural marker onwards is not the answer."""
    m = SPECIAL.search(text)
    return (text[:m.start()], True) if m else (text, False)


def periodic_cut(seq):
    """Index at which a repeating tail starts, or None. Works on words or characters.

    Looks for the shortest block length p whose repetition runs to the very end of the
    sequence, and keeps one copy of it. The final repeat is allowed to be incomplete, which
    matters because the token limit chops the last cycle mid-way — requiring whole repeats
    missed exactly the runaway outputs this is for.

    The bar is three occurrences (the last may be partial): a block of p items must be
    periodic over more than 2p items. Two repeats is not enough — real sentences do repeat a
    phrase twice.
    """
    n = len(seq)
    for p in range(1, min(MAX_BLOCK_WORDS, n // 2) + 1):
        # The last p items are the block itself, so the comparison starts one block back and
        # walks left for as long as each item equals the one p positions later.
        start = n - p
        while start - 1 >= 0 and seq[start - 1] == seq[start - 1 + p]:
            start -= 1
        if n - start > 2 * p:            # three occurrences, final one may be partial
            return start + p
    return None


def trim_word_loop(text):
    """Rule 2: drop a repeating tail of words, keeping one copy of the repeated block."""
    words = text.split()
    cut = periodic_cut(words)
    return (" ".join(words[:cut]), True) if cut is not None else (text, False)


def trim_char_loop(text):
    """Rule 3: the same test inside one long unbroken run of characters.

    Only fires on runs of MIN_RUN_CHARS or more with no spaces, so ordinary long words in
    either language are never touched.
    """
    changed = False
    out = []
    for word in text.split():
        if len(word) >= MIN_RUN_CHARS:
            cut = periodic_cut(word)
            if cut is not None:
                word = word[:cut]
                changed = True
        out.append(word)
    return " ".join(out), changed


def clean(raw):
    """Apply the three cuts in order. Returns (hypothesis, {which rules fired})."""
    text, hit_special = cut_at_special(raw)
    text, hit_word = trim_word_loop(text)
    text, hit_char = trim_char_loop(text)
    return strip_boilerplate(text), {
        "special": hit_special, "word_loop": hit_word, "char_loop": hit_char}


def rescore_condition(name, rows):
    """Re-measure one condition from its raw generations. Same metrics, same aggregation."""
    spec = CONDITIONS[name]
    new_rows, hyps, refs = [], [], []
    fired = {"special": 0, "word_loop": 0, "char_loop": 0, "any": 0}
    for r in rows:
        hyp, hits = clean(r["raw"])
        for key, was_hit in hits.items():
            fired[key] += bool(was_hit)
        fired["any"] += any(hits.values())
        row = copy.deepcopy(r)
        row["hyp_original"] = r["hyp"]
        row["hyp"] = hyp
        row["cuts"] = sorted(k for k, v in hits.items() if v)
        ref = r["ref"]
        hyps.append(hyp)
        refs.append(ref)
        if spec["metric"] == "wer":
            w_edits, w_len = wer(ref, hyp)
            c_edits, c_len = cer(ref, hyp)
            row.update(wer=100.0 * w_edits / w_len if w_len else None,
                       cer=100.0 * c_edits / c_len if c_len else None,
                       wer_edits=w_edits, wer_ref_words=w_len,
                       cer_edits=c_edits, cer_ref_chars=c_len)
        else:
            row["chrfpp"] = CHRFPP_SCORER.sentence_score(hyp, [ref]).score
        new_rows.append(row)

    summary = {"condition": name, "metric": spec["metric"], "n": len(new_rows),
               "cuts_applied": fired}
    if spec["metric"] == "wer":
        summary["wer"] = 100.0 * sum(r["wer_edits"] for r in new_rows) / max(
            1, sum(r["wer_ref_words"] for r in new_rows))
        summary["cer"] = 100.0 * sum(r["cer_edits"] for r in new_rows) / max(
            1, sum(r["cer_ref_chars"] for r in new_rows))
    else:
        summary["chrfpp"] = CHRFPP_SCORER.corpus_score(hyps, [refs]).score
        bleu = BLEU_SCORER.corpus_score(hyps, [refs])
        summary.update(bleu=bleu.score, bleu_bp=bleu.bp, hyp_len=bleu.sys_len)
    return new_rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="output")
    ap.add_argument("--out", required=True, help="directory for the re-scored result files")
    ap.add_argument("--compare", required=True, help="path for the before/after table")
    ap.add_argument("--min-n", type=int, default=340)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    changes = []
    for path in sorted(glob.glob(os.path.join(args.results, "fleurs_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("conditions"):
            continue
        if min(p["summary"]["n"] for p in data["conditions"].values()) < args.min_n:
            continue
        label = data["run"]["label"]
        for name, payload in data["conditions"].items():
            before = payload["summary"]
            rows, after = rescore_condition(name, payload["rows"])
            payload["rows"] = rows
            # Carried through untouched: these describe how decoding ran, not how it scored.
            for key in ("stop_reasons", "cap_hit_rate", "mean_generated_tokens"):
                after[key] = before.get(key)
            payload["summary"] = after
            payload["summary_original"] = before
            changes.append((label, name, before, after))
        data["run"]["rescored"] = {
            "rules": ["cut at first special token", "trim repeated trailing word block",
                      "trim repeated run inside an unbroken 60+ char word"],
            "min_repeats": MIN_REPEATS, "max_block_words": MAX_BLOCK_WORDS,
            "min_run_chars": MIN_RUN_CHARS,
            "note": "hypotheses re-derived from the stored raw generations; nothing regenerated",
        }
        with open(os.path.join(args.out, os.path.basename(path)), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"rescored {os.path.basename(path)}")

    write_comparison(changes, args.compare)


def write_comparison(changes, path):
    """Before/after, one table per metric family, so the two are never mixed."""
    def fmt(v):
        return "—" if v is None else f"{v:.2f}"

    chrf = [c for c in changes if c[3]["metric"] == "chrf"]
    werr = [c for c in changes if c[3]["metric"] == "wer"]
    lines = ["# FLEURS — effect of cutting runaway generations", "",
             "Same generations, re-measured after removing text that comes after the model "
             "either signalled it had finished (a structural end marker) or began repeating "
             "itself. Nothing was regenerated and no answer was repaired. The rules are "
             "applied identically to every variant, including the base model.", "",
             "`cut %` is the share of the 340 sentences where any rule fired.", ""]

    lines += ["## Translation conditions (chrF++, higher is better)", "",
              "| model | condition | before | after | change | cut % |", "|---|---|---|---|---|---|"]
    for label, name, b, a in sorted(chrf, key=lambda c: -(c[3]["chrfpp"] - c[2]["chrfpp"])):
        pct = 100.0 * a["cuts_applied"]["any"] / a["n"]
        lines.append(f"| {label} | {name} | {fmt(b['chrfpp'])} | {fmt(a['chrfpp'])} | "
                     f"**{a['chrfpp'] - b['chrfpp']:+.2f}** | {pct:.0f} |")

    lines += ["", "## Transcription and copy control (WER % / CER %, lower is better)", "",
              "| model | condition | WER before | WER after | change | CER before | CER after "
              "| cut % |", "|---|---|---|---|---|---|---|---|"]
    for label, name, b, a in sorted(werr, key=lambda c: c[3]["wer"] - c[2]["wer"]):
        pct = 100.0 * a["cuts_applied"]["any"] / a["n"]
        lines.append(f"| {label} | {name} | {fmt(b['wer'])} | {fmt(a['wer'])} | "
                     f"**{a['wer'] - b['wer']:+.2f}** | {fmt(b['cer'])} | {fmt(a['cer'])} | "
                     f"{pct:.0f} |")

    lines += ["", "## Which rule fired (share of 340 sentences, %)", "",
              "| model | condition | end marker | word loop | character loop |",
              "|---|---|---|---|---|"]
    for label, name, _, a in changes:
        c = a["cuts_applied"]
        lines.append(f"| {label} | {name} | {100.0 * c['special'] / a['n']:.0f} | "
                     f"{100.0 * c['word_loop'] / a['n']:.0f} | "
                     f"{100.0 * c['char_loop'] / a['n']:.0f} |")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
