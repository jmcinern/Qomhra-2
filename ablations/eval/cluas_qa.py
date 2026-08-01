#!/usr/bin/env python3
"""CLUAS (LC Aural) listening-comprehension QA: base model (+ ablation checkpoints).

Irish audio in, a short Irish answer out, per exam question. This is the priority
benchmark of the LUMI eval matrix over base + 4 ablations (text/speech/both/aligned).
Answers are marked OFF this script, against the official scheme, by an LLM judge — here
we only GENERATE answers and write them as {question_id: answer} JSON (the exact shape
LC-Aural-Bench/scripts/scoring_judge.py + evaluate.py consume).

Three conditions per model (evaluate.py's core metric is the gap between the first two):
  just_audio       hear the clip's audio, answer the question   (listening)
  no_context       question only, no audio/transcript           (blind control)
  just_transcript  gold transcript as text, no audio            (text ceiling)

Prompting mirrors the IWSLT finding: base is instruct-tuned -> chat template; the CPT
ablations were pretrained on raw <|endoftext|>-separated documents -> raw format (chat is
OOD and they babble). Few-shot demos are TEXT-ONLY Ceist->Freagra pairs from a held-out
year (see cluas_to_parquet.py) with '\\n' newlines — the clips are ~60s so audio demos
would pile multiple 30s windows into context.

Clips are 48-79s > the 30s Qwen audio window, so each clip is CHUNKED into <=30s windows,
one <|AUDIO|> segment per chunk, in order.

  python -m eval.cluas_qa --data .../cluas_2013.parquet --shots 3 \
    --conditions just_audio no_context just_transcript \
    [--checkpoints <ckpt> ...] [--labels ...] [--prompt-format raw] --out-dir output
"""
import argparse
import io
import json
import math
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

# MIOpen on the LUMI rocm6.2 container cannot build its conv kernel DBs, so every conv in
# the audio tower errors out. Routing conv through PyTorch's native fallback fixes it.
# Must be set before the first conv. (See iwslt_st.py / miopen-conv-disabled-lumi.)
torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_baseline import SYS, strip_boilerplate  # noqa: E402
from audio_compare import check_not_nan  # noqa: E402
from iwslt_st import decode_audio, AUDIO_MARKERS, RAW_STOP, TARGET_SR  # noqa: E402

SNAPSHOT_GLOB = ("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")
AUDIO_WINDOW_S = 30.0  # Qwen2.5-Omni audio encoder window.
# Irish instruction: listen and answer the question in Irish, concisely.
INSTR_GA = ("Éist leis an taifead agus freagair an cheist seo a leanas i nGaeilge. "
            "Tabhair freagra gonta amháin agus ná habair aon rud eile.")

# The raw-format models answer, then (inside the token window) echo the demo pattern by
# emitting the next "Ceist: ... Freagra: ...". The real answer is everything before that
# next marker, so we treat "Ceist:"/"Freagra:" as a soft stop when parsing the answer:
# drop a leading label the model repeated, then cut at the next marker.
_QA_MARKER = re.compile(r"(?:Ceist|Freagra)\s*:", re.IGNORECASE)


def first_answer(text):
    t = re.sub(r"^\s*(?:Ceist|Freagra)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    m = _QA_MARKER.search(t)
    if m:
        t = t[:m.start()]
    return t.strip()


# --------------------------------------------------------------------------
# Chunk a clip into <=win-second windows (a small overlap so a word split across a
# boundary survives in one of the two windows). Each chunk -> one <|AUDIO|> segment.
# --------------------------------------------------------------------------
def chunk_audio(wav, sr=TARGET_SR, win=AUDIO_WINDOW_S, overlap=1.0):
    w = int(win * sr)
    hop = int((win - overlap) * sr)
    if len(wav) <= w:
        return [wav]
    chunks = [wav[i:i + w] for i in range(0, len(wav), hop) if i < len(wav)]
    # A trailing window shorter than the 1s overlap is already fully contained in
    # the previous window; such a sliver has too few frames for the audio encoder
    # pooler and crashes it, so drop it (always keep at least one chunk).
    min_len = int(overlap * sr)
    kept = [c for c in chunks if len(c) >= min_len]
    return kept or chunks[:1]


# --------------------------------------------------------------------------
# Prompt construction, format- and condition-aware.
# --------------------------------------------------------------------------
def build_raw(demos, chunks, transcript, question, condition, demo_sep="\n"):
    parts = [f"Ceist: {q}\nFreagra: {a}{RAW_STOP}" for q, a in demos]
    if condition == "just_audio":
        head = AUDIO_MARKERS * len(chunks) + "\n"
        parts.append(f"{head}Ceist: {question}\nFreagra:")
    elif condition == "just_transcript":
        parts.append(f"Tras-scríbhinn: {transcript}\nCeist: {question}\nFreagra:")
    else:  # no_context
        parts.append(f"Ceist: {question}\nFreagra:")
    return demo_sep.join(parts)


def build_chat(proc, demos, chunks, transcript, question, condition):
    conv = [{"role": "system", "content": [{"type": "text", "text": INSTR_GA}]}]
    for q, a in demos:
        conv.append({"role": "user", "content": [{"type": "text", "text": f"Ceist: {q}"}]})
        conv.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    content = []
    if condition == "just_audio":
        content += [{"type": "audio", "audio": "c"} for _ in chunks]
        content.append({"type": "text", "text": f"Ceist: {question}"})
    elif condition == "just_transcript":
        content.append({"type": "text",
                        "text": f"Tras-scríbhinn: {transcript}\nCeist: {question}"})
    else:
        content.append({"type": "text", "text": f"Ceist: {question}"})
    conv.append({"role": "user", "content": content})
    return proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def answer(thinker, proc, demos, chunks, transcript, question, condition, device,
           prompt_format="chat", max_new_tokens=64, repetition_penalty=1.0,
           no_repeat_ngram=0):
    tok = proc.tokenizer
    audios = chunks if condition == "just_audio" else []
    if prompt_format == "raw":
        eos_id = tok.convert_tokens_to_ids(RAW_STOP)
        text = build_raw(demos, chunks, transcript, question, condition)
    else:
        eos_id = tok.convert_tokens_to_ids("<|im_end|>")
        text = build_chat(proc, demos, chunks, transcript, question, condition)
    inputs = proc(text=text, audio=(audios or None), sampling_rate=TARGET_SR,
                  return_tensors="pt", padding=True).to(device)
    gen = dict(max_new_tokens=max_new_tokens, do_sample=False,
               eos_token_id=eos_id, pad_token_id=tok.pad_token_id)
    if repetition_penalty and repetition_penalty != 1.0:
        gen["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram:
        gen["no_repeat_ngram_size"] = no_repeat_ngram
    out = thinker.generate(**inputs, **gen)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_model(thinker, proc, clips, demos, conditions, device, label, prompt_format,
              max_new_tokens=64, repetition_penalty=1.0, no_repeat_ngram=0):
    """Returns {condition: {question_id: answer}} and a rows list for inspection."""
    per_cond = {c: {} for c in conditions}
    rows = []
    for clip in clips:
        chunks = chunk_audio(clip["wav"])
        for cond in conditions:
            for q in clip["questions"]:
                raw = answer(thinker, proc, demos, chunks, clip["transcript"],
                             q["text_ga"], cond, device, prompt_format=prompt_format,
                             max_new_tokens=max_new_tokens,
                             repetition_penalty=repetition_penalty,
                             no_repeat_ngram=no_repeat_ngram)
                check_not_nan(f"{label}/{cond}", raw)
                ans = first_answer(strip_boilerplate(raw))
                per_cond[cond][q["question_id"]] = ans
                rows.append({"model": label, "condition": cond,
                             "question_id": q["question_id"], "question": q["text_ga"],
                             "answer": ans, "raw": raw})
    return per_cond, rows


def load_clips(parquet_path, n_clips, year=None):
    import pyarrow.parquet as pq
    recs = pq.read_table(parquet_path).to_pylist()
    if year:
        recs = [r for r in recs if r.get("year") == year]
        if not recs:
            raise SystemExit(f"no clips for year {year} in {parquet_path}")
    if n_clips:
        recs = recs[:n_clips]
    clips = []
    for r in recs:
        clips.append({
            "snippet_id": r["snippet_id"],
            "wav": decode_audio(r["audio"]),
            "transcript": r["gold_transcript"],
            "questions": json.loads(r["questions_json"]),
        })
    n_q = sum(len(c["questions"]) for c in clips)
    print(f"[data] {len(clips)} clips, {n_q} questions "
          f"(max {max(c['wav'].shape[0] for c in clips)/TARGET_SR:.0f}s)", flush=True)
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cluas_<year>.parquet")
    ap.add_argument("--demos", default=None, help="<data>.demos.json (default: derive)")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--n-clips", type=int, default=0, help="limit clips (0 = all)")
    ap.add_argument("--year", type=int, default=0,
                    help="only clips from this exam year (0 = all years in the parquet)")
    ap.add_argument("--conditions", nargs="+",
                    default=["just_audio", "no_context", "just_transcript"])
    ap.add_argument("--prompt-format", choices=["chat", "raw"], default="chat",
                    help="chat for base (instruct), raw for CPT ablations")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="uniform decoding knob (A/B probe); 1.0 = off")
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="uniform no_repeat_ngram_size (A/B probe); 0 = off")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tag", default=None, help="filename tag, e.g. 2013")
    args = ap.parse_args()
    gen_kw = dict(max_new_tokens=args.max_new_tokens,
                  repetition_penalty=args.repetition_penalty,
                  no_repeat_ngram=args.no_repeat_ngram)

    tag = args.tag or os.path.splitext(os.path.basename(args.data))[0].replace("cluas_", "")
    demo_path = args.demos or (args.data + ".demos.json")
    with open(demo_path, encoding="utf-8") as f:
        demos = [(d["question"], d["answer"]) for d in json.load(f)][:args.shots]
    print(f"[demos] {len(demos)} text-only Ceist->Freagra shots", flush=True)

    if args.baseline is None:
        import glob
        snaps = glob.glob(SNAPSHOT_GLOB)
        if not snaps:
            raise SystemExit("no local Qwen2.5-Omni-3B snapshot; pass --baseline")
        args.baseline = snaps[0]

    clips = load_clips(args.data, args.n_clips, year=(args.year or None))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(args.baseline)

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    def dump(label, per_cond):
        for cond, ans in per_cond.items():
            p = os.path.join(args.out_dir, f"cluas_{tag}_{label}_{cond}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(ans, f, ensure_ascii=False, indent=2)
            print(f"  wrote {p} ({len(ans)} answers)", flush=True)

    # ---- baseline (or the single model, if this invocation is raw for a ckpt) ----
    if not args.checkpoints or args.prompt_format == "chat":
        from generate_compare import load_baseline
        print("[baseline] loading...", flush=True)
        thinker = load_baseline(args.baseline, device, dtype, attn="eager")
        per_cond, rows = run_model(thinker, proc, clips, demos, args.conditions,
                                   device, "baseline", args.prompt_format, **gen_kw)
        dump("baseline", per_cond)
        all_rows += rows
        del thinker
        torch.cuda.empty_cache()

    if args.checkpoints:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
        from qomhra.checkpoint import load_for_eval
        for i, ckpt in enumerate(args.checkpoints):
            label = args.labels[i] if i < len(args.labels) else os.path.basename(ckpt)
            print(f"[{label}] loading {ckpt} ...", flush=True)
            model, _ = load_for_eval(ckpt, device=device, dtype=dtype)
            per_cond, rows = run_model(model.thinker, proc, clips, demos,
                                       args.conditions, device, label, args.prompt_format,
                                       **gen_kw)
            dump(label, per_cond)
            all_rows += rows
            del model
            torch.cuda.empty_cache()

    combined = os.path.join(args.out_dir, f"cluas_{tag}_{args.prompt_format}_rows.json")
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {combined} ({len(all_rows)} rows)", flush=True)

    print("\n" + "=" * 72 + "\nSAMPLE ANSWERS\n" + "=" * 72)
    seen = 0
    for r in all_rows:
        if r["condition"] == "just_audio":
            print(f"[{r['model']}] {r['question_id']}: {r['question']}")
            print(f"    -> {r['answer']!r}")
            seen += 1
            if seen >= 8:
                break


if __name__ == "__main__":
    main()
