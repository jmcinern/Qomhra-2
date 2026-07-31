#!/usr/bin/env python3
"""CLUAS (LC Aural) listening QA — zero-shot, unit-in, no post-processing.

Rebuild of `cluas_qa.py` under the rerun contract. Three things changed:

  1. Zero-shot. The old script defaulted to three text-only Ceist->Freagra demos.
     Measured on the 250h model, three-shot gave identical_output_rate 0.96 between
     correct and mismatched audio -- it disconnects the input rather than helping.
  2. Unit-in. The old script fed <|audio_bos|><|AUDIO|><|audio_eos|> through the Qwen
     audio tower, which no discrete ablation trains. Speech now enters as mHuBERT unit
     ids in the expanded vocabulary, exactly as in training.
  3. No post-processing. `strip_boilerplate()` and the `first_answer()` marker regex are
     both gone. What the model emits is what gets scored.

Clips are 40-93 s = 1,500-3,400 dedup units, past the 1024 context every ablation trains
at, so `just_audio` windows each clip and asks the question once per window. Every window
answer is kept in the rows file; the judge file takes one per question via --select.

Conditions, unchanged in meaning from the original:
  just_audio       unit window + question   (listening)
  no_context       question only            (blind control)
  just_transcript  gold transcript + question (text ceiling)

  python -m eval.cluas_qa_units --data data/cluas_all.parquet \
      --units data/cluas_all.units.parquet --checkpoint <ckpt> --label asr250h \
      --n-questions 10 --out-dir output
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# MIOpen on the LUMI rocm6.2 container cannot build its conv kernel DBs, so every conv in
# the audio tower errors out. Only --speech-input tower hits a conv, but this must be set
# before the first one, and it is inert for the unit path. (See iwslt_st.py.)
torch.backends.cudnn.enabled = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "train")))

from discrete_eval_common import (  # noqa: E402
    AUDIO_MARKERS,
    SPEECH_END_TEXT,
    SPEECH_START_TEXT,
    TRANSCRIPT_START,
    TRANSCRIPT_START_TEXT,
    base_snapshot,
    chunk_units,
    decode,
    decode_prepared,
    print_smoke,
    reference_token_budget,
    smoke_gate,
    speech_span,
    units_budget,
)

CONDITIONS = ("just_audio", "no_context", "just_transcript")

# Continuous-audio path, for models whose audio tower is intact: the superseded
# ablations, and the stock base. No discrete ablation trains this tower, so a
# discrete checkpoint can never be scored through it -- but base can only be scored
# through it, since it has no embedding for the unit ids. Markers and literal
# sentinel spellings are shared with the other evaluators in discrete_eval_common.
TARGET_SR = 16000
AUDIO_WINDOW_S = 30.0  # Qwen2.5-Omni audio encoder window.
INSTR_GA = ("Éist leis an taifead agus freagair an cheist seo a leanas i nGaeilge. "
            "Tabhair an freagra amháin.")
TEXT_EOS = "<|endoftext|>"
CHAT_EOS = "<|im_end|>"


def rubric_answer_candidates(rubric, question_text=""):
    """Extract answer-bearing rubric clauses solely to size the decode budget.

    The candidates never enter the prompt or scored hypothesis.  CLUAS rubrics
    conventionally place an accepted answer immediately before ``= N mharc``.
    """
    if not rubric:
        return []
    text = str(rubric)
    score_markers = list(re.finditer(
        r"=\s*\d+\s*mharc", text, flags=re.IGNORECASE
    ))
    candidates = []
    cursor = 0
    for marker in score_markers:
        candidate = text[cursor:marker.start()]
        cursor = marker.end()
        # The first rubric clause also carries the question.  Answers themselves
        # are declarative, so the final question mark is a reliable boundary even
        # when PDF extraction has joined the question and answer on one line.
        candidate = candidate.rsplit("?", 1)[-1]
        candidate = re.sub(r"\s+", " ", candidate).strip()
        normalized_question = re.sub(r"\s+", " ", question_text).strip()
        if normalized_question and candidate.startswith(normalized_question):
            candidate = candidate[len(normalized_question):].strip()
        # Some extracted rubrics put a standalone total ("2 mharc + 2 mharc")
        # between an instruction and the first accepted answer.
        embedded_totals = list(re.finditer(
            r"\b\d+\s*mharc(?:\s*\+\s*\d+\s*mharc)*\b",
            candidate,
            flags=re.IGNORECASE,
        ))
        if embedded_totals:
            candidate = candidate[embedded_totals[-1].end():].strip()
        candidate = re.sub(
            r"^(?:[\-\u2022]\s*|\d+[.)]\s*)", "", candidate
        ).strip()
        if candidate:
            candidates.append(candidate)
    return candidates


def chunk_audio(wav, sr=TARGET_SR, win=AUDIO_WINDOW_S, overlap=1.0):
    """Split a clip into <= win-second windows, one <|AUDIO|> segment each.

    A trailing sliver shorter than the overlap is already contained in the previous
    window and has too few frames for the encoder's pooler, so it is dropped.
    """
    width = int(win * sr)
    hop = int((win - overlap) * sr)
    if len(wav) <= width:
        return [wav]
    chunks = [wav[i:i + width] for i in range(0, len(wav), hop) if i < len(wav)]
    kept = [c for c in chunks if len(c) >= int(overlap * sr)]
    return kept or chunks[:1]


def demo_ids(tokenizer, demos, eos_id):
    """Text-only Ceist->Freagra demonstrations, each ending in the TRAINED end token.

    Few-shot is off by default and is not part of the scored contract. It exists to test
    one specific thing: whether an explicit, terminated example teaches the model where to
    stop, which is the run-on failure seen on conversation items. The end token is emitted
    as an id rather than as text so the demonstration ends in exactly the token generation
    is told to stop on -- if the two disagree, the demo teaches the wrong boundary.

    Few-shot has a measured hazard here: on the 250h model it produced 0.96 identical
    output between correct and mismatched audio, i.e. it disconnected the input rather
    than helping. Always run --mismatch-control alongside it.
    """
    ids = []
    for question, answer in demos:
        ids += tokenizer.encode(f"Ceist: {question}\nFreagra: {answer}",
                                add_special_tokens=False) + [eos_id]
    return ids


def build_prompt(tokenizer, condition, question, units=None, transcript=None,
                 answer_cue="ceist", demo_prefix=()):
    """One prompt. `demo_prefix` is empty unless --shots was passed.

    `answer_cue` selects what tells the model to start answering. `transcript_start` is
    the sentinel the discrete-ASR runs were trained on, but it means "transcribe now",
    not "answer now"; `ceist` uses the Irish Freagra: label the exam itself uses. Neither
    is native to a QA task -- no ablation trains one -- so the choice is recorded in the
    run metadata and both are available to the smoke.
    """
    if condition == "just_audio":
        prefix = speech_span(units)
    elif condition == "just_transcript":
        prefix = tokenizer.encode(f"Tras-scríbhinn: {transcript}\n",
                                  add_special_tokens=False)
    elif condition == "no_context":
        prefix = []
    else:
        raise ValueError(f"unknown condition: {condition}")

    # Demonstrations precede the context so the query keeps the position nearest the
    # answer, matching how the demos themselves are laid out.
    prefix = list(demo_prefix) + prefix
    if answer_cue == "transcript_start":
        return prefix + tokenizer.encode(f"Ceist: {question}\n",
                                         add_special_tokens=False) + [TRANSCRIPT_START]
    return prefix + tokenizer.encode(f"Ceist: {question}\nFreagra:",
                                     add_special_tokens=False)


def build_tower_text(question, condition, transcript=None, answer_cue="ceist",
                     demos=(), eos_token=""):
    """The ablations' raw prompt, with the waveform occupying the unit slot.

    Mirrors build_prompt() clause for clause: same delimiters, same labels, same
    order, no extra whitespace. The only substitution is inside the speech span --
    base cannot ingest unit ids, so the tower's audio markers sit exactly where
    speech_span() puts the units, and the two sentinels go in as literal text.

    Demos are text here, so the end token goes in as text too.
    """
    if condition == "just_audio":
        prefix = SPEECH_START_TEXT + AUDIO_MARKERS + SPEECH_END_TEXT
    elif condition == "just_transcript":
        prefix = f"Tras-scríbhinn: {transcript}\n"
    elif condition == "no_context":
        prefix = ""
    else:
        raise ValueError(f"unknown condition: {condition}")
    head = "".join(f"Ceist: {q}\nFreagra: {a}{eos_token}" for q, a in demos)
    head += prefix
    if answer_cue == "transcript_start":
        return head + f"Ceist: {question}\n" + TRANSCRIPT_START_TEXT
    return head + f"Ceist: {question}\nFreagra:"


def build_tower_chat(processor, question, condition, transcript=None, demos=()):
    """Native Qwen chat prompt used only by the stock audio-tower control."""
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": INSTR_GA}]}
    ]
    for demo_question, demo_answer in demos:
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": f"Ceist: {demo_question}"}],
        })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": demo_answer}],
        })
    content = []
    if condition == "just_audio":
        content.extend([{"type": "audio", "audio": "c"}])
        content.append({"type": "text", "text": f"Ceist: {question}"})
    elif condition == "just_transcript":
        content.append({
            "type": "text",
            "text": f"Tras-scríbhinn: {transcript}\nCeist: {question}",
        })
    else:
        content.append({"type": "text", "text": f"Ceist: {question}"})
    conversation.append({"role": "user", "content": content})
    return processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )


def stratified_questions(clips, n_questions, seed):
    """Select a deterministic all-year question subset without changing source order."""
    if not n_questions:
        return clips
    by_year = {}
    for clip in clips:
        year = int(clip.get("year") or 0)
        for question in clip["questions"]:
            by_year.setdefault(year, []).append(
                (str(question["question_id"]), clip["snippet_id"])
            )
    years = sorted(by_year)
    if n_questions < len(years):
        years = years[:n_questions]
    base, remainder = divmod(n_questions, len(years))
    selected = set()
    for year_index, year in enumerate(years):
        take = min(len(by_year[year]), base + (year_index < remainder))
        ranked = sorted(
            by_year[year],
            key=lambda item: hashlib.sha256(
                f"{seed}:{year}:{item[0]}:{item[1]}".encode()
            ).digest(),
        )
        selected.update(question_id for question_id, _ in ranked[:take])
    result = []
    for clip in clips:
        kept = [q for q in clip["questions"] if str(q["question_id"]) in selected]
        if kept:
            result.append({**clip, "questions": kept})
    count = sum(len(clip["questions"]) for clip in result)
    if count != min(n_questions, sum(len(v) for v in by_year.values())):
        raise RuntimeError(f"stratified selection requested {n_questions}, got {count}")
    print(f"[data] stratified {count} questions across {len(years)} years "
          f"(seed={seed})", flush=True)
    return result


def load_clips(data_path, units_path, year=None, n_clips=0, with_audio=False,
               n_questions=0, selection_seed=2137):
    import pyarrow.parquet as pq

    columns = ["snippet_id", "duration_s", "gold_transcript", "questions_json"]
    if with_audio:
        columns.append("audio")
    # cluas_all.parquet carries `year`; the single-year files do not.
    if "year" in pq.read_schema(data_path).names:
        columns.append("year")
    records = pq.read_table(data_path, columns=columns).to_pylist()
    if year:
        records = [r for r in records if r.get("year") == year]
        if not records:
            raise SystemExit(f"no clips for year {year} in {data_path}")
    records.sort(key=lambda r: str(r["snippet_id"]))
    if n_clips:
        records = records[:n_clips]

    units_table = pq.read_table(units_path, columns=["key", "units"])
    store = {k.as_py(): np.asarray(u.as_py(), dtype=np.int64)
             for k, u in zip(units_table["key"], units_table["units"])}

    clips = []
    for record in records:
        key = str(record["snippet_id"])
        if key not in store:
            raise SystemExit(f"no units for clip {key}; run prepare_eval_units.py")
        clip = {
            "snippet_id": key,
            "duration_s": record["duration_s"],
            "transcript": record["gold_transcript"],
            "units": store[key],
            "questions": json.loads(record["questions_json"]),
            "year": record.get("year"),
        }
        if with_audio:
            from iwslt_st import decode_audio
            clip["wav"] = decode_audio(record["audio"])
        clips.append(clip)
    clips = stratified_questions(clips, n_questions, selection_seed)
    total_q = sum(len(c["questions"]) for c in clips)
    lengths = [len(c["units"]) for c in clips]
    print(f"[data] {len(clips)} clips, {total_q} questions; dedup units per clip "
          f"min {min(lengths)} median {int(np.median(lengths))} max {max(lengths)}",
          flush=True)
    return clips


def run(thinker, tokenizer, clips, conditions, eos_id, args, processor=None,
        demos=()):
    tower = args.speech_input == "tower"
    primary_eos_id = eos_id[0] if isinstance(eos_id, (list, tuple)) else eos_id
    prefix = () if tower or not demos else demo_ids(
        tokenizer, demos, primary_eos_id)
    rows = []
    started = time.time()
    for position, clip in enumerate(clips):
        # The audio-dependence control: the same question against a DIFFERENT clip's
        # speech. If few-shot has disconnected the input, the two answers match.
        other = clips[(position + 1) % len(clips)] if len(clips) > 1 else None
        for question in clip["questions"]:
            question_text = question["text_ga"]
            rubric_candidates = rubric_answer_candidates(
                question.get("rubric_ga", ""), question_text
            )
            if rubric_candidates:
                candidate_budgets = [
                    reference_token_budget(
                        tokenizer,
                        candidate,
                        multiplier=args.reference_budget_multiplier,
                        minimum=args.min_new_tokens,
                        hard_max=args.max_new_tokens,
                    )
                    for candidate in rubric_candidates
                ]
                reference_tokens, generation_budget = max(
                    candidate_budgets, key=lambda pair: pair[0]
                )
            else:
                reference_tokens = None
                generation_budget = args.max_new_tokens
            question_tokens = len(tokenizer.encode(question_text,
                                                   add_special_tokens=False))
            chunk_size = args.chunk_units or units_budget(
                args.seq_len, generation_budget, question_tokens + 16)

            for condition in conditions:
                if condition != "just_audio":
                    windows = [None]
                elif tower:
                    # Keep every native-audio prompt within the tower's 30 s window.
                    # One answer is decoded per chunk and selected by the same recorded
                    # rule used for unit windows.
                    windows = chunk_audio(clip["wav"])
                else:
                    windows = chunk_units(clip["units"], chunk_size)

                def decode_one(window, transcript=clip["transcript"],
                               condition=condition, question_text=question_text):
                    if tower:
                        if args.prompt_format == "chat":
                            text = build_tower_chat(
                                processor, question_text, condition,
                                transcript=transcript, demos=demos)
                        else:
                            text = build_tower_text(
                                question_text, condition,
                                transcript=transcript,
                                answer_cue=args.answer_cue,
                                demos=demos, eos_token=args.eos_token)
                        audios = [window] if condition == "just_audio" else None
                        inputs = processor(text=text, audio=audios,
                                           sampling_rate=TARGET_SR,
                                           return_tensors="pt", padding=True)
                        return decode_prepared(thinker, tokenizer,
                                               inputs.to(thinker.device), eos_id,
                                               generation_budget,
                                               stop_strings=args.stop_strings)
                    prompt = build_prompt(
                        tokenizer, condition, question_text, units=window,
                        transcript=transcript, answer_cue=args.answer_cue,
                        demo_prefix=prefix)
                    return decode(thinker, tokenizer, prompt, eos_id,
                                  generation_budget,
                                  stop_strings=args.stop_strings)

                for index, window in enumerate(windows):
                    result = decode_one(window)
                    if (args.mismatch_control and condition == "just_audio"
                            and other is not None):
                        if tower:
                            other_chunks = chunk_audio(other["wav"])
                            swap = other_chunks[min(index, len(other_chunks) - 1)]
                        else:
                            swap = chunk_units(other["units"], chunk_size)[
                                min(index, len(chunk_units(other["units"],
                                                           chunk_size)) - 1)]
                        mismatched = decode_one(swap)
                        result["mismatch_hypothesis"] = mismatched["hypothesis"]
                        result["mismatch_identical"] = (
                            mismatched["hypothesis"] == result["hypothesis"])
                    rows.append({
                        "item": f"{question['question_id']}/{condition}"
                                + (f"/w{index}" if condition == "just_audio" else ""),
                        "snippet_id": clip["snippet_id"],
                        "question_id": question["question_id"],
                        "question": question_text,
                        "rubric_answer_candidates": rubric_candidates,
                        "reference_tokens": reference_tokens,
                        "generation_budget": generation_budget,
                        "condition": condition,
                        "window": index,
                        "n_windows": len(windows),
                        # In tower mode `window` is a list of 30 s waveform segments,
                        # not units, so the two counts are reported separately.
                        "window_units": (0 if window is None or tower else len(window)),
                        "audio_chunks": (1 if tower and window is not None else 0),
                        **result,
                    })
                    if len(rows) % 25 == 0:
                        print(
                            f"[{args.label}] {len(rows)} decoded rows "
                            f"({(time.time() - started) / len(rows):.2f}s/row)",
                            flush=True,
                        )
    return rows


def _rank(row, select):
    """Higher is better. Ties fall back to window order, which is stable."""
    if select == "confidence":
        # The model's own mean log-probability for the answer it gave. An empty decode
        # cannot win: it has no tokens to be confident about.
        if not row["hypothesis"].strip() or row["mean_logprob"] is None:
            return float("-inf")
        return row["mean_logprob"]
    if select == "longest":
        return len(row["hypothesis"])
    return 0  # first


def judge_files(rows, select):
    """{condition: {question_id: answer}} — the shape the LC-Aural-Bench judge consumes.

    A windowed question yields one answer per window and the judge takes one. `confidence`
    picks the window the model was most sure of, which is the only rule here that uses
    evidence rather than position -- `first` always returns the opening of the clip
    regardless of where the answer lies, and `longest` rewards run-on.

    This ranks whole decodes against each other. It never edits one, so it stays inside
    the no-post-processing rule, and every window's output remains in the rows file.
    """
    per_condition = {}
    best = {}
    for row in rows:
        key = (row["condition"], row["question_id"])
        score = _rank(row, select)
        if key not in best or score > best[key]:
            best[key] = score
            per_condition.setdefault(row["condition"], {})[
                row["question_id"]] = row["hypothesis"]
    return per_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cluas_*.parquet")
    ap.add_argument("--units", required=True, help="prepare_eval_units.py pack output")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--n-clips", type=int, default=0, help="limit clips (0 = all)")
    ap.add_argument("--n-questions", type=int, default=0,
                    help="deterministic all-year stratified question subset")
    ap.add_argument("--selection-seed", type=int, default=2137)
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="trained context; sets the unit window when --chunk-units is 0")
    ap.add_argument("--chunk-units", type=int, default=0,
                    help="fixed unit window (0 = derive from --seq-len)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--reference-budget-multiplier", type=float, default=1.25)
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--answer-cue", choices=["ceist", "transcript_start"],
                    default="ceist")
    ap.add_argument("--speech-input", choices=["units", "tower"], default="units",
                    help="units = the rerun contract; tower = continuous audio through "
                         "the Qwen encoder, a diagnostic for superseded checkpoints "
                         "whose vocabulary was never expanded")
    ap.add_argument("--eos-token", default="<|im_end|>",
                    help="primary trained end token; discrete ASR used <|im_end|>")
    ap.add_argument("--additional-eos-token", action="append",
                    default=[TEXT_EOS],
                    help="also accept this trained document boundary (repeatable)")
    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="raw",
                    help="raw for discrete checkpoints; chat for stock tower baseline")
    ap.add_argument("--stop-strings", nargs="*", default=["Ceist:", "Freagra:"],
                    help="decoding-time stop boundary, checked alongside the trained "
                         "eos token; a model that has answered and starts a new "
                         "question halts there rather than running to the cap. Pass "
                         "no values to disable (--stop-strings with nothing after it).")
    ap.add_argument("--select", choices=["confidence", "first", "longest"],
                    default="confidence",
                    help="which window's answer reaches the judge file")
    ap.add_argument("--shots", type=int, default=0,
                    help="text-only Ceist->Freagra demos, each ending in the trained "
                         "end token; 0 = the zero-shot contract")
    ap.add_argument("--demos", default=None, help="<data>.demos.json (default: derive)")
    ap.add_argument("--mismatch-control", action="store_true",
                    help="re-run just_audio against another clip's speech and report "
                         "the identical-output rate; use whenever --shots > 0")
    ap.add_argument("--model-dir", default=None, help="tokenizer source (default: local base snapshot)")
    ap.add_argument("--baseline", action="store_true",
                    help="load stock Qwen2.5-Omni-3B (untrained on this contract) "
                         "instead of a discrete-rerun checkpoint; text-only conditions "
                         "only -- base was never given the expanded unit vocabulary")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    tower = args.speech_input == "tower"
    clips = load_clips(args.data, args.units, year=(args.year or None),
                       n_clips=args.n_clips,
                       with_audio=(tower and "just_audio" in args.conditions),
                       n_questions=args.n_questions,
                       selection_seed=args.selection_seed)

    from transformers import AutoTokenizer
    from qomhra.checkpoint import load_for_eval

    snapshot = args.model_dir or base_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    processor = None
    if tower:
        from transformers import Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(snapshot)
    # Both boundaries occur legitimately across the raw/chat contracts.  Accepting
    # either stops generation but preserves the exact emitted token in every row.
    eos_tokens = [args.eos_token] + args.additional_eos_token
    eos_tokens = list(dict.fromkeys(eos_tokens))
    eos_id = [tokenizer.convert_tokens_to_ids(token) for token in eos_tokens]
    if any(value is None or value == tokenizer.unk_token_id for value in eos_id):
        raise SystemExit(f"tokenizer does not know one of {eos_tokens!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[{args.label}] loading {args.checkpoint} ...", flush=True)
    if args.baseline:
        if not tower:
            raise SystemExit("--baseline requires --speech-input tower: base has "
                             "no embedding for the unit ids")
        from generate_compare import load_baseline
        thinker = load_baseline(args.checkpoint, device, dtype, attn="eager")
        model = type("M", (), {"thinker": thinker})()
    else:
        model, _ = load_for_eval(args.checkpoint, device=device, dtype=dtype)

    demos = []
    if args.shots:
        demo_path = args.demos or (args.data + ".demos.json")
        with open(demo_path, encoding="utf-8") as handle:
            demos = [(d["question"], d["answer"])
                     for d in json.load(handle)][:args.shots]
        print(f"[demos] {len(demos)} shots, each terminated with "
              f"{args.eos_token!r}", flush=True)
        if not args.mismatch_control:
            print("[warn] --shots without --mismatch-control: few-shot has been "
                  "measured to disconnect the audio (0.96 identical output on the "
                  "250h model) and this run cannot detect that", flush=True)

    rows = run(model.thinker, tokenizer, clips, args.conditions, eos_id, args,
               processor=processor, demos=demos)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.data))[0]
    metadata = {
        "eval": "cluas",
        "label": args.label,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "units": os.path.abspath(args.units),
        "shots": args.shots,
        "mismatch_control": args.mismatch_control,
        "post_processing": "none",
        "speech_input": ("mhubert units in expanded vocab" if not tower else
                         "continuous audio through the Qwen tower"),
        "prompt_format": args.prompt_format if tower else "raw",
        "answer_cue": args.answer_cue,
        "eos_token": args.eos_token,
        "accepted_eos_tokens": eos_tokens,
        "accepted_eos_ids": eos_id,
        "stop_strings": args.stop_strings,
        "seq_len": args.seq_len,
        "chunk_units": args.chunk_units or "derived from seq_len",
        "max_new_tokens": args.max_new_tokens,
        "generation_budget": (
            "per question: ceil(longest rubric answer tokens * "
            f"{args.reference_budget_multiplier}) + 1, minimum "
            f"{args.min_new_tokens}, hard ceiling {args.max_new_tokens}; "
            "falls back to the hard ceiling only when no rubric answer parses"
        ),
        "window_select": args.select,
        "do_sample": False,
        "device": device,
        "dtype": str(dtype),
    }
    rows_path = os.path.join(args.out_dir, f"{tag}_{args.label}_units_rows.json")
    with open(rows_path, "w", encoding="utf-8") as handle:
        json.dump({"run": metadata, "rows": rows}, handle, ensure_ascii=False, indent=2)

    for condition, answers in judge_files(rows, args.select).items():
        path = os.path.join(args.out_dir, f"{tag}_{args.label}_{condition}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(answers, handle, ensure_ascii=False, indent=2)
        print(f"  wrote {path} ({len(answers)} answers)", flush=True)
    print(f"  wrote {rows_path} ({len(rows)} rows)", flush=True)

    for condition in args.conditions:
        subset = [r for r in rows if r["condition"] == condition]
        if subset:
            print_smoke(f"cluas / {condition} / {args.label}",
                        smoke_gate(subset), subset)


if __name__ == "__main__":
    main()
