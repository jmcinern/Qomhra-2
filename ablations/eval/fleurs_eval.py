#!/usr/bin/env python3
"""FLEURS ga/en modality x language benchmark: base model or one CPT ablation checkpoint.

FLEURS is parallel at the sentence level, so the SAME sentences can be scored with the input
arriving as text or as speech, and with the output language matching the input or not. That
crossing is the point: it factorises where a model breaks. A model that transcribes Irish
speech but cannot translate it has an intact encoder and a broken language mapping; one that
translates Irish text but fails on Irish speech has the opposite. Each model gets a signature
across the conditions, not a single score.

  text_ga2en   Irish text   -> English text   translation ability, no audio (the CEILING for
                                              st_ga2en: speech can never beat it)
  text_en2ga   English text -> Irish text     translation, reverse direction
  asr_ga       Irish audio  -> Irish text     Irish transcription (cross-modal only)
  asr_en       English audio-> English text   English transcription — the control
  st_ga2en     Irish audio  -> English text   cross-modal AND cross-lingual
  st_en2ga     English audio-> Irish text     the reverse composition
  copy_ga      Irish text   -> Irish text     format control: separates "cannot do the task"
                                              from "cannot follow any instruction at all"

METRIC RULE (enforced by the CONDITIONS table, not by convention): transcription and the copy
control are scored with WER/CER, translation with chrF++/BLEU. The two families are never
placed side by side and never subtracted across, because a WER minus a chrF is a number that
looks meaningful and is not.

Transcription scores against FLEURS's own normalised transcription column, not a homegrown
normaliser: Irish orthography (lenition, eclipsis, the sineadh fada, apostrophes) makes
homegrown normalisation a silent source of error. WER above 100% is possible and is reported
as-is — a model emitting more wrong words than the reference has really done that.

Prompting follows the IWSLT/CLUAS contract: base and 10% checkpoints use the chat template,
final CPT checkpoints use raw <|endoftext|>-separated documents (they were pretrained into a
plain-document regime and babble under chat). 3-shot, greedy, batch 1, bf16, eager attention.
All conditions run inside ONE model load — loading is a large fraction of the cost.

  python fleurs_eval.py --data .../fleurs_parallel_test.parquet --label base \
      [--checkpoint <ckpt>] [--prompt-format raw] [--n 10] --out out.json
"""
import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

# MIOpen in the LUMI rocm6.2 container cannot build its conv kernel DBs, so the audio tower's
# conv1d dies. Route conv through PyTorch's native fallback. Must precede the first conv.
torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_baseline import SYS, norm, strip_boilerplate, wer  # noqa: E402
from audio_compare import check_not_nan  # noqa: E402
# Scorers imported, not re-instantiated: identical sacreBLEU settings to IWSLT (chrF++
# nc:6|nw:2|beta:2, BLEU tokenizer 13a) is what makes the two benchmarks comparable.
from iwslt_st import (AUDIO_MARKERS, BLEU_SCORER, CHRFPP_SCORER, RAW_STOP,  # noqa: E402
                      TARGET_SR, decode_audio)
from cluas_qa import chunk_audio  # noqa: E402

SNAPSHOT_GLOB = ("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")
RAW_BACKUP_STOPS = ("<|audio_bos|>", "<|AUDIO|>")
CHAT_STOP = "<|im_end|>"
CHAT_BACKUP_STOPS = ("<|im_start|>",)

# Derived, not inherited: measured on this parquet with the model's own tokeniser, the Irish
# references run to 157 tokens (mean 61.5, p99 118) — more than twice the English ones,
# because the tokeniser is not Irish-friendly. IWSLT's 64 would truncate almost every Irish
# output and manufacture a collapse that is really a cap. 256 is ~1.6x the longest reference.
MAX_NEW_TOKENS = 256

# ---------------------------------------------------------------------------
# The conditions. Each names its input column/modality, its reference column, and its metric
# family — so a condition cannot accidentally be scored with the wrong metric.
# ---------------------------------------------------------------------------
CONDITIONS = {
    "text_ga2en": dict(
        modality="text", in_col="ga_raw", ref_col="en_raw", metric="chrf",
        instr="Translate the following Irish text into English. "
              "Output only the English translation and nothing else."),
    "text_en2ga": dict(
        modality="text", in_col="en_raw", ref_col="ga_raw", metric="chrf",
        instr="Translate the following English text into Irish. "
              "Output only the Irish translation and nothing else."),
    "asr_ga": dict(
        modality="audio", in_col="ga_audio", ref_col="ga_norm", metric="wer",
        instr="Write down exactly the words spoken in the Irish audio. "
              "Output only the Irish transcription and nothing else."),
    "asr_en": dict(
        modality="audio", in_col="en_audio", ref_col="en_norm", metric="wer",
        instr="Write down exactly the words spoken in the English audio. "
              "Output only the English transcription and nothing else."),
    # Wording identical to iwslt_st.INSTR so the two benchmarks stay comparable.
    "st_ga2en": dict(
        modality="audio", in_col="ga_audio", ref_col="en_raw", metric="chrf",
        instr="Translate the spoken Irish in the audio into English. "
              "Output only the English translation and nothing else."),
    "st_en2ga": dict(
        modality="audio", in_col="en_audio", ref_col="ga_raw", metric="chrf",
        instr="Translate the spoken English in the audio into Irish. "
              "Output only the Irish translation and nothing else."),
    "copy_ga": dict(
        modality="text", in_col="ga_norm", ref_col="ga_norm", metric="wer",
        instr="Repeat the following Irish text exactly. "
              "Output only the text and nothing else."),
}
DEFAULT_CONDITIONS = ["text_ga2en", "text_en2ga", "asr_ga", "asr_en",
                      "st_ga2en", "st_en2ga", "copy_ga"]


# Any <|...|> run is a special token in this tokenizer's vocabulary. Matching the shape rather
# than a fixed list means a marker we did not think of still ends the answer. fleurs_rescore
# imports this so the translation and transcription pipelines cut at the same place.
SPECIAL = re.compile(r"<\|[^|>]{1,40}\|>")


def cut_at_special(text):
    """Everything from the first structural marker onwards is not the answer.

    Decoding stops only on the stop token the prompt format asked for, but the continued
    pretraining checkpoints were trained on documents separated by <|endoftext|> and emit that
    instead. The generation is then correct up to the marker and runaway text after it. The
    tokeniser's skip_special_tokens deletes the marker but keeps what follows, so the cut has
    to be made on the raw decode.
    """
    m = SPECIAL.search(text)
    return text[:m.start()] if m else text


def cer(ref, hyp, truncate=False):
    """Levenshtein over characters of the normalised strings. Returns (edits, n_ref).

    Mirrors asr_baseline.wer (same normalisation, same edit costs, same truncate flag) so WER
    and CER of one row are always measured on the same text.
    """
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0, 0
    if truncate:
        h = h[:len(r)]
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1], len(r)


# ---------------------------------------------------------------------------
# Prompt construction. One row = one (input, reference) pair for the active condition.
# ---------------------------------------------------------------------------
def as_chunks(spec, value):
    """Audio inputs become a LIST of <=30s windows; text inputs pass through untouched.

    All but a handful of FLEURS clips fit the Qwen2.5-Omni audio window in one piece. Three
    Irish recordings (of 343) run 30.7-39.7s, so rather than drop those sentences — which
    would remove them from all six conditions and shrink the parallel design — they are split
    the way CLUAS split its ~60s clips: one <|AUDIO|> segment per window, in order.
    """
    if spec["modality"] != "audio":
        return value
    return chunk_audio(value)


def build_raw(spec, demos, test_in, demo_sep="\n"):
    """Plain-document few-shot: what the CPT checkpoints were actually pretrained on.

    The task is carried by the demonstrations rather than an instruction — which is also why
    copy_ga is a meaningful control here: same shape, only the demo outputs differ.
    """
    audios = []
    parts = []
    for demo_in, demo_out in demos:
        if spec["modality"] == "audio":
            chunks = as_chunks(spec, demo_in)
            audios.extend(chunks)
            parts.append(f"{AUDIO_MARKERS * len(chunks)}\n{demo_out}{RAW_STOP}")
        else:
            parts.append(f"{demo_in}\n{demo_out}{RAW_STOP}")
    if spec["modality"] == "audio":
        chunks = as_chunks(spec, test_in)
        audios.extend(chunks)
        parts.append(f"{AUDIO_MARKERS * len(chunks)}\n")
    else:
        parts.append(f"{test_in}\n")
    return demo_sep.join(parts), audios


def build_chat(proc, spec, demos, test_in):
    """Instruction-tuned layout. Text item before audio item (established in asr_baseline)."""
    audios = []
    conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]}]

    def user_turn(value):
        content = [{"type": "text", "text": spec["instr"]}]
        if spec["modality"] == "audio":
            chunks = as_chunks(spec, value)
            audios.extend(chunks)
            content += [{"type": "audio", "audio": "x"} for _ in chunks]
        else:
            content.append({"type": "text", "text": value})
        return {"role": "user", "content": content}

    for demo_in, demo_out in demos:
        conv.append(user_turn(demo_in))
        conv.append({"role": "assistant", "content": [{"type": "text", "text": demo_out}]})
    conv.append(user_turn(test_in))
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    return text, audios


@torch.no_grad()
def generate(thinker, proc, spec, demos, test_in, device, prompt_format, max_new_tokens,
             demo_sep="\n"):
    """One generation, preserving how and where decoding stopped."""
    tok = proc.tokenizer
    if prompt_format == "raw":
        stop_tokens = (RAW_STOP,) + RAW_BACKUP_STOPS
        text, audios = build_raw(spec, demos, test_in, demo_sep)
    else:
        stop_tokens = (CHAT_STOP,) + CHAT_BACKUP_STOPS
        text, audios = build_chat(proc, spec, demos, test_in)

    stop_ids = {}
    for token in stop_tokens:
        token_id = tok.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tok.unk_token_id:
            stop_ids[token_id] = token
    if tok.convert_tokens_to_ids(stop_tokens[0]) not in stop_ids:
        raise ValueError(f"tokenizer does not recognise stop token {stop_tokens[0]!r}")

    inputs = proc(text=text, audio=(audios or None), sampling_rate=TARGET_SR,
                  return_tensors="pt", padding=True).to(device)
    out = thinker.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           eos_token_id=list(stop_ids), pad_token_id=tok.pad_token_id)
    generated = out[0][inputs["input_ids"].shape[1]:].tolist()
    last = generated[-1] if generated else None
    return {
        "text": tok.decode(generated, skip_special_tokens=True),
        "raw": tok.decode(generated, skip_special_tokens=False),
        "generated_tokens": len(generated),
        "stop_reason": stop_ids.get(
            last, "max_new_tokens" if len(generated) >= max_new_tokens else "unknown"),
    }


# ---------------------------------------------------------------------------
# One condition, end to end, with its scores baked in (never scored ad hoc afterwards).
# ---------------------------------------------------------------------------
def run_condition(thinker, proc, name, rows, demo_rows, device, label, prompt_format,
                  max_new_tokens, demo_sep="\n"):
    spec = CONDITIONS[name]
    demos = [(r[spec["in_col"]], r[spec["ref_col"]]) for r in demo_rows]
    out_rows, hyps, refs = [], [], []
    started = time.time()
    for index, r in enumerate(rows, start=1):
        gen = generate(thinker, proc, spec, demos, r[spec["in_col"]], device,
                       prompt_format, max_new_tokens, demo_sep)
        check_not_nan(f"{label}/{name}", gen["text"])
        hyp = strip_boilerplate(gen["text"])
        ref = r[spec["ref_col"]]
        hyps.append(hyp)
        refs.append(ref)
        record = {
            "id": r["id"], "condition": name, "ref": ref, "hyp": hyp, "raw": gen["raw"],
            "generated_tokens": gen["generated_tokens"], "stop_reason": gen["stop_reason"],
            "ga_gender": r["ga_gender"], "en_gender": r["en_gender"],
            "ga_duration_s": r["ga_duration_s"], "en_duration_s": r["en_duration_s"],
        }
        if spec["metric"] == "wer":
            w_edits, w_len = wer(ref, hyp)
            c_edits, c_len = cer(ref, hyp)
            # The reported figures apply two bounds in order. First cut at the first special
            # token, which recovers the checkpoints that stop in the right place but emit the
            # wrong marker. Then cut to the reference length, which bounds the score at 100
            # for the checkpoints that do not stop at all. Without them a silent model
            # outranks one that transcribes: WER over 500 is reachable and meaningless. The
            # unbounded figures stay in wer/cer so the size of the correction is on record.
            hyp_bounded = strip_boilerplate(cut_at_special(gen["raw"]))
            record["hyp_bounded"] = hyp_bounded
            wt_edits, _ = wer(ref, hyp_bounded, truncate=True)
            ct_edits, _ = cer(ref, hyp_bounded, truncate=True)
            record.update(wer=100.0 * w_edits / w_len if w_len else None,
                          cer=100.0 * c_edits / c_len if c_len else None,
                          wer_trunc=100.0 * wt_edits / w_len if w_len else None,
                          cer_trunc=100.0 * ct_edits / c_len if c_len else None,
                          wer_edits=w_edits, wer_ref_words=w_len,
                          cer_edits=c_edits, cer_ref_chars=c_len,
                          wer_trunc_edits=wt_edits, cer_trunc_edits=ct_edits)
        else:
            record["chrfpp"] = CHRFPP_SCORER.sentence_score(hyp, [ref]).score
        out_rows.append(record)
        if index % 50 == 0 or index == len(rows):
            print(f"[{label}/{name}] {index}/{len(rows)} "
                  f"({(time.time() - started) / index:.2f}s/row)", flush=True)

    stop_reasons = {}
    for r in out_rows:
        stop_reasons[r["stop_reason"]] = stop_reasons.get(r["stop_reason"], 0) + 1
    generated_total = sum(r["generated_tokens"] for r in out_rows)
    summary = {
        "condition": name, "metric": spec["metric"], "n": len(out_rows),
        "stop_reasons": stop_reasons,
        "cap_hit_rate": stop_reasons.get("max_new_tokens", 0) / len(out_rows) if out_rows else 0,
        "mean_generated_tokens": generated_total / len(out_rows) if out_rows else 0,
    }
    if spec["metric"] == "wer":
        # Corpus WER/CER = total edits / total reference length (the standard aggregate; not
        # the mean of per-row rates, which over-weights short sentences).
        summary["wer"] = 100.0 * sum(r["wer_edits"] for r in out_rows) / max(
            1, sum(r["wer_ref_words"] for r in out_rows))
        summary["cer"] = 100.0 * sum(r["cer_edits"] for r in out_rows) / max(
            1, sum(r["cer_ref_chars"] for r in out_rows))
        summary["wer_trunc"] = 100.0 * sum(r["wer_trunc_edits"] for r in out_rows) / max(
            1, sum(r["wer_ref_words"] for r in out_rows))
        summary["cer_trunc"] = 100.0 * sum(r["cer_trunc_edits"] for r in out_rows) / max(
            1, sum(r["cer_ref_chars"] for r in out_rows))
    else:
        summary["chrfpp"] = CHRFPP_SCORER.corpus_score(hyps, [refs]).score
        bleu = BLEU_SCORER.corpus_score(hyps, [refs])
        summary["bleu"] = bleu.score
        summary["bleu_bp"] = bleu.bp
        summary["hyp_len"] = bleu.sys_len
        summary["ref_len"] = bleu.ref_len
    return {"summary": summary, "rows": out_rows}


def load_data(parquet_path, shots, n, only_ids=None):
    import pyarrow.parquet as pq
    recs = pq.read_table(parquet_path).to_pylist()
    demo_rows = [r for r in recs if r["split"] == "demo"][:shots]
    test_rows = [r for r in recs if r["split"] == "test"]
    if only_ids:
        test_rows = [r for r in test_rows if r["id"] in set(only_ids)]
    if n:
        test_rows = test_rows[:n]
    demo_ids = {r["id"] for r in demo_rows}
    overlap = demo_ids & {r["id"] for r in test_rows}
    assert not overlap, f"demo sentences leaked into the scored set: {overlap}"
    # Audio is decoded once here and reused across every condition and every row.
    for r in demo_rows + test_rows:
        r["ga_audio"] = decode_audio(r["ga_audio"])
        r["en_audio"] = decode_audio(r["en_audio"])
    print(f"[data] {len(demo_rows)} demo sentences {sorted(demo_ids)}, "
          f"{len(test_rows)} scored sentences", flush=True)
    return demo_rows, test_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="fleurs_parallel_test.parquet")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--checkpoint", default=None, help="CPT checkpoint (omit for base)")
    ap.add_argument("--label", required=True, help="e.g. base, text_final, speech_pct10")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--n", type=int, default=0, help="limit scored sentences (0 = all)")
    ap.add_argument("--only-ids", nargs="*", type=int, default=None,
                    help="score only these FLEURS sentence ids (targeted checks)")
    ap.add_argument("--prompt-format", choices=["chat", "raw"], default="chat")
    ap.add_argument("--demo-sep", choices=["nl", "nlnl"], default="nl")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    unknown = [c for c in args.conditions if c not in CONDITIONS]
    if unknown:
        ap.error(f"unknown conditions: {unknown}")
    if args.baseline is None:
        import glob
        snaps = glob.glob(SNAPSHOT_GLOB)
        if not snaps:
            raise SystemExit("no local Qwen2.5-Omni-3B snapshot; pass --baseline")
        args.baseline = snaps[0]

    demo_rows, test_rows = load_data(args.data, args.shots, args.n, args.only_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(args.baseline)

    if args.checkpoint:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "train"))
        from qomhra.checkpoint import load_for_eval
        print(f"[{args.label}] loading {args.checkpoint} ...", flush=True)
        model, _ = load_for_eval(args.checkpoint, device=device, dtype=dtype)
        thinker = model.thinker
        model_path = args.checkpoint
    else:
        from generate_compare import load_baseline
        print(f"[{args.label}] loading base ...", flush=True)
        # eager, NOT sdpa: masked audio on ROCm SDPA returns NaNs and the model then decodes
        # a solid wall of '!'.
        thinker = load_baseline(args.baseline, device, dtype, attn="eager")
        model_path = args.baseline

    import sacrebleu
    results = {"run": {
        "label": args.label,
        "model_path": os.path.abspath(model_path),
        "data": os.path.abspath(args.data),
        "conditions": args.conditions,
        "shots": args.shots,
        "demo_ids": [r["id"] for r in demo_rows],
        "scored_n": len(test_rows),
        "prompt_format": args.prompt_format,
        "demo_sep": args.demo_sep if args.prompt_format == "raw" else None,
        "do_sample": False,
        "repetition_penalty": None,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": 1,
        "device": device,
        "dtype": str(dtype),
        "attention": ("eager" if not args.checkpoint else
                      "eager audio tower; decoder uses checkpoint configuration"),
        "rocm_gqa_in_sdpa": "disabled by qomhra.model for checkpoints",
        "target_sample_rate_hz": TARGET_SR,
        "stop_tokens": ([RAW_STOP] + list(RAW_BACKUP_STOPS) if args.prompt_format == "raw"
                        else [CHAT_STOP] + list(CHAT_BACKUP_STOPS)),
        "sacrebleu_version": sacrebleu.__version__,
        # chrF++/BLEU signatures are filled in after scoring: sacreBLEU refuses to produce a
        # signature until the metric has actually been used.
        "chrfpp_signature": None,
        "bleu_signature": None,
        "wer_cer_normalisation": "asr_baseline.norm (lowercase, strip punctuation, "
                                 "collapse whitespace) over FLEURS normalised column",
        "wer_cer_truncation": "wer_trunc/cer_trunc cut the raw decode at the first special "
                              "token and then cut the result to the reference length before "
                              "aligning, bounding the score at 100%; wer/cer are the "
                              "unbounded figures over the whole decode",
    }, "conditions": {}}

    demo_sep = {"nl": "\n", "nlnl": "\n\n"}[args.demo_sep]
    for name in args.conditions:
        results["conditions"][name] = run_condition(
            thinker, proc, name, test_rows, demo_rows, device, args.label,
            args.prompt_format, args.max_new_tokens, demo_sep)
        s = results["conditions"][name]["summary"]
        score = (f"WER {s['wer']:6.2f}  CER {s['cer']:6.2f}" if s["metric"] == "wer"
                 else f"chrF++ {s['chrfpp']:6.2f}  BLEU {s['bleu']:6.2f}")
        print(f"[{args.label}] {name:11s} {score}  cap-hit {100 * s['cap_hit_rate']:5.1f}%",
              flush=True)

    if any(CONDITIONS[c]["metric"] == "chrf" for c in args.conditions):
        results["run"]["chrfpp_signature"] = str(CHRFPP_SCORER.get_signature())
        results["run"]["bleu_signature"] = str(BLEU_SCORER.get_signature())

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    print("\n" + "=" * 78)
    print(f"FLEURS  {args.label}  ({args.shots}-shot {args.prompt_format}, "
          f"max_new={args.max_new_tokens}, n={len(test_rows)})")
    print("=" * 78)
    for name in args.conditions:
        s = results["conditions"][name]["summary"]
        score = (f"WER {s['wer']:7.2f}  CER {s['cer']:7.2f}" if s["metric"] == "wer"
                 else f"chrF++ {s['chrfpp']:7.2f}  BLEU {s['bleu']:7.2f}")
        print(f"  {name:11s} {score}   stops {s['stop_reasons']}")
    print("\nSAMPLES")
    for name in args.conditions:
        rows = results["conditions"][name]["rows"]
        if rows:
            print(f"\n[{name}] REF: {rows[0]['ref']!r}\n{'':12s}HYP: {rows[0]['hyp']!r}")


if __name__ == "__main__":
    main()
