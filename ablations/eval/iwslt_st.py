#!/usr/bin/env python3
"""IWSLT2023 ga->eng speech translation: base model (+ optional ablation checkpoints).

Irish speech in, English text out — so this is TRANSLATION, not ASR: the reference
language differs from the audio language and the metric is BLEU against dev.eng (not WER).
It is the first cell of the LUMI eval matrix over base + 4 ablations (text/speech/both/
aligned) and doubles as an instruction-following probe: the base model is instruct-tuned,
and few-shot chat that it handles well is the "before" reference for CPT degradation.

Prompting is model-dependent: base uses its instruction-tuned chat format, while CPT
checkpoints use raw <|endoftext|>-separated documents. For chat, the instruction-before-
audio ordering was established (
in asr_baseline). Demos are the head of the sorted dev set; the tail is scored — disjoint.

Data is a single parquet (see iwslt_to_parquet.py): audio bytes + eng ref + duration_s,
one file instead of hundreds of loose wavs (LUMI small-files quota).

  python -m eval.iwslt_st --data .../iwslt2023_dev.parquet --shots 3 \
      [--checkpoints <ckpt> ...] [--labels ...] --out output/iwslt_st_base.json
"""
import argparse
import io
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import sacrebleu
import torch
from sacrebleu.metrics import BLEU, CHRF

# ROCm/MIOpen on the LUMI rocm6.2 container cannot create its kernel DBs: it writes
# gfx90a6e.ukdb / *.ufdb.txt as DIRECTORIES (not files), so every conv returns
# miopenStatusInternalError and the audio tower's conv1d dies. Disabling the MIOpen path
# routes conv through PyTorch's native fallback — fine for eval (a handful of small convs
# per utterance). Must be set before the first conv. (The text-only evals never hit this.)
torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_baseline import SYS, strip_boilerplate  # noqa: E402
from audio_compare import check_not_nan  # noqa: E402

SNAPSHOT_GLOB = ("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")
# Text item before audio item; asks for translation, and for the translation ONLY (the
# chat wrapper otherwise scores as BLEU-killing insertions — same lesson as ASR).
INSTR = ("Translate the spoken Irish in the audio into English. "
         "Output only the English translation and nothing else.")
TARGET_SR = 16000
AUDIO_WINDOW_S = 30.0  # Qwen2.5-Omni audio encoder window.
MAX_NEW_TOKENS = 128

# Raw (non-chat) prompting. CPT checkpoints were pretrained on documents separated by
# <|endoftext|>, so evaluate them in that native format rather than with the chat template.
AUDIO_MARKERS = "<|audio_bos|><|AUDIO|><|audio_eos|>"
RAW_STOP = "<|endoftext|>"
RAW_BACKUP_STOPS = ("<|audio_bos|>", "<|AUDIO|>")
CHAT_STOP = "<|im_end|>"
CHAT_BACKUP_STOPS = ("<|im_start|>",)


# ---------------------------------------------------------------------------
# Audio: decode wav bytes -> 16k mono float32. IWSLT dev is 48k mono 16-bit, so we
# read PCM with the stdlib (no soundfile/librosa in the container) and resample with
# torchaudio (which is present) down to the 16k the Qwen feature extractor expects.
# ---------------------------------------------------------------------------
def decode_audio(raw_bytes):
    import wave
    w = wave.open(io.BytesIO(raw_bytes))
    sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    assert sw == 2, f"expected 16-bit PCM, got sampwidth={sw}"
    data = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            .astype(np.float32) / 32768.0)
    if ch == 2:                              # stereo -> mono
        data = data.reshape(-1, 2).mean(axis=1)
    if sr != TARGET_SR:
        data = resample(data, sr, TARGET_SR)
    return np.ascontiguousarray(data, dtype=np.float32)


def resample(data, sr_in, sr_out):
    import torchaudio.functional as AF
    t = torch.from_numpy(np.ascontiguousarray(data))
    return AF.resample(t, sr_in, sr_out).numpy()


BLEU_SCORER = BLEU(tokenize="13a")
CHRFPP_SCORER = CHRF(char_order=6, word_order=2, beta=2)


# ---------------------------------------------------------------------------
# Few-shot chat generation.
# ---------------------------------------------------------------------------
@torch.no_grad()
def translate(thinker, proc, demos, test_wav, device, max_new_tokens=MAX_NEW_TOKENS,
              prompt_format="chat", demo_sep="\n", audio_cache="none",
              demo_embedding_cache=None):
    """Generate one translation and preserve how/where decoding stopped."""
    tok = proc.tokenizer
    audios = []
    if prompt_format == "raw":
        stop_tokens = (RAW_STOP,) + RAW_BACKUP_STOPS
        parts = []
        for wav, ref in demos:
            parts.append(f"{AUDIO_MARKERS}\n{ref}{RAW_STOP}")
            audios.append(wav)
        parts.append(f"{AUDIO_MARKERS}\n")
        audios.append(test_wav)
        text = demo_sep.join(parts)
    else:
        stop_tokens = (CHAT_STOP,) + CHAT_BACKUP_STOPS
        conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]}]
        for wav, ref in demos:
            conv.append({"role": "user", "content": [
                {"type": "text", "text": INSTR}, {"type": "audio", "audio": "demo"}]})
            conv.append({"role": "assistant", "content": [{"type": "text", "text": ref}]})
            audios.append(wav)
        conv.append({"role": "user", "content": [
            {"type": "text", "text": INSTR}, {"type": "audio", "audio": "test"}]})
        audios.append(test_wav)
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    stop_ids = {}
    for token in stop_tokens:
        token_id = tok.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tok.unk_token_id:
            stop_ids[token_id] = token
    required_id = tok.convert_tokens_to_ids(stop_tokens[0])
    if required_id not in stop_ids:
        raise ValueError(f"tokenizer does not recognize required stop token {stop_tokens[0]!r}")

    inputs = proc(text=text, audio=audios, sampling_rate=TARGET_SR,
                  return_tensors="pt", padding=True).to(device)
    generation_inputs = dict(inputs)
    if audio_cache == "demo-embeddings":
        if demo_embedding_cache is None:
            raise ValueError("demo_embedding_cache is required for demo-embeddings mode")
        input_features = inputs["input_features"]
        feature_mask = inputs["feature_attention_mask"]
        demo_count = len(demos)
        if "embeddings" not in demo_embedding_cache:
            demo_embedding_cache["embeddings"] = thinker.get_audio_features(
                input_features[:demo_count],
                feature_attention_mask=feature_mask[:demo_count],
            ).detach()
        test_embeddings = thinker.get_audio_features(
            input_features[demo_count:],
            feature_attention_mask=feature_mask[demo_count:],
        )
        audio_embeddings = torch.cat(
            (demo_embedding_cache["embeddings"], test_embeddings), dim=0)
        inputs_embeds = thinker.get_input_embeddings()(inputs["input_ids"])
        _, _, audio_mask = thinker.get_placeholder_mask(
            inputs["input_ids"], inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_mask, audio_embeddings.to(inputs_embeds.device, inputs_embeds.dtype))
        generation_inputs.pop("input_features")
        generation_inputs["inputs_embeds"] = inputs_embeds

    out = thinker.generate(**generation_inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           eos_token_id=list(stop_ids), pad_token_id=tok.pad_token_id)
    generated_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    last_id = generated_ids[-1] if generated_ids else None
    stop_reason = stop_ids.get(
        last_id, "max_new_tokens" if len(generated_ids) >= max_new_tokens else "unknown")
    return {
        "text": tok.decode(generated_ids, skip_special_tokens=True),
        "raw": tok.decode(generated_ids, skip_special_tokens=False),
        "generated_token_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "stop_reason": stop_reason,
    }


def evaluate(thinker, proc, demos, tests, device, label="model", prompt_format="chat",
             demo_sep="\n", max_new_tokens=MAX_NEW_TOKENS, audio_cache="none"):
    hyps, refs, rows = [], [], []
    demo_embedding_cache = {}
    started = time.time()
    for index, r in enumerate(tests, start=1):
        generation = translate(
            thinker, proc, demos, r["wav"], device,
            max_new_tokens=max_new_tokens, prompt_format=prompt_format,
            demo_sep=demo_sep, audio_cache=audio_cache,
            demo_embedding_cache=demo_embedding_cache)
        check_not_nan(label, generation["text"])
        hyp = strip_boilerplate(generation["text"])
        hyps.append(hyp)
        refs.append(r["eng"])
        rows.append({
            "name": r["name"], "ref": r["eng"], "hyp": hyp,
            "raw": generation["raw"],
            "generated_token_ids": generation["generated_token_ids"],
            "generated_tokens": generation["generated_tokens"],
            "stop_reason": generation["stop_reason"],
        })
        if index % 25 == 0 or index == len(tests):
            elapsed = time.time() - started
            print(f"[{label}] generated {index}/{len(tests)} rows "
                  f"({elapsed / index:.2f}s/row)", flush=True)
    bleu = BLEU_SCORER.corpus_score(hyps, [refs])
    chrfpp = CHRFPP_SCORER.corpus_score(hyps, [refs])
    stop_reasons = {}
    for row in rows:
        reason = row["stop_reason"]
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    generated_total = sum(row["generated_tokens"] for row in rows)
    detail = {
        "sacrebleu_version": sacrebleu.__version__,
        "bleu_signature": str(BLEU_SCORER.get_signature()),
        "chrfpp_signature": str(CHRFPP_SCORER.get_signature()),
        "precisions": bleu.precisions,
        "bp": bleu.bp,
        "hyp_len": bleu.sys_len,
        "ref_len": bleu.ref_len,
        "chrfpp": chrfpp.score,
        "generation": {
            "stop_reasons": stop_reasons,
            "total_tokens": generated_total,
            "mean_tokens": generated_total / len(rows) if rows else 0.0,
            "max_tokens": max((row["generated_tokens"] for row in rows), default=0),
        },
    }
    return {"bleu": bleu.score, "chrfpp": chrfpp.score,
            "detail": detail, "rows": rows}


# ---------------------------------------------------------------------------
def load_split(parquet_path, shots, n, max_seconds):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path)
    recs = t.to_pylist()  # sorted at pack time; positional demo/test split
    demos_raw = recs[:shots]
    tests_raw = recs[shots:]
    long = [r for r in tests_raw if r["duration_s"] > max_seconds]
    tests_raw = [r for r in tests_raw if r["duration_s"] <= max_seconds]
    if n:
        tests_raw = tests_raw[:n]
    demos = [(decode_audio(r["audio"]), r["eng"]) for r in demos_raw]
    tests = [{"name": r["name"], "eng": r["eng"],
              "wav": decode_audio(r["audio"])} for r in tests_raw]
    print(f"[data] {len(demos)} demos, {len(tests)} test utts "
          f"(<= {max_seconds}s; {len(long)} longer skipped)", flush=True)
    return demos, tests


def run_metadata(args, label, model_path, device, dtype, test_count):
    """Configuration needed to reproduce one model's rows and aggregate scores."""
    return {
        "label": label,
        "model_path": os.path.abspath(model_path),
        "data": os.path.abspath(args.data),
        "test_count": test_count,
        "shots": args.shots,
        "prompt_format": args.prompt_format,
        "demo_sep": args.demo_sep if args.prompt_format == "raw" else None,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "audio_cache": args.audio_cache,
        "batch_size": 1,
        "device": device,
        "dtype": str(dtype),
        "attention": ("eager" if label == "baseline" else
                      "eager audio tower; decoder uses checkpoint configuration"),
        "rocm_gqa_in_sdpa": "disabled by qomhra.model for checkpoints",
        "stop_tokens": ([RAW_STOP] + list(RAW_BACKUP_STOPS) if
                        args.prompt_format == "raw" else
                        [CHAT_STOP] + list(CHAT_BACKUP_STOPS)),
        "target_sample_rate_hz": TARGET_SR,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="iwslt dev parquet (see iwslt_to_parquet.py)")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--n", type=int, default=0, help="limit test utts (0 = all)")
    ap.add_argument("--max-seconds", type=float, default=AUDIO_WINDOW_S)
    ap.add_argument("--prompt-format", choices=["chat", "raw"], default="chat",
                    help="chat for base; raw for CPT checkpoints")
    ap.add_argument("--demo-sep", choices=["nl", "nlnl"], default="nl",
                    help="raw only: inter-example separator")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip base generation in the separate raw CPT invocation; "
                         "the base result is produced by the chat invocation")
    ap.add_argument("--allow-checkpoint-chat", action="store_true",
                    help="allow a diagnostic CPT chat run outside the default contract")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--audio-cache", choices=["none", "demo-embeddings"], default="none",
                    help="experimental reuse of the fixed demonstrations' audio embeddings")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    demo_sep = {"nl": "\n", "nlnl": "\n\n"}[args.demo_sep]
    if args.prompt_format == "raw" and not args.skip_baseline:
        ap.error("raw prompting is for CPT checkpoints; pass --skip-baseline")
    if (args.checkpoints and args.prompt_format != "raw" and
            not args.allow_checkpoint_chat):
        ap.error("CPT checkpoint chat diagnostics require --allow-checkpoint-chat")
    if args.skip_baseline and not args.checkpoints:
        ap.error("--skip-baseline requires at least one checkpoint")
    if args.labels and len(args.labels) != len(args.checkpoints):
        ap.error("--labels must have exactly one label per checkpoint")
    if args.baseline is None:
        import glob
        snaps = glob.glob(SNAPSHOT_GLOB)
        if not snaps:
            raise SystemExit("no local Qwen2.5-Omni-3B snapshot; pass --baseline")
        args.baseline = snaps[0]

    demos, tests = load_split(args.data, args.shots, args.n, args.max_seconds)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(args.baseline)

    results = {}
    from generate_compare import load_baseline
    if not args.skip_baseline:
        print("[baseline] loading...", flush=True)
        # eager, NOT sdpa: masked audio on ROCm SDPA can return NaNs.
        thinker = load_baseline(args.baseline, device, dtype, attn="eager")
        results["baseline"] = evaluate(
            thinker, proc, demos, tests, device, "baseline",
            prompt_format=args.prompt_format, demo_sep=demo_sep,
            max_new_tokens=args.max_new_tokens, audio_cache=args.audio_cache)
        results["baseline"]["run"] = run_metadata(
            args, "baseline", args.baseline, device, dtype, len(tests))
        print(f"[baseline] chrF++ = {results['baseline']['chrfpp']:.2f}, "
              f"BLEU = {results['baseline']['bleu']:.2f}", flush=True)
        del thinker
        torch.cuda.empty_cache()

    if args.checkpoints:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
        from qomhra.checkpoint import load_for_eval
        for i, ckpt in enumerate(args.checkpoints):
            label = args.labels[i] if i < len(args.labels) else os.path.basename(ckpt)
            print(f"[{label}] loading {ckpt} ...", flush=True)
            model, _ = load_for_eval(ckpt, device=device, dtype=dtype)
            results[label] = evaluate(
                model.thinker, proc, demos, tests, device, label,
                prompt_format=args.prompt_format, demo_sep=demo_sep,
                max_new_tokens=args.max_new_tokens, audio_cache=args.audio_cache)
            results[label]["run"] = run_metadata(
                args, label, ckpt, device, dtype, len(tests))
            print(f"[{label}] chrF++ = {results[label]['chrfpp']:.2f}, "
                  f"BLEU = {results[label]['bleu']:.2f}", flush=True)
            del model
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    sep_note = f", sep={args.demo_sep}" if args.prompt_format == "raw" else ""
    print(f"IWSLT2023 ga->eng SPEECH TRANSLATION  "
          f"(chrF++ / BLEU, {args.shots}-shot {args.prompt_format}{sep_note}, "
          f"max_new={args.max_new_tokens})")
    print("=" * 72)
    for label, r in results.items():
        d = r["detail"]
        print(f"  {label:24s} chrF++ {r['chrfpp']:6.2f}  BLEU {r['bleu']:6.2f}  "
              f"(BP {d['bp']:.3f}, {d['hyp_len']}/{d['ref_len']} tok)")
        print(f"  {'':24s} stops {d['generation']['stop_reasons']}")

    print("\n" + "=" * 72 + "\nSAMPLE TRANSLATIONS\n" + "=" * 72)
    for j in range(min(4, len(tests))):
        first_result = next(iter(results.values()))
        print(f"\nREF: {first_result['rows'][j]['ref']!r}")
        for label, r in results.items():
            print(f"  {label:24s} {r['rows'][j]['hyp']!r}")

    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
