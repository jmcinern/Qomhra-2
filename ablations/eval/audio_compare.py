#!/usr/bin/env python3
"""Audio-input before/after across the ablation checkpoints: ASR (WER) + answer.

Completes the audio row of the eval matrix. Baseline numbers (see asr_baseline.py):
Irish WER 105.0%, English WER 0.0% — English is the CONTROL, so an Irish-only change
here is Irish-specific rather than the rig moving.

Expectation-setting, because a naive read of this table will mislead: ablations 2/3
train audio with a next-audio-frame REGRESSION objective. They never see audio->text.
A large ASR gain from them would be surprising; "unchanged" or "degraded" are the
plausible outcomes, and degradation is itself a result (the shared thinker being pulled
off its text distribution by the audio path). Ablation 4 is the one built to move WER,
because it is the only one that aligns speech to transcripts. This script exists partly
to check ablation 4 starts from an undamaged base.

  python -m eval.audio_compare --checkpoints <run>/checkpoints/step_000250 ...
"""
import argparse
import glob
import json
import os
import sys
import wave

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_baseline import INSTR, ROOT, SYS, JSONL, strip_boilerplate, wer  # noqa: E402

SNAPSHOT_GLOB = ("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")
# The TTS'd question pair: not an ASR test (asking a question lets the model answer
# instead of transcribe, and both tasks then return byte-identical output). This is
# the comprehension/demo cell.
QUESTIONS = [
    ("ga", "ga_capital_16k.wav", "Cad é príomhchathair na hÉireann?"),
    ("en", "en_capital_16k.wav", "What is the capital city of Ireland?"),
]
ANSWER_INSTR = "Listen to the question in the audio and answer it."


def read_wav(path):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1, f"need 16k mono: {path}"
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16
                         ).astype(np.float32) / 32768.0


@torch.no_grad()
def ask(thinker, proc, wav, path, instr, device, max_new_tokens=100):
    tok = proc.tokenizer
    eos_ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "text", "text": instr},
                                         {"type": "audio", "audio": path}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=[wav], sampling_rate=16000,
                  return_tensors="pt", padding=True).to(device)
    out = thinker.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def check_not_nan(label, text):
    """A NaN forward decodes to argmax=token 0 = '!'. That is not a crash: it scores a
    plausible ~100% WER and reads as 'the model can't do Irish'. This exact failure
    silently invalidated a whole baseline column (ROCm SDPA + masked audio batch), so
    fail loudly instead of reporting a number that means nothing."""
    body = text.strip().replace("!", "")
    if text.strip() and not body:
        raise SystemExit(
            f"[{label}] decoded to all '!' — NaN logits, almost certainly ROCm SDPA on a "
            f"masked audio batch. Load this model with attn_implementation='eager'. "
            f"Refusing to score it.")


def evaluate(thinker, proc, asr_rows, device, audio_dir, label="model"):
    res = {"asr": [], "answers": []}
    tot_e = tot_n = 0
    for r in asr_rows:
        path = os.path.join(ROOT, r["audio"])
        raw = ask(thinker, proc, read_wav(path), path, INSTR, device)
        check_not_nan(label, raw)
        # Unstripped, the chat wrapper scores as insertions: a PERFECT English
        # transcription measured 200%. Score the transcription, not the chatter.
        hyp = strip_boilerplate(raw)
        e, n = wer(r["text"], hyp)
        tot_e, tot_n = tot_e + e, tot_n + n
        res["asr"].append({"ref": r["text"], "hyp": hyp, "wer": 100 * e / max(n, 1)})
    res["wer"] = 100 * tot_e / max(tot_n, 1)
    for lang, fname, ref in QUESTIONS:
        path = os.path.join(audio_dir, fname)
        res["answers"].append({"lang": lang, "ref": ref,
                               "out": ask(thinker, proc, read_wav(path), path,
                                          ANSWER_INSTR, device)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--audio-dir", default=os.path.join(os.path.dirname(__file__),
                                                        "..", "Before_After", "input"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--heldout", action="store_true",
                    help="score ONLY utterances excluded from aligned training")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.baseline is None:
        args.baseline = glob.glob(SNAPSHOT_GLOB)[0]

    # Same utterances for every model, so the comparison is like-for-like.
    rows = [json.loads(l) for l in open(JSONL)]
    asr_rows = [r for r in rows if r["duration_s"] <= args.max_seconds]
    if args.heldout:
        # The aligned run trains on ~1 epoch of this same file, so without this the
        # ASR score is measuring memorisation. Same md5 rule the dataset excludes by.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
        from qomhra.data import AlignedClipDataset
        asr_rows = [r for r in asr_rows if AlignedClipDataset.is_heldout(r["audio"])]
    asr_rows = asr_rows[:args.n]
    print(f"[data] {len(asr_rows)} Irish ASR utterances"
          f"{' (HELD OUT of training)' if args.heldout else ' (may include training data)'}",
          flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(args.baseline)

    results = {}
    from generate_compare import load_baseline
    print("[baseline] loading...", flush=True)
    # eager, NOT sdpa: audio makes ragged/masked batches and ROCm's SDPA returns NaN on
    # those. A NaN baseline decodes to a wall of '!' (token 0) and still scores ~100%
    # WER — it looks like a real "baseline can't do Irish" result. Checkpoints escape
    # this via model.py::_force_eager_attention; the baseline path has no such guard.
    thinker = load_baseline(args.baseline, device, dtype, attn="eager")
    results["baseline"] = evaluate(thinker, proc, asr_rows, device, args.audio_dir,
                                   label="baseline")
    print(f"[baseline] Irish WER = {results['baseline']['wer']:.1f}%", flush=True)
    del thinker
    torch.cuda.empty_cache()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from qomhra.checkpoint import load_for_eval
    for i, ckpt in enumerate(args.checkpoints):
        label = args.labels[i] if i < len(args.labels) else os.path.basename(ckpt)
        print(f"[{label}] loading {ckpt} ...", flush=True)
        model, _ = load_for_eval(ckpt, device=device, dtype=dtype)
        results[label] = evaluate(model.thinker, proc, asr_rows, device, args.audio_dir,
                                  label=label)
        print(f"[{label}] Irish WER = {results[label]['wer']:.1f}%", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("IRISH ASR (WER, lower is better; baseline English control = 0.0%)")
    print("=" * 72)
    for label, r in results.items():
        print(f"  {label:24s} {r['wer']:6.1f}%")

    print("\n" + "=" * 72)
    print("AUDIO -> ANSWER")
    print("=" * 72)
    for label, r in results.items():
        print(f"\n--- {label}")
        for a in r["answers"]:
            print(f"  [{a['lang']}] {a['ref']}\n      -> {a['out'][:160]!r}")

    print("\n" + "=" * 72)
    print("SAMPLE TRANSCRIPTIONS")
    print("=" * 72)
    for j in range(min(4, len(asr_rows))):
        print(f"\nREF: {results['baseline']['asr'][j]['ref']!r}")
        for label, r in results.items():
            print(f"  {label:24s} {r['asr'][j]['wer']:6.1f}%  {r['asr'][j]['hyp']!r}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
