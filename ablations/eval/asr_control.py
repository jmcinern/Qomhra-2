#!/usr/bin/env python3
"""English ASR control + the boilerplate-stripping fix.

Two jobs:

1. CONTROL. Baseline Irish ASR scored WER 191.9%. That is only interpretable against
   English audio of the same kind — a STATEMENT, not a question (a question lets the
   model answer instead of transcribe, which confounded the earlier sweep). If English
   scores near 0 here, the Irish failure is Irish-specific rather than a broken harness.

2. STRIPPING. The model wraps transcriptions in chat boilerplate ('The exact words
   spoken in the audio are "...". If you have any other audio...'), and those words
   score as insertions. Unstripped, a model that merely becomes less chatty looks like
   it learned Irish. Stripping is therefore required for the before/after to mean
   anything. Prefer a quoted span when present, else drop known lead-ins/trailers.
"""
import argparse
import glob
import os
import re
import sys
import wave

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_baseline import SYS, INSTR, strip_boilerplate, wer  # noqa: E402


def read_wav(path):
    w = wave.open(path)
    raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


CLIPS = [
    ("en", "en_statement_16k.wav", "The capital city of Ireland is Dublin."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--audio-dir", default="../Before_After/input")
    args = ap.parse_args()
    if args.model is None:
        args.model = glob.glob("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                               "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import (AutoConfig, Qwen2_5OmniProcessor,
                              Qwen2_5OmniThinkerForConditionalGeneration)
    cfg = AutoConfig.from_pretrained(args.model)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, config=cfg.thinker_config, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    print("[load] OK\n", flush=True)

    for lang, fname, ref in CLIPS:
        path = os.path.join(args.audio_dir, fname)
        wav = read_wav(path)
        conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [
                    {"type": "text", "text": INSTR},
                    {"type": "audio", "audio": path}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[wav], sampling_rate=16000,
                      return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = thinker.generate(**inputs, max_new_tokens=100, do_sample=False,
                                   eos_token_id=eos_id, pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cleaned = strip_boilerplate(raw)
        e_raw, n = wer(ref, raw)
        e_cln, _ = wer(ref, cleaned)
        print("=" * 72)
        print(f"[{lang}] {fname} ({wav.size/16000:.2f}s)")
        print("=" * 72)
        print(f"  REF          : {ref!r}")
        print(f"  HYP (raw)    : {raw!r}")
        print(f"  HYP (cleaned): {cleaned!r}")
        print(f"  WER raw      : {100*e_raw/max(n,1):6.1f}%")
        print(f"  WER cleaned  : {100*e_cln/max(n,1):6.1f}%", flush=True)

    # Stripping must be verified on the actual Irish outputs it will be applied to,
    # not just on well-behaved English.
    print("\n" + "=" * 72)
    print("STRIPPER CHECK on real Irish baseline outputs")
    print("=" * 72)
    samples = [
        'The exact words spoken in the audio are "ready does the". If you have any '
        'other audio to transcribe or need help with something else, feel free to let me know.',
        'The exact words spoken in the audio are "My name doctor he also saw solution '
        'in the." If you have any other audio to transcribe or need further help, feel free.',
        'The brother.',
        "I'm sorry, I can't recognize the words in the audio. Maybe you could try again.",
    ]
    for s in samples:
        print(f"  {s[:58]!r:64s} -> {strip_boilerplate(s)!r}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
