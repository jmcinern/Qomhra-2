#!/usr/bin/env python3
"""Baseline audio-input check: transcribe + answer, Irish vs English.

Text probing showed the thinker answers 'what is the capital of Ireland' instantly in
English and collapses into an 'is Eirinn is Eirinn' loop in Irish. These are the SAME
question TTS'd into both languages, so the audio path either reproduces that asymmetry
(the encoder passes language through and the weakness is in the thinker) or it fails
in both (the weakness is the audio front-end). Two tasks separate the two failure modes:

  transcribe -> does the encoder hear Irish phonetics at all? (WER-able, no thinker
                reasoning needed)
  answer     -> does the thinker comprehend Irish that arrived via speech?

Transcribe failing but answer failing too => front-end. Transcribe fine, answer bad
=> thinker, same as text. That distinction decides whether ablation 4 has anything to
work with.

Qwen's exact default system prompt is required on the audio path (the model warns when
it is modified), so it is pinned verbatim below rather than left to the template.
"""
import argparse
import glob
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
       "capable of perceiving auditory and visual inputs, as well as generating text "
       "and speech.")

CLIPS = [
    ("ga", "ga_capital_16k.wav", "Cad é príomhchathair na hÉireann?"),
    ("en", "en_capital_16k.wav", "What is the capital city of Ireland?"),
]

# Transcribe instructions given in English for both clips: we are testing whether the
# ENCODER conveys Irish speech, not whether the thinker can follow an Irish instruction
# (which the text probe already showed it cannot). Keeps the two rows comparable.
TASKS = [
    ("transcribe", "Please transcribe the speech in this audio exactly, word for word."),
    ("answer", "Listen to the question in the audio and answer it."),
]


def read_wav(path):
    import wave
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16k mono"
    raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--audio-dir", default=".")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()
    if args.model is None:
        args.model = glob.glob("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                               "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import (AutoConfig, Qwen2_5OmniProcessor,
                              Qwen2_5OmniThinkerForConditionalGeneration)

    print(f"[load] {args.model}", flush=True)
    cfg = AutoConfig.from_pretrained(args.model)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, config=cfg.thinker_config, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    proc = Qwen2_5OmniProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    # generation_config.eos_token_id is None here; without this generate() never stops.
    eos_ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    print("[load] OK", flush=True)

    for lang, fname, ref in CLIPS:
        path = os.path.join(args.audio_dir, fname)
        wav = read_wav(path)
        print("\n" + "=" * 72)
        print(f"[{lang}] {fname}  ({wav.size / 16000:.2f}s)   reference: {ref!r}")
        print("=" * 72, flush=True)

        for task, instr in TASKS:
            conv = [
                {"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": path},
                    {"type": "text", "text": instr},
                ]},
            ]
            text = proc.apply_chat_template(conv, add_generation_prompt=True,
                                            tokenize=False)
            inputs = proc(text=text, audio=[wav], sampling_rate=16000,
                          return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = thinker.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, eos_token_id=eos_ids,
                                       pad_token_id=tok.pad_token_id)
            n_prompt = inputs["input_ids"].shape[1]
            new = out[0][n_prompt:]
            stopped = bool((new == eos_id).any())
            print(f"\n  {task:11s} -> {tok.decode(new, skip_special_tokens=True)!r}")
            print(f"  {'':11s}    ({new.shape[0]} tokens, terminated={stopped})",
                  flush=True)

    print("\n" + "=" * 72, flush=True)


if __name__ == "__main__":
    main()
