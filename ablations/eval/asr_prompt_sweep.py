#!/usr/bin/env python3
"""Find an instruction that forces VERBATIM transcription, not an answer.

audio_probe.py asked the model to transcribe and it answered the question instead —
byte-identical output for the transcribe and answer tasks on the English clip. Without
a working ASR prompt there is no WER, and WER is the only hard number the eval has.

Two candidate causes, swept independently here:
  order   - Qwen's own docs put the TEXT item before the AUDIO item; we had audio first.
  wording - "Return only the transcription." is an explicit output-format constraint;
            "transcribe ... word for word" merely describes the task.

The English clip is the discriminator: its reference is a question, so answering
('Dublin') and transcribing ('What is the capital city of Ireland?') are trivially
distinguishable. A variant passes only if English output matches the reference rather
than answering it. Irish is reported alongside but cannot adjudicate — we already know
the encoder garbles it.
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
    ("en", "en_capital_16k.wav", "What is the capital city of Ireland?"),
    ("ga", "ga_capital_16k.wav", "Cad é príomhchathair na hÉireann?"),
]

INSTRUCTIONS = [
    ("doc", "Transcribe this audio to text. Return only the transcription."),
    ("terse", "Repeat the words in the audio exactly. Do not answer them."),
    ("asr", "You are a speech recognition system. Output the exact words spoken "
            "in the audio and nothing else."),
]
ORDERS = ["text_first", "audio_first"]


def read_wav(path):
    import wave
    w = wave.open(path)
    raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--audio-dir", default=".")
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
    eos_ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    print("[load] OK", flush=True)

    for lang, fname, ref in CLIPS:
        path = os.path.join(args.audio_dir, fname)
        wav = read_wav(path)
        print("\n" + "=" * 74)
        print(f"[{lang}] reference: {ref!r}")
        print("=" * 74, flush=True)
        for iname, instr in INSTRUCTIONS:
            for order in ORDERS:
                a = {"type": "audio", "audio": path}
                t = {"type": "text", "text": instr}
                content = [t, a] if order == "text_first" else [a, t]
                conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                        {"role": "user", "content": content}]
                text = proc.apply_chat_template(conv, add_generation_prompt=True,
                                                tokenize=False)
                inputs = proc(text=text, audio=[wav], sampling_rate=16000,
                              return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    out = thinker.generate(**inputs, max_new_tokens=48, do_sample=False,
                                           eos_token_id=eos_ids,
                                           pad_token_id=tok.pad_token_id)
                new = out[0][inputs["input_ids"].shape[1]:]
                got = tok.decode(new, skip_special_tokens=True)
                print(f"\n  {iname:6s} {order:12s} -> {got!r}", flush=True)
    print("\n" + "=" * 74, flush=True)


if __name__ == "__main__":
    main()
