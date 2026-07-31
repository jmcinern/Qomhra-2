#!/usr/bin/env python3
"""Baseline Irish ASR: the 'before' number for ablation 4.

Uses held-out utterances from aligned_setup.jsonl, which ship their own reference
transcripts — so this is the one eval cell that yields a metric rather than a judgement.

Held-out split is by md5 of the audio path, NOT by row order: the file must stay stable
as rows are added, and training reads the same jsonl. Anything with md5 % 40 == 0 (~2.5%,
~200 utts) is eval-only and must be excluded from every training run.

Prompt is `asr` wording with the text item before the audio item — the prompt sweep
showed that framing produces an actual transcription attempt, while Qwen's own
documented wording ("Transcribe this audio... Return only the transcription.") triggers
a hallucinated "I'm just a text-based AI" refusal.

WER is computed on lightly normalised text (lowercase, punctuation stripped) so the
score reflects recognition rather than the model's formatting whims.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import unicodedata
import wave

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
       "capable of perceiving auditory and visual inputs, as well as generating text "
       "and speech.")
INSTR = ("You are a speech recognition system. Output the exact words spoken "
         "in the audio and nothing else.")

ROOT = "/scratch/project_465002364/audio"
JSONL = f"{ROOT}/conversations/aligned_setup.jsonl"


def is_heldout(audio_path):
    """Stable eval split, independent of row order or file growth."""
    return int(hashlib.md5(audio_path.encode()).hexdigest(), 16) % 40 == 0


# Double quotes only. Including the apostrophe made "I'm sorry, I can't recognize the
# words" strip to "m sorry, I can" — it pairs the apostrophe in I'm with the one in
# can't. Refusals are common in the Irish outputs, so that silently mangled the rows
# that matter most. The model quotes transcriptions with double quotes.
QUOTED = re.compile(r'["“”](.+?)["“”]', re.S)
LEADS = [
    r"^the exact words spoken in the audio are\s*:?\s*",
    r"^the words spoken in the audio are\s*:?\s*",
    r"^the audio says\s*:?\s*",
    r"^sure[,!]?\s*(here.s the transcription\s*:?\s*)?",
]
TRAILERS = [
    r"\s*if you (have|want|need)\b.*$",
    r"\s*(feel free|let me know)\b.*$",
    r"\s*maybe you (could|meant)\b.*$",
    r"\s*(could|can) you please\b.*$",
]


def strip_boilerplate(s):
    """Recover the transcription proper from the model's chat wrapper.

    Required, not cosmetic: a perfect English transcription scored 200% WER unstripped
    because 'If you have any other questions...' counts as insertions. Without this, a
    model that merely becomes less chatty looks like it learned Irish.
    """
    t = s.strip()
    m = QUOTED.search(t)
    if m and len(m.group(1).split()) >= 1:
        return m.group(1).strip()
    for pat in LEADS:
        t = re.sub(pat, "", t, flags=re.I)
    for pat in TRAILERS:
        t = re.sub(pat, "", t, flags=re.I)
    return t.strip()


def norm(s):
    """Lowercase, strip punctuation, collapse space. Keeps Irish accented letters."""
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def wer(ref, hyp, truncate=False):
    """Levenshtein over words. Returns (edits, n_ref).

    With truncate=True the hypothesis is cut to the reference length before
    aligning. Both sequences are then at most n_ref long, so the edit distance
    cannot exceed n_ref and WER is bounded at 100% without needing a cap. Use it
    where the decoder may fail to emit EOS and run to the token limit: the
    resulting insertions are a decoding artifact rather than a transcription
    error, and unbounded they dominate the score. Off by default so the existing
    suites keep reporting plain WER.
    """
    r, h = norm(ref).split(), norm(hyp).split()
    if not r:
        return 0, 0
    if truncate:
        h = h[:len(r)]
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1], len(r)


def read_wav(path):
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1
    raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    args = ap.parse_args()
    if args.model is None:
        args.model = glob.glob("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                               "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")[0]

    rows = [json.loads(l) for l in open(JSONL)]
    held = [r for r in rows if is_heldout(r["audio"])]
    print(f"[split] {len(rows)} total -> {len(held)} held out "
          f"({100*len(held)/len(rows):.1f}%)", flush=True)
    use = [r for r in held if r["duration_s"] <= args.max_seconds][:args.n]
    print(f"[split] scoring {len(use)} utterances <= {args.max_seconds}s", flush=True)

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
    print("[load] OK\n", flush=True)

    tot_e = tot_n = 0
    for i, r in enumerate(use):
        path = os.path.join(ROOT, r["audio"])
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
                                   eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Unstripped, the model's chat wrapper scores as insertions: a PERFECT English
        # transcription measured 200% that way. Score the transcription, not the chatter.
        hyp = strip_boilerplate(raw)
        e, n = wer(r["text"], hyp)
        tot_e, tot_n = tot_e + e, tot_n + n
        print(f"[{i}] {r['duration_s']:.1f}s  WER={100*e/max(n,1):6.1f}%")
        print(f"    REF: {r['text']!r}")
        print(f"    HYP: {hyp!r}\n", flush=True)

    print("=" * 72)
    print(f"BASELINE IRISH ASR  --  corpus WER = {100*tot_e/max(tot_n,1):.1f}%  "
          f"({tot_e} edits / {tot_n} ref words, {len(use)} utts)")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
