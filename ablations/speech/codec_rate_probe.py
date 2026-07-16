#!/usr/bin/env python3
"""Measure Qwen2.5-Omni's speech-codec token rate and value range.

Why: the released Qwen2.5-Omni has no waveform -> codec-token encoder, so we cannot
build Talker training targets from Irish audio. The hypothesis under test is that
Qwen2.5's codec is Kyutai-Mimi-compatible, which would let the public Mimi encoder
stand in for the missing `qwen-tts-tokenizer`.

The config already argues against it:
    Qwen2.5 token2wav.dit_config.num_embeds = 8193   (one flat table, ~8192 codes)
    Mimi                    codebook_size   = 2048   (RVQ, multi-codebook), 12.5 Hz

This probe settles it empirically WITHOUT needing to guess a de-interleave
convention (which a decode-and-compare test would, risking a false negative):

    ~25 tok/s, values spanning 0..8191   -> single 25 Hz stream, NOT Mimi.
    ~50 tok/s                            -> consistent with 4 x 12.5 Hz RVQ levels
                                            interleaved; Mimi hypothesis survives,
                                            proceed to the decode test.

It doubles as the end-to-end check that the full stack (thinker -> talker ->
token2wav -> waveform) actually loads and vocalises on this container.

Run inside the container on one GCD; see codec_rate_probe.sh.
"""
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    # A repo id sends the tokenizer's `is_base_mistral` check to the HF API, which the
    # compute nodes cannot reach; a local path short-circuits it as `_is_local`.
    ap.add_argument("--model", default=None,
                    help="default: the local Qwen2.5-Omni-3B snapshot dir")
    ap.add_argument("--speaker", default="Chelsie")
    ap.add_argument("--spk", default=None,
                    help="spk_dict.safetensors (default: next to the snapshot's .pt)")
    ap.add_argument("--out", default=None, help="optional .npz dump of codes+wav")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog. "
                                      "Speech synthesis is working correctly today.")
    args = ap.parse_args()
    import glob as _glob
    snaps = _glob.glob("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                       "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")
    if args.model is None:
        if not snaps:
            raise SystemExit("no local Qwen2.5-Omni-3B snapshot found")
        args.model = snaps[0]
    if args.spk is None:
        found = _glob.glob(os.path.join(args.model, "spk_dict.safetensors"))
        if not found:
            raise SystemExit("spk_dict.safetensors not found — run convert_spk_dict.py first")
        args.spk = found[0]
    print(f"model={args.model}\nspk  ={args.spk}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    from convert_spk_dict import load_spk_dict

    print("[load] full Omni (thinker + talker + token2wav)...", flush=True)
    # Qwen2_5OmniForConditionalGeneration.from_pretrained() = super().from_pretrained()
    # + load_speakers(), and only that second half is blocked (it torch.loads spk_dict.pt,
    # which transformers refuses below torch 2.6 regardless of weights_only). So we call
    # the parent's from_pretrained for the weights — all safetensors, entirely unaffected —
    # and supply speaker_map from the converted safetensors file instead. No torch.load, no
    # security check disabled: exactly the route the guard's own message recommends.
    # bf16 for the LM stack; generate() internally floats token2wav itself.
    cls = Qwen2_5OmniForConditionalGeneration
    model = super(cls, cls).from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    model.speaker_map = load_spk_dict(args.spk)
    print(f"[load] OK  has_talker={model.has_talker}  "
          f"speakers={list(model.speaker_map.keys())}", flush=True)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)

    # Hook token2wav to capture exactly the tensor generate() hands it:
    #   talker_generate_codes = talker_result[:, talker_input_ids.shape[1]: -1]
    #   wav = self.token2wav(talker_generate_codes, ...)
    captured = {}

    def hook(_module, hook_args, _kwargs):
        captured["code"] = hook_args[0].detach().to("cpu")
        return None

    handle = model.token2wav.register_forward_pre_hook(hook, with_kwargs=True)

    conv = [
        {"role": "system", "content": [{"type": "text",
         "text": "You are Qwen, a virtual human that can talk."}]},
        {"role": "user", "content": [{"type": "text", "text": args.text}]},
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)

    print("[gen] generating speech...", flush=True)
    with torch.no_grad():
        text_ids, wav = model.generate(
            **inputs, speaker=args.speaker, return_audio=True,
            do_sample=False, max_new_tokens=64, thinker_max_new_tokens=64,
        )
    handle.remove()

    code = captured.get("code")
    if code is None:
        raise SystemExit("FAIL: token2wav was never called; no codes captured.")

    wav = wav.float().reshape(-1).cpu()
    # Qwen2.5-Omni's BigVGAN emits 24 kHz.
    sr = int(getattr(getattr(model.config, "token2wav_config", None),
                     "bigvgan_config", None).sampling_rate) \
        if hasattr(getattr(model.config.token2wav_config, "bigvgan_config"),
                   "sampling_rate") else 24000
    dur = wav.numel() / sr
    n = code.reshape(-1).numel()

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"  transcript      : {processor.batch_decode(text_ids, skip_special_tokens=True)[0][:120]!r}")
    print(f"  code tensor     : shape={tuple(code.shape)} dtype={code.dtype}")
    print(f"  n codes         : {n}")
    print(f"  code min/max    : {int(code.min())} / {int(code.max())}")
    print(f"  distinct codes  : {int(torch.unique(code).numel())}")
    print(f"  wav samples     : {wav.numel()}  @ {sr} Hz")
    print(f"  duration        : {dur:.3f} s")
    print(f"  --> TOKEN RATE  : {n / dur:.2f} tok/s")
    print("=" * 66)
    rate = n / dur
    if 20 <= rate <= 30:
        print("  VERDICT: ~25 Hz single stream -> NOT Mimi (12.5 Hz RVQ). Hypothesis dead.")
    elif 45 <= rate <= 55:
        print("  VERDICT: ~50 tok/s -> consistent with 4 x 12.5 Hz interleaved RVQ.")
        print("           Mimi hypothesis SURVIVES; run the decode comparison next.")
    else:
        print(f"  VERDICT: unexpected rate {rate:.1f} tok/s -- inspect before concluding.")
    print("=" * 66, flush=True)

    if args.out:
        np.savez(args.out, code=code.numpy(), wav=wav.numpy(), sr=sr)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
