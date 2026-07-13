#!/usr/bin/env python3
"""Single-clip smoke test for speech-tknz.py's Omni audio path.

Verifies, on one WAV, that:
  (a) the audio-only weight load succeeds (no substantive missing params),
  (b) the encoder forward runs and returns a (n_tokens, 2048) hidden state,
  (c) the forward's n_tokens == the deterministic count from the length formula,
  (d) prints the observed tokens/sec so we can sanity-check the ~25 Hz rate.

Run inside the multitorch container BEFORE the full 1 h run:
    singularity exec -B <sqsh?> -B /scratch/project_465002364:/scratch/... \
        $SIF python smoke_test.py --wav <file.wav> --model-dir <snapshot>
"""
import argparse
import importlib.util
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "sptknz", os.path.join(HERE, "speech-tknz.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="trim the clip to this many seconds for the test")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    import torch
    T._FE = T.load_feature_extractor(args.model_dir)
    print("feature extractor:", type(T._FE).__name__)
    T._ENC = T.load_audio_encoder(args.model_dir)
    T._DEVICE = args.device
    if args.device == "cuda":
        T._ENC = T._ENC.to("cuda")
    n_params = sum(p.numel() for p in T._ENC.parameters())
    print(f"audio encoder params: {n_params/1e6:.1f} M")

    wav, dur = T.load_wav(args.wav)
    wav = wav[: int(args.seconds * T.SAMPLE_RATE)]
    clip_s = len(wav) / T.SAMPLE_RATE
    print(f"clip: {clip_s:.1f}s")

    input_features, feature_lens = T.mel_and_lens([wav])
    print("mel input_features:", tuple(input_features.shape),
          "feature_lens:", feature_lens.tolist())

    det = int(T.count_tokens(feature_lens).sum().item())
    print(f"deterministic token count: {det}  (~{det/clip_s:.2f} tok/s)")

    hidden, out_lens = T.encode(input_features, feature_lens)
    print("encoder last_hidden_state:", tuple(hidden.shape),
          "(expect (n_tokens, 2048))")
    fwd = hidden.shape[0]
    print(f"forward token count: {fwd}  (~{fwd/clip_s:.2f} tok/s)")

    ok = (fwd == det) and (hidden.shape[-1] == 2048)
    print("MATCH" if ok else "MISMATCH", "det=%d fwd=%d dim=%d"
          % (det, fwd, hidden.shape[-1]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
