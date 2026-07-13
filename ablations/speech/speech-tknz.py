#!/usr/bin/env python3
"""
Ablations — Qwen2.5-Omni-3B speech "tokenizer" / token counter (LUMI-C & LUMI-G).

Unlike the mHuBERT path (audio-tknz.py), Qwen2.5-Omni does NOT emit discrete input
tokens for speech. Its audio path is:

    wav (16 kHz mono) --WhisperFeatureExtractor--> 128-mel @ 100 Hz
      --Qwen2_5OmniAudioEncoder (conv stride-2 + transformer + avg-pool stride-2
        + proj to 2048)--> ~25 Hz sequence of CONTINUOUS 2048-d embeddings

Each of those output positions occupies exactly one LLM input slot — i.e. it is one
"token" of the model's context budget, directly comparable to a text token. So for
the ablations the meaningful quantities are:

  (1) TOKEN COUNT  — deterministic in audio length (~25 tok/s), used to equalise the
      Speech budget against the 2.13 B-token text corpus. Computed exactly from the
      feature-extractor mel length via the encoder's own output-length formula.
  (2) THROUGHPUT   — how fast we can run the audio encoder forward (the real compute,
      and what the model pays every step at train time). Benchmarked here on CPU
      (LUMI-C, fork pool) and GPU (LUMI-G, batched) to decide where to run.

There is nothing discrete to persist, so by default we only write per-file token
counts (a .tsv) + a totals JSON. --save-embeddings optionally caches the 2048-d
encoder output per file (fp16 .npy) if we later want precomputed audio features.

Container (per ablations/README): /appl/local/laifs/containers/lumi-multitorch-latest.sif
  transformers 4.57.6, torch 2.10+rocm7, torchaudio (no soundfile) -> we load WAV
  with scipy.io.wavfile (files are 16 kHz mono int16 PCM).
"""
import argparse
import csv
import json
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import time
import glob
import multiprocessing as mp

import numpy as np

SAMPLE_RATE = 16000

# ---- globals populated per-worker (CPU) / once (GPU) --------------------------
_FE = None          # WhisperFeatureExtractor
_ENC = None         # Qwen2_5OmniAudioEncoder
_DEVICE = "cpu"
_DTYPE = None       # torch dtype for encoder forward


# ------------------------------------------------------------------------------
# model loading — ONLY the audio tower (encoder+adapter), not the 3B LLM/Talker
# ------------------------------------------------------------------------------
def load_feature_extractor(model_dir):
    from transformers import AutoFeatureExtractor
    # Qwen2_5OmniProcessor bundles a WhisperFeatureExtractor for audio; grab it
    # directly so we don't drag in the image processor / tokenizer.
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    # AutoFeatureExtractor may return the Qwen omni processor's FE wrapper; if it
    # exposes a .feature_extractor attribute (processor), unwrap it.
    return getattr(fe, "feature_extractor", fe)


def load_audio_encoder(model_dir, dtype=None):
    """Build Qwen2_5OmniAudioEncoder and load ONLY thinker.audio_tower.* weights
    from the sharded safetensors — avoids materialising the 3B text model."""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
    )
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(model_dir)
    audio_cfg = cfg.thinker_config.audio_config
    enc = Qwen2_5OmniAudioEncoder(audio_cfg)

    prefix = "thinker.audio_tower."
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        weight_map = json.load(open(idx_path))["weight_map"]
        shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    else:
        shards = [os.path.basename(p)
                  for p in glob.glob(os.path.join(model_dir, "*.safetensors"))]

    state = {}
    for shard in shards:
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = f.get_tensor(key)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    # A few buffers (e.g. rotary/inv_freq) are non-persistent and legitimately
    # "missing"; anything substantive missing means the prefix/layout changed.
    real_missing = [m for m in missing if "inv_freq" not in m and "rotary" not in m]
    if real_missing:
        print(f"WARN encoder missing {len(real_missing)} params, "
              f"e.g. {real_missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"WARN encoder unexpected {len(unexpected)} params, "
              f"e.g. {unexpected[:5]}", file=sys.stderr)
    enc.eval()
    if dtype is not None:
        enc = enc.to(dtype)
    return enc


# ------------------------------------------------------------------------------
# audio I/O — scipy (no soundfile in the container); 16 kHz mono int16 PCM
# ------------------------------------------------------------------------------
def load_wav(path):
    """Return (float32 mono @16 kHz in [-1,1], duration_s)."""
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        wav = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        wav = (data.astype(np.float32) - 128.0) / 128.0
    else:  # already float
        wav = data.astype(np.float32)
    if sr != SAMPLE_RATE:
        import torch, torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
        sr = SAMPLE_RATE
    return wav, len(wav) / float(sr)


def iter_chunks(wav, chunk_s):
    n = chunk_s * SAMPLE_RATE
    for start in range(0, len(wav), n):
        yield wav[start:start + n]


# ------------------------------------------------------------------------------
# core: mel -> token count (deterministic) and optional encoder forward
# ------------------------------------------------------------------------------
def mel_and_lens(wav_chunks):
    """Feature-extract a list of 1-D waveforms -> (input_features, feature_lens).
    Returns torch tensors ready for the encoder plus the true mel length per chunk."""
    import torch
    feats = _FE(wav_chunks, sampling_rate=SAMPLE_RATE,
                return_attention_mask=True, return_tensors="pt")
    input_features = feats["input_features"]              # (B, 128, T) padded
    attn = feats["feature_attention_mask"]                # (B, T)
    feature_lens = attn.sum(-1).to(torch.long)            # (B,) true mel frames
    return input_features, feature_lens


def count_tokens(feature_lens):
    """Exact number of LLM-input audio tokens per utterance, via the encoder's own
    conv+pool output-length arithmetic (no forward needed)."""
    out = _ENC._get_feat_extract_output_lengths(feature_lens)
    # transformers returns either out_len or (aftercnn_len, out_len)
    out_len = out[1] if isinstance(out, (tuple, list)) else out
    return out_len


def encode(input_features, feature_lens):
    """Run the audio encoder forward; returns (last_hidden_state, out_lens).
    Packs the padded (B,128,T) batch into the concatenated layout the tower wants."""
    import torch
    aftercnn = _ENC._get_feat_extract_output_lengths(feature_lens)
    aftercnn_lens = aftercnn[0] if isinstance(aftercnn, (tuple, list)) else feature_lens
    # Concatenate valid mel frames across the batch: (128, sum_T)
    packed = torch.cat(
        [input_features[i, :, :feature_lens[i]] for i in range(input_features.shape[0])],
        dim=1,
    )
    packed = packed.to(_DEVICE, _DTYPE) if _DTYPE else packed.to(_DEVICE)
    with torch.no_grad():
        out = _ENC(packed, feature_lens=feature_lens.to(_DEVICE),
                   aftercnn_lens=aftercnn_lens.to(_DEVICE))
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    out_lens = aftercnn[1] if isinstance(aftercnn, (tuple, list)) else None
    return hidden, out_lens


# ------------------------------------------------------------------------------
# per-file processing (used by both CPU workers and the GPU loop)
# ------------------------------------------------------------------------------
def process_file(task):
    """task=(dest_label, path, chunk_s, run_encoder, save_emb, out_npy).
    Returns (dest_label, filename, dur_s, n_tokens)."""
    dest_label, path, chunk_s, run_encoder, save_emb, out_npy = task
    wav, dur = load_wav(path)
    chunks = list(iter_chunks(wav, chunk_s))
    n_tokens = 0
    embs = []
    for ch in chunks:
        input_features, feature_lens = mel_and_lens([ch])
        if run_encoder:
            hidden, _ = encode(input_features, feature_lens)
            n_tokens += hidden.shape[0]
            if save_emb:
                embs.append(hidden.float().cpu().numpy().astype(np.float16))
        else:
            n_tokens += int(count_tokens(feature_lens).sum().item())
    if save_emb and embs:
        os.makedirs(os.path.dirname(out_npy), exist_ok=True)
        np.save(out_npy, np.concatenate(embs, axis=0))
    return dest_label, os.path.basename(path), round(dur, 3), n_tokens


# ------------------------------------------------------------------------------
# manifest
# ------------------------------------------------------------------------------
def read_manifest(manifest, audio_root, limit=0):
    rows = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            label, name = r["dest_label"], r["filename"]
            p = os.path.join(audio_root, label, name)
            if not os.path.isfile(p):
                print(f"WARN missing: {p}", file=sys.stderr)
                continue
            rows.append((label, p))
            if limit and len(rows) >= limit:
                break
    return rows


def out_npy_path(out_dir, audio_root, path):
    rel = os.path.relpath(path, audio_root)
    return os.path.join(out_dir, os.path.splitext(rel)[0] + ".f16.npy")


# ------------------------------------------------------------------------------
# CPU worker init
# ------------------------------------------------------------------------------
def init_worker(model_dir, device, dtype, run_encoder):
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    global _FE, _ENC, _DEVICE, _DTYPE
    _DEVICE = device
    _DTYPE = dtype
    _FE = load_feature_extractor(model_dir)
    # Always need the encoder object for the output-length formula; only move it to
    # device / load full weights when we actually run the forward.
    _ENC = load_audio_encoder(model_dir, dtype=dtype if run_encoder else None)
    if run_encoder and device != "cpu":
        _ENC = _ENC.to(device)


# ------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--workers", type=int, default=128,
                    help="CPU fork-pool size (device=cpu only)")
    ap.add_argument("--chunk-s", type=int, default=30,
                    help="split long files into <=chunk_s windows for the encoder")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                    default="float32")
    ap.add_argument("--count-only", action="store_true",
                    help="skip the encoder forward; deterministic token count only "
                         "(fast, exact — answers the budget question)")
    ap.add_argument("--save-embeddings", action="store_true",
                    help="cache 2048-d encoder output per file as fp16 .npy")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N files")
    args = ap.parse_args()

    import torch
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    run_encoder = not args.count_only
    dtype_arg = dtype if (run_encoder and args.device == "cuda") else None

    rows = read_manifest(args.manifest, args.audio_root, args.limit)
    print(f"files: {len(rows)}  device={args.device}  "
          f"mode={'count-only' if args.count_only else 'encoder'}", flush=True)
    if not rows:
        print("nothing to do", flush=True)
        return
    os.makedirs(args.out_dir, exist_ok=True)

    tasks = [
        (label, p, args.chunk_s, run_encoder, args.save_embeddings,
         out_npy_path(args.out_dir, args.audio_root, p))
        for (label, p) in rows
    ]

    results = []
    t0 = time.time()
    if args.device == "cpu":
        n_workers = max(1, min(args.workers, len(tasks)))
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers, initializer=init_worker,
                      initargs=(args.model_dir, "cpu", dtype_arg, run_encoder)) as pool:
            done = 0
            for res in pool.imap_unordered(process_file, tasks):
                results.append(res)
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} files", flush=True)
    else:
        init_worker(args.model_dir, "cuda", dtype_arg, run_encoder)
        for i, task in enumerate(tasks, 1):
            results.append(process_file(task))
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} files", flush=True)
    wall = time.time() - t0

    # ---- report ----
    tsv_path = os.path.join(args.out_dir, "speech_token_counts.tsv")
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dest_label", "filename", "duration_s", "n_tokens"])
        w.writerows(sorted(results))

    total_tokens = sum(r[3] for r in results)
    total_dur = sum(r[2] for r in results)
    audio_h = total_dur / 3600.0
    summary = {
        "device": args.device,
        "mode": "count-only" if args.count_only else "encoder",
        "dtype": args.dtype if run_encoder else None,
        "workers": args.workers if args.device == "cpu" else 1,
        "files": len(results),
        "audio_hours": round(audio_h, 4),
        "total_tokens": int(total_tokens),
        "tokens_per_audio_hour": round(total_tokens / audio_h, 1) if audio_h else 0,
        "wall_s": round(wall, 1),
        "audio_hours_per_wall_hour": round(audio_h / (wall / 3600.0), 2) if wall else 0,
        "rtf_x": round(total_dur / wall, 1) if wall else 0,  # x real-time
    }
    with open(os.path.join(args.out_dir, "speech_token_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
