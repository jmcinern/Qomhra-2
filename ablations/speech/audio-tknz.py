#!/usr/bin/env python3
"""
Ablations Job-1 — mHuBERT-147 audio tokenizer (CPU, LUMI-C).

Copy of full-train-est/audio-tknz.py (read-only original, DO NOT edit that one)
with one scale addition for the full-corpus run: file-level skip-existing so a
SLURM job array is idempotent and restartable (see filter in main()).

Tokenize audio into discrete mHuBERT-147 units. No refitting (uses the released
faiss codebook as-is), no dedup (one unit ID per 50 Hz frame is kept).

Pipeline per audio chunk:
  wav (16 kHz mono) --do_normalize--> HubertModel(mHuBERT-147-base-2nd-iter)
  --> hidden_states[9] (50 Hz, 768-d) --> faiss OPQ+IVF coarse-quantizer search
  --> unit IDs in [0, 999]

The faiss assignment (load_index / get_centroids_index) is the official recipe
from utter-project/mHuBERT-147-scripts/03_faiss_indices/faiss_index_creation.py.

Parallelism: one process per CPU core. The 4.74 GB faiss index is loaded ONCE in
the parent and inherited read-only by workers via fork (copy-on-write), so it is
not duplicated per worker. Long files are split into fixed-length chunks that are
distributed across workers and reassembled in order per file.
"""
import argparse
import csv
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import time
import multiprocessing as mp

import numpy as np

SAMPLE_RATE = 16000
LAYER = 9  # hidden_states[9] == mHuBERT-147 layer 9 (tuple length 13)
# The seven convolutional layers have a 400-sample receptive field and a
# 320-sample (20 ms) total stride.  A bare 30 s block therefore emits 1499
# frames, not 1500.  Adjacent blocks overlap by the 80-sample lookahead below
# so concatenated outputs retain one globally regular 50 Hz timeline.
FEATURE_STRIDE = 320
FEATURE_RECEPTIVE_FIELD = 400
RIGHT_CONTEXT = FEATURE_RECEPTIVE_FIELD - FEATURE_STRIDE

# ---- globals populated in the parent (index) and per-worker (model) ----
_INDEX = None
_INDEX_IVF = None
_MODEL = None
_PROCESSOR = None


# --------------------------------------------------------------------------
# faiss assignment — verbatim recipe from mHuBERT-147-scripts
# --------------------------------------------------------------------------
def load_index(index_path):
    import faiss
    index = faiss.read_index(index_path)
    index_ivf = faiss.extract_index_ivf(index)
    return index, index_ivf


def get_centroids_index(xq, index, index_ivf):
    """xq: float32 [T, 768] raw layer-9 features -> centroid IDs [T,1]."""
    import faiss
    opq_mt = faiss.downcast_VectorTransform(index.chain.at(0))
    xq_t = opq_mt.apply_py(xq)
    _, C = index_ivf.quantizer.search(xq_t, 1)
    return C


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def init_worker(model_dir):
    import torch
    torch.set_num_threads(1)
    from transformers import AutoModel, AutoFeatureExtractor
    global _MODEL, _PROCESSOR
    _PROCESSOR = AutoFeatureExtractor.from_pretrained(model_dir)
    _MODEL = AutoModel.from_pretrained(model_dir)
    _MODEL.eval()


def _load_chunk(path, start, stop):
    """Read [start, stop) frames, return mono float32 @ 16 kHz."""
    import soundfile as sf
    wav, sr = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # -> mono
    if sr != SAMPLE_RATE:
        import torch, torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    return wav


def tokenize_chunk(task):
    """Tokenize one valid block plus enough right context for its last frame.

    ``task = (path, chunk_idx, start, read_stop, target_input_samples)``.
    Non-final blocks read 80 samples from the following block.  A recording's
    final block is right-zero-padded instead.  In both cases every valid block
    contributes exactly ``ceil(valid_16k_samples / 320)`` units, so concatenating
    chunks produces a cache that can be indexed at 50 Hz without boundary drift.
    """
    import torch
    path, chunk_idx, start, read_stop, target_input_samples = task
    wav = _load_chunk(path, start, read_stop)
    if wav.shape[0] < target_input_samples:
        wav = np.pad(wav, (0, target_input_samples - wav.shape[0]))
    elif wav.shape[0] > target_input_samples:
        wav = wav[:target_input_samples]
    inputs = _PROCESSOR(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        feats = _MODEL(inputs.input_values,
                       output_hidden_states=True).hidden_states[LAYER]
    feats = feats.squeeze(0).numpy().astype("float32")  # [T, 768]
    units = get_centroids_index(feats, _INDEX, _INDEX_IVF)[:, 0].astype(np.uint16)
    expected = (
        (target_input_samples - FEATURE_RECEPTIVE_FIELD) // FEATURE_STRIDE + 1
    )
    if units.shape != (expected,):
        raise RuntimeError(
            f"{path} chunk {chunk_idx}: mHuBERT emitted {units.shape[0]} frames, "
            f"expected {expected} for {target_input_samples} input samples"
        )
    return path, chunk_idx, units


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def read_manifest(manifest, audio_root):
    """Yield (dest_label, abs_path) for each row, resolving the file on disk."""
    rows = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            label, name = r["dest_label"], r["filename"]
            p = os.path.join(audio_root, label, name)
            if not os.path.isfile(p):
                print(f"WARN missing: {p}", file=sys.stderr)
                continue
            rows.append((label, p))
    return rows


def build_tasks(rows, chunk_s):
    """Split each file into <=chunk_s chunks -> flat task list (longest first)."""
    import soundfile as sf
    tasks, totals = [], {}
    for _, path in rows:
        info = sf.info(path)
        n, sr = info.frames, info.samplerate
        totals[path] = n
        chunk_frames = int(round(chunk_s * sr))
        right_context_src = int(np.ceil(RIGHT_CONTEXT * sr / SAMPLE_RATE))
        idx = 0
        for start in range(0, n, chunk_frames):
            valid_stop = min(start + chunk_frames, n)
            read_stop = min(valid_stop + right_context_src, n)
            valid_16k = int(round((valid_stop - start) * SAMPLE_RATE / sr))
            n_output = (valid_16k + FEATURE_STRIDE - 1) // FEATURE_STRIDE
            target_input_samples = (
                (n_output - 1) * FEATURE_STRIDE + FEATURE_RECEPTIVE_FIELD
            )
            tasks.append(
                (path, idx, start, read_stop, target_input_samples)
            )
            idx += 1
    tasks.sort(key=lambda t: totals[t[0]], reverse=True)  # balance long files
    return tasks


def out_path(out_dir, audio_root, path):
    rel = os.path.relpath(path, audio_root)
    dst = os.path.join(out_dir, os.path.splitext(rel)[0] + ".npy")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--faiss-index", required=True)
    output = ap.add_mutually_exclusive_group(required=True)
    output.add_argument("--out-dir")
    output.add_argument(
        "--parquet-out",
        help="write one Parquet unit store instead of one .npy file per utterance",
    )
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--chunk-s", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N files only")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-tokenize files whose .npy already exists (default: skip)")
    args = ap.parse_args()

    rows = read_manifest(args.manifest, args.audio_root)
    if args.limit:
        rows = rows[: args.limit]
    print(f"files: {len(rows)}", flush=True)

    if args.parquet_out and os.path.exists(args.parquet_out) and not args.overwrite:
        print(f"nothing to do (Parquet already exists): {args.parquet_out}", flush=True)
        return

    # Scale addition: file-level skip-existing so the SLURM array is idempotent
    # and restartable. This applies only to the legacy per-file output mode.
    if args.out_dir and not args.overwrite:
        n_before = len(rows)
        rows = [(label, p) for (label, p) in rows
                if not os.path.exists(out_path(args.out_dir, args.audio_root, p))]
        skipped = n_before - len(rows)
        if skipped:
            print(f"skipped {skipped} already-tokenized files; {len(rows)} remain",
                  flush=True)
    if not rows:
        print("nothing to do (all files already tokenized)", flush=True)
        return

    tasks = build_tasks(rows, args.chunk_s)
    print(f"chunks: {len(tasks)} (chunk_s={args.chunk_s})", flush=True)

    # Load the index ONCE here; workers inherit it via fork (copy-on-write).
    global _INDEX, _INDEX_IVF
    print("loading faiss index ...", flush=True)
    t0 = time.time()
    _INDEX, _INDEX_IVF = load_index(args.faiss_index)
    print(f"index loaded in {time.time()-t0:.0f}s", flush=True)

    n_workers = max(1, min(args.workers, len(tasks)))
    results = {}  # path -> {chunk_idx: units}
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=init_worker,
                  initargs=(args.model_dir,)) as pool:
        done = 0
        for path, chunk_idx, units in pool.imap_unordered(tokenize_chunk, tasks):
            results.setdefault(path, {})[chunk_idx] = units
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} chunks", flush=True)

    n_units = 0
    packed = []
    for label, path in rows:
        chunks = results.get(path, {})
        if not chunks:
            print(f"WARN no output: {path}", file=sys.stderr)
            continue
        units = np.concatenate([chunks[i] for i in sorted(chunks)])
        if args.parquet_out:
            packed.append((label, os.path.basename(path), units))
        else:
            np.save(out_path(args.out_dir, args.audio_root, path), units)
        n_units += units.shape[0]
    if args.parquet_out:
        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(os.path.dirname(os.path.abspath(args.parquet_out)), exist_ok=True)
        temp = args.parquet_out + ".tmp"
        pq.write_table(
            pa.table(
                {
                    "dest_label": [row[0] for row in packed],
                    "filename": [row[1] for row in packed],
                    "units": pa.array(
                        [row[2] for row in packed], type=pa.list_(pa.uint16())
                    ),
                }
            ),
            temp,
            compression="zstd",
        )
        os.replace(temp, args.parquet_out)
    destination = args.parquet_out or args.out_dir
    print(f"DONE files={len(rows)} units={n_units} out={destination}", flush=True)


if __name__ == "__main__":
    main()
