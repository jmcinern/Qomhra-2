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
    """task = (path, chunk_idx, start, stop). Returns (path, chunk_idx, units)."""
    import torch
    path, chunk_idx, start, stop = task
    wav = _load_chunk(path, start, stop)
    inputs = _PROCESSOR(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        feats = _MODEL(inputs.input_values,
                       output_hidden_states=True).hidden_states[LAYER]
    feats = feats.squeeze(0).numpy().astype("float32")  # [T, 768]
    units = get_centroids_index(feats, _INDEX, _INDEX_IVF)[:, 0].astype(np.uint16)
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
    chunk_frames = chunk_s * SAMPLE_RATE
    tasks, totals = [], {}
    for _, path in rows:
        n = sf.info(path).frames
        totals[path] = n
        idx = 0
        for start in range(0, n, chunk_frames):
            tasks.append((path, idx, start, min(start + chunk_frames, n)))
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
    ap.add_argument("--out-dir", required=True)
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

    # Scale addition: file-level skip-existing so the SLURM array is idempotent
    # and restartable. out_path() only makes the parent dir, so this check is a
    # pure existence test; --overwrite forces a full re-run of the shard.
    if not args.overwrite:
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
    for _, path in rows:
        chunks = results.get(path, {})
        if not chunks:
            print(f"WARN no output: {path}", file=sys.stderr)
            continue
        units = np.concatenate([chunks[i] for i in sorted(chunks)])
        np.save(out_path(args.out_dir, args.audio_root, path), units)
        n_units += units.shape[0]
    print(f"DONE files={len(rows)} units={n_units} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
