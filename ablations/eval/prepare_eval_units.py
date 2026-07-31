#!/usr/bin/env python3
"""Turn eval audio into mHuBERT unit sequences, the same way training data was made.

CLUAS and IWSLT audio lives inside Parquet as wav bytes, but `speech/audio_tknz_gpu.py`
reads files from disk via a TSV manifest. So this runs in two passes with the tokenizer
in between:

  export   parquet -> node-local wavs + manifest.tsv
           (run audio_tknz_gpu.py here: manifest + audio-root -> one .npy per clip)
  pack     .npy dir -> one units Parquet keyed by clip id

The units Parquet holds each clip's FULL dedup unit sequence. Windowing for CLUAS is a
prompt-level concern and happens in `cluas_qa_units.py`, so chunk size can change
without re-tokenizing 1.8 hours of exam audio.

Dedup (consecutive duplicates removed) matches `speech/prepare_fleurs_discrete_asr.py`
and the training packer. Unit ids stay 0-999 here; the +151936 offset is applied at
prompt-build time, alongside the sentinels.

  python -m eval.prepare_eval_units export --data data/cluas_all.parquet \
      --key snippet_id --out-wav $TMP/cluas_wav --manifest $TMP/cluas.tsv
  python -m eval.prepare_eval_units pack --manifest $TMP/cluas.tsv \
      --units-dir $TMP/cluas_units --out data/cluas_all.units.parquet
"""
import argparse
import csv
import io
import os
import wave

import numpy as np

UNIT_COUNT = 1000
LABEL = "clips"  # single dest_label; audio_tknz_gpu.py paths are <root>/<label>/<name>


def _decode(raw_bytes):
    """wav bytes -> (float32 mono, sample_rate). Stdlib only; no soundfile here."""
    handle = wave.open(io.BytesIO(raw_bytes))
    sample_width = handle.getsampwidth()
    if sample_width != 2:
        raise RuntimeError(f"expected 16-bit PCM, got sampwidth={sample_width}")
    frames = handle.readframes(handle.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if handle.getnchannels() == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    return data, handle.getframerate()


def export(args):
    import pyarrow.parquet as pq

    table = pq.read_table(args.data, columns=[args.key, "audio"])
    keys = table[args.key].to_pylist()
    audio = table["audio"].to_pylist()

    wav_dir = os.path.join(args.out_wav, LABEL)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)) or ".", exist_ok=True)

    rows = []
    for key, raw in zip(keys, audio):
        data, sample_rate = _decode(raw)
        name = f"{key}.wav"
        with wave.open(os.path.join(wav_dir, name), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(sample_rate)
            out.writeframes((data * 32767.0).astype(np.int16).tobytes())
        rows.append({"dest_label": LABEL, "filename": name,
                     "duration_s": f"{len(data) / sample_rate:.6f}"})

    with open(args.manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dest_label", "filename", "duration_s"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    total = sum(float(r["duration_s"]) for r in rows)
    print(f"exported {len(rows)} clips ({total / 3600:.2f} h) -> {wav_dir}", flush=True)
    print(f"manifest -> {args.manifest}", flush=True)


def dedup(raw, source):
    raw = np.asarray(raw, dtype=np.int64)
    if raw.ndim != 1 or not len(raw) or raw.min() < 0 or raw.max() >= UNIT_COUNT:
        raise RuntimeError(f"{source}: invalid units shape={raw.shape}")
    return raw[np.r_[True, raw[1:] != raw[:-1]]]


def pack(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with open(args.manifest, newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))

    keys, units_out, n_raw, n_dedup, durations = [], [], [], [], []
    for row in manifest:
        name = row["filename"]
        path = os.path.join(args.units_dir, row["dest_label"],
                            os.path.splitext(name)[0] + ".npy")
        if not os.path.isfile(path):
            raise RuntimeError(f"missing tokenized audio: {path}")
        raw = np.load(path)
        clean = dedup(raw, name)
        keys.append(os.path.splitext(name)[0])
        units_out.append(clean.astype(np.uint16))
        n_raw.append(len(raw))
        n_dedup.append(len(clean))
        durations.append(float(row["duration_s"]))

    table = pa.table({
        "key": pa.array(keys, type=pa.string()),
        "units": pa.array(units_out, type=pa.list_(pa.uint16())),
        "n_raw": pa.array(n_raw, type=pa.int32()),
        "n_dedup": pa.array(n_dedup, type=pa.int32()),
        "duration_s": pa.array(durations, type=pa.float32()),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")

    raw_total, dedup_total = sum(n_raw), sum(n_dedup)
    print(f"packed {len(keys)} clips -> {args.out}", flush=True)
    print(f"  raw units      {raw_total}", flush=True)
    print(f"  dedup units    {dedup_total} (retention {dedup_total / raw_total:.4f})",
          flush=True)
    print(f"  dedup per clip min {min(n_dedup)} median {int(np.median(n_dedup))} "
          f"max {max(n_dedup)}", flush=True)
    print(f"  frame rate     {raw_total / sum(durations):.2f} Hz", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    command = sub.add_parser("export")
    command.add_argument("--data", required=True, help="eval parquet with an audio column")
    command.add_argument("--key", required=True, help="id column: snippet_id | name")
    command.add_argument("--out-wav", required=True, help="node-local wav root")
    command.add_argument("--manifest", required=True)
    command.set_defaults(func=export)

    command = sub.add_parser("pack")
    command.add_argument("--manifest", required=True)
    command.add_argument("--units-dir", required=True, help="audio_tknz_gpu.py --out-dir")
    command.add_argument("--out", required=True)
    command.set_defaults(func=pack)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
