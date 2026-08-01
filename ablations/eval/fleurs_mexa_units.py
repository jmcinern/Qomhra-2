#!/usr/bin/env python3
"""Turn the MEXA FLEURS sentences' audio into mHuBERT unit IDs.

Two subcommands, matching the existing FLEURS unit pipeline:

    export    fleurs_parallel_test.parquet -> wavs + a manifest audio_tknz_gpu.py can read
    attach    that tokenizer's units.parquet -> one unit Parquet keyed by sentence id

The sentence selection is IDENTICAL to mexa_embed.py::load_sentences -- test rows only,
sorted by id, first n -- because the whole point is that the unit embeddings and the text
embeddings are row-for-row parallel. The selection is duplicated rather than imported so
this script can run in the mhubert environment, which does not have transformers.

Each sentence contributes TWO clips, `ga` and `en`, so `dest_label` is the language and
`filename` is `<id>.wav`. That is the same (dest_label, filename) key the tokenizer emits,
so `attach` can join without guessing.
"""
import argparse
import io
import json
import os

import numpy as np

SAMPLE_RATE = 16000
UNIT_COUNT = 1000
LANGS = ("ga", "en")


def load_sentences(parquet_path, n):
    """First n test-split sentence ids, sorted. Mirrors mexa_embed.py::load_sentences."""
    import pyarrow.parquet as pq
    records = pq.read_table(parquet_path).to_pylist()
    test = sorted((r for r in records if r["split"] == "test"), key=lambda r: r["id"])
    if len(test) < n:
        raise SystemExit(f"{parquet_path}: only {len(test)} test rows, need {n}")
    chosen = test[:n]
    print(f"[data] {len(records)} rows, {len(test)} test, using first {n} "
          f"(ids {chosen[0]['id']}..{chosen[-1]['id']})", flush=True)
    return chosen


def decode_audio(blob):
    """FLEURS audio cell -> mono float32 at 16 kHz.

    The parquet stores either raw file bytes or a {'bytes':..., 'path':...} struct; both
    appear in HF audio columns depending on how the file was written.
    """
    import soundfile as sf
    if isinstance(blob, dict):
        blob = blob.get("bytes") or blob.get("array")
    if isinstance(blob, (bytes, bytearray, memoryview)):
        wav, sr = sf.read(io.BytesIO(bytes(blob)), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    else:
        wav, sr = np.asarray(blob, dtype=np.float32), SAMPLE_RATE
    if sr != SAMPLE_RATE:
        import torch, torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    return wav.astype(np.float32)


def export(args):
    """Write 2n wavs and the manifest. Wavs are a scratch intermediate, not an artefact."""
    import soundfile as sf
    rows = load_sentences(args.data, args.n)
    for lang in LANGS:
        os.makedirs(os.path.join(args.out_dir, lang), exist_ok=True)

    lines = ["dest_label\tfilename\tduration_s"]
    written = 0
    for row in rows:
        for lang in LANGS:
            wav = decode_audio(row[f"{lang}_audio"])
            if wav.size == 0:
                raise SystemExit(f"sentence {row['id']} {lang}: empty audio")
            name = f"{row['id']}.wav"
            sf.write(os.path.join(args.out_dir, lang, name), wav, SAMPLE_RATE)
            lines.append(f"{lang}\t{name}\t{wav.size / SAMPLE_RATE:.6f}")
            written += 1

    with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    ids = [r["id"] for r in rows]
    with open(args.sentence_ids, "w", encoding="utf-8") as f:
        json.dump({"data": os.path.abspath(args.data), "n": args.n,
                   "sentence_ids": ids, "languages": list(LANGS)}, f, indent=2)
    print(f"[export] {written} clips ({len(rows)} sentences x {len(LANGS)} langs) -> "
          f"{args.out_dir}\n[export] manifest {args.manifest}", flush=True)


def dedup(raw, source):
    """Consecutive duplicates removed -- byte-identical to
    prepare_fleurs_discrete_asr.py::dedup, because the model was TRAINED on deduped units.
    Feeding raw 50 Hz units would be a different input distribution."""
    raw = np.asarray(raw, dtype=np.int64)
    if raw.ndim != 1 or not len(raw) or raw.min() < 0 or raw.max() >= UNIT_COUNT:
        raise RuntimeError(f"{source}: invalid units shape={raw.shape}")
    return raw[np.r_[True, raw[1:] != raw[:-1]]]


def attach(args):
    """Join the tokenizer's output back onto sentence ids, deduped, one row per (id, lang).

    audio_tknz_gpu.py writes one .npy per input file, mirroring the audio tree, so the units
    for sentence 42's Irish clip land at <units-dir>/ga/42.npy.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = json.load(open(args.sentence_ids, encoding="utf-8"))
    ids = meta["sentence_ids"]

    out = {"id": [], "lang": [], "units": [], "n_raw": [], "n_dedup": []}
    for sid in ids:
        for lang in LANGS:
            path = os.path.join(args.units_dir, lang, f"{sid}.npy")
            if not os.path.isfile(path):
                raise SystemExit(f"tokenizer produced no units for {lang}/{sid}: {path}")
            raw = np.asarray(np.load(path), dtype=np.int64)
            deduped = dedup(raw, path)
            out["id"].append(sid)
            out["lang"].append(lang)
            out["units"].append(deduped.astype(np.uint16))
            out["n_raw"].append(len(raw))
            out["n_dedup"].append(len(deduped))

    pq.write_table(pa.table({
        "id": pa.array(out["id"], type=pa.int64()),
        "lang": pa.array(out["lang"], type=pa.string()),
        "units": pa.array(out["units"], type=pa.list_(pa.uint16())),
        "n_raw": pa.array(out["n_raw"], type=pa.int64()),
        "n_dedup": pa.array(out["n_dedup"], type=pa.int64()),
    }), args.out, compression="zstd")

    by_lang = {lang: [d for l, d in zip(out["lang"], out["n_dedup"]) if l == lang]
               for lang in LANGS}
    summary = {
        "sentences": len(ids),
        "rows": len(out["id"]),
        "deduplication": "consecutive duplicates removed",
        "median_dedup_units": {k: int(np.median(v)) for k, v in by_lang.items()},
        "max_dedup_units": {k: int(max(v)) for k, v in by_lang.items()},
        "sentence_ids": ids,
    }
    json.dump(summary, open(args.summary, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "sentence_ids"}, indent=2),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    e = sub.add_parser("export")
    e.add_argument("--data", required=True, help="fleurs_parallel_test.parquet")
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--out-dir", required=True, help="scratch dir for the wavs")
    e.add_argument("--manifest", required=True)
    e.add_argument("--sentence-ids", required=True)
    e.set_defaults(func=export)

    a = sub.add_parser("attach")
    a.add_argument("--units-dir", required=True,
                   help="--out-dir of audio_tknz_gpu.py; holds <lang>/<id>.npy")
    a.add_argument("--sentence-ids", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--summary", required=True)
    a.set_defaults(func=attach)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
