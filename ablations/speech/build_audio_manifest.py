#!/usr/bin/env python3
"""
Ablations Job-1 — build (and shard) the mHuBERT tokenization manifest.

Run on LUMI AFTER the transfer, against the landed layout:
    <audio-root>/<dest_label>/<file>.wav   (+ optional <dest_label>/wavs.dur)

Emits a manifest.tsv (header: dest_label\tfilename\tduration_s\tsrc_path) with one
row per .wav actually present on disk, so the manifest is guaranteed consistent
with what audio-tknz.py will resolve. Durations come from each dir's wavs.dur when
present (fast, no decode); otherwise from soundfile.info (needs --decode-fallback).

With --shard-hours, also writes manifest_shardNN.tsv split by cumulative duration
(files sorted longest-first so shards are balanced) for a SLURM job array.

Usage (on LUMI):
    python build_audio_manifest.py \
        --audio-root /scratch/project_465002364/audio/unlabelled_full \
        --out-manifest /scratch/project_465002364/audio/unlabelled_full/manifest.tsv \
        --shard-hours 800
"""
import argparse
import csv
import os
import sys


def read_wavs_dur(dur_path):
    """Return {basename: duration_s} from a wavs.dur file, or {} if absent."""
    durs = {}
    if not os.path.isfile(dur_path):
        return durs
    with open(dur_path, newline="") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # "filename<sp>duration"; filenames may contain spaces -> split on last ws
            i = line.rfind(" ")
            if i < 0:
                continue
            name, dur = line[:i].strip(), line[i + 1:].strip()
            try:
                durs[os.path.basename(name)] = float(dur)
            except ValueError:
                continue
    return durs


def duration_fallback(path):
    """Decode-based duration (frames / samplerate). Only used with --decode-fallback."""
    import soundfile as sf
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def collect_rows(audio_root, decode_fallback):
    rows = []  # (dest_label, filename, duration_s, src_path)
    n_missing_dur = 0
    labels = sorted(d for d in os.listdir(audio_root)
                    if os.path.isdir(os.path.join(audio_root, d)))
    for label in labels:
        ldir = os.path.join(audio_root, label)
        durs = read_wavs_dur(os.path.join(ldir, "wavs.dur"))
        wavs = sorted(f for f in os.listdir(ldir) if f.lower().endswith(".wav"))
        for name in wavs:
            src = os.path.join(ldir, name)
            if name in durs:
                dur = durs[name]
            elif decode_fallback:
                dur = duration_fallback(src)
            else:
                n_missing_dur += 1
                dur = ""  # unknown; sharding will treat as 0
            rows.append((label, name, dur, src))
    if n_missing_dur:
        print(f"WARN: {n_missing_dur} files had no wavs.dur entry "
              f"(pass --decode-fallback to fill durations)", file=sys.stderr)
    return rows


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dest_label", "filename", "duration_s", "src_path"])
        w.writerows(rows)


def write_shards(out_manifest, rows, shard_hours):
    """Split rows into shards of ~shard_hours each (longest-first bin-packing)."""
    def dur(r):
        try:
            return float(r[2])
        except (ValueError, TypeError):
            return 0.0
    ordered = sorted(rows, key=dur, reverse=True)
    cap = shard_hours * 3600.0
    shards, cur, cur_s = [], [], 0.0
    for r in ordered:
        if cur and cur_s + dur(r) > cap:
            shards.append(cur)
            cur, cur_s = [], 0.0
        cur.append(r)
        cur_s += dur(r)
    if cur:
        shards.append(cur)
    base = out_manifest[:-4] if out_manifest.endswith(".tsv") else out_manifest
    for i, sh in enumerate(shards):
        write_manifest(f"{base}_shard{i:02d}.tsv", sh)
        print(f"  shard{i:02d}: {len(sh)} files, "
              f"{sum(dur(r) for r in sh) / 3600:.1f} h")
    print(f"wrote {len(shards)} shards -> {base}_shardNN.tsv")
    return len(shards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--shard-hours", type=float, default=0,
                    help="if >0, also emit manifest_shardNN.tsv split by duration")
    ap.add_argument("--decode-fallback", action="store_true",
                    help="use soundfile to get duration when wavs.dur is missing")
    args = ap.parse_args()

    rows = collect_rows(args.audio_root, args.decode_fallback)
    write_manifest(args.out_manifest, rows)

    total_h = sum(float(r[2]) for r in rows if r[2] not in ("", None)) / 3600.0
    n_labels = len(set(r[0] for r in rows))
    print(f"manifest: {len(rows)} files, {n_labels} labels, {total_h:.1f} h "
          f"-> {args.out_manifest}")

    # Per-label breakdown
    by_label = {}
    for label, _, d, _ in rows:
        try:
            dv = float(d)
        except (ValueError, TypeError):
            dv = 0.0
        c, s = by_label.get(label, (0, 0.0))
        by_label[label] = (c + 1, s + dv)
    print("per-label (files, hours):")
    for label in sorted(by_label):
        c, s = by_label[label]
        print(f"  {label:40s} {c:6d}  {s / 3600:8.1f} h")

    if args.shard_hours > 0:
        write_shards(args.out_manifest, rows, args.shard_hours)


if __name__ == "__main__":
    main()
