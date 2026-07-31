#!/usr/bin/env python3
"""Pack the parallel FLEURS ga_ie/en_us test split into ONE parquet (login node, LUMI).

FLEURS is parallel at the sentence level: the same sentence exists as Irish text, English
text, Irish speech and English speech. That is what lets the FLEURS benchmark cross input
modality (text/speech) with language mapping (same-language/cross-language) and score every
cell on the SAME sentences. So both languages go in one file, keyed on the FLEURS sentence
id — MEXA works off exactly these pairs and can reuse this file unchanged.

Audio is read straight out of the shipped test.tar.gz archives and stored as wav bytes: the
alternative is ~1500 loose small files on /scratch, which the parallel filesystem punishes
(LUMI small-files/inode quota). Same reasoning as iwslt_to_parquet.py.

Utterance selection (deterministic, and it keeps all six conditions parallel):
  * each sentence has 2-3 speaker recordings; take the FIRST by wav filename sort that is
    <= --max-seconds (30s, the Qwen2.5-Omni audio window);
  * drop the sentence entirely if either language has no qualifying recording — dropping it
    removes it from all six conditions at once, so the cells stay row-for-row identical.

Rows are sorted by sentence id and the first --shots of them are marked split="demo"; the
rest are split="test". Demos are therefore disjoint from the scored set by construction.

Usage (login node):
  python3 fleurs_to_parquet.py \
    --ga-tsv fleurs_ga_ie_test.tsv --en-tsv fleurs_en_us_test.tsv \
    --ga-tar data/fleurs/ga_ie_test.tar.gz --en-tar data/fleurs/en_us_test.tar.gz \
    --out data/fleurs_parallel_test.parquet
"""
import argparse
import array
import io
import os
import struct
import tarfile
import wave

import pyarrow as pa
import pyarrow.parquet as pq

# FLEURS tsv columns (no header). Column 3 is the dataset's own normalised transcription —
# scoring uses it rather than a homegrown normaliser, because Irish orthography (lenition,
# eclipsis, the sineadh fada, apostrophes) makes homegrown normalisation a silent error source.
ID, WAV, RAW, NORM, CHARS, N_SAMPLES, GENDER = range(7)
SAMPLE_RATE = 16000


def read_tsv(path):
    """{sentence_id: [row, ...]} — several speaker recordings share one sentence id."""
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            by_id.setdefault(int(cols[ID]), []).append(cols)
    return by_id


def pick(rows, max_seconds):
    """First recording by filename sort that fits the audio window; None if none does."""
    for row in sorted(rows, key=lambda r: r[WAV]):
        if int(row[N_SAMPLES]) / SAMPLE_RATE <= max_seconds:
            return row
    return None


def load_tar(path):
    """{basename: wav bytes} for one language's test archive."""
    out = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".wav"):
                out[os.path.basename(member.name)] = tar.extractfile(member).read()
    return out


def to_pcm16(raw_bytes):
    """FLEURS wavs are 32-bit FLOAT (format tag 3), which neither the stdlib wave module nor
    the eval-side decode_audio (asserts 16-bit) can read. Convert once here, at pack time, so
    everything downstream sees the same 16-bit PCM mono it already expects.

    Returns (wav_bytes, n_frames, sample_rate).
    """
    if raw_bytes[:4] != b"RIFF" or raw_bytes[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    fmt_tag = channels = sample_rate = bits = None
    data = None
    pos = 12                                   # walk the chunk list; 'fact' etc. may precede 'data'
    while pos + 8 <= len(raw_bytes):
        chunk_id = raw_bytes[pos:pos + 4]
        size = struct.unpack("<I", raw_bytes[pos + 4:pos + 8])[0]
        body = raw_bytes[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt_tag, channels, sample_rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            data = body
        pos += 8 + size + (size & 1)           # chunks are word-aligned
    if fmt_tag is None or data is None:
        raise ValueError("wav missing fmt or data chunk")

    if fmt_tag == 3 and bits == 32:
        floats = array.array("f")
        floats.frombytes(data[:len(data) - len(data) % 4])
        samples = array.array("h", (
            max(-32768, min(32767, int(round(v * 32767.0)))) for v in floats))
    elif fmt_tag == 1 and bits == 16:
        samples = array.array("h")
        samples.frombytes(data[:len(data) - len(data) % 2])
    else:
        raise ValueError("unsupported wav format tag=%s bits=%s" % (fmt_tag, bits))
    if channels == 2:                          # not expected in FLEURS, but cheap to handle
        samples = array.array("h", (
            (samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples) - 1, 2)))
        channels = 1

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue(), len(samples), sample_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga-tsv", required=True)
    ap.add_argument("--en-tsv", required=True)
    ap.add_argument("--ga-tar", required=True)
    ap.add_argument("--en-tar", required=True)
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="Qwen2.5-Omni audio window; longer recordings are not used")
    ap.add_argument("--shots", type=int, default=3,
                    help="lowest N sentence ids are marked split=demo")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ga_tsv, en_tsv = read_tsv(args.ga_tsv), read_tsv(args.en_tsv)
    shared = sorted(set(ga_tsv) & set(en_tsv))
    print(f"[tsv] {len(ga_tsv)} ga sentence ids, {len(en_tsv)} en, {len(shared)} in both",
          flush=True)

    chosen, dropped = [], []
    for sid in shared:
        g, e = pick(ga_tsv[sid], args.max_seconds), pick(en_tsv[sid], args.max_seconds)
        if g is None or e is None:
            dropped.append(sid)
            continue
        chosen.append((sid, g, e))
    print(f"[select] kept {len(chosen)} sentences, dropped {len(dropped)} with no recording "
          f"<= {args.max_seconds}s in one language: {dropped}", flush=True)

    print(f"[audio] reading {args.ga_tar} ...", flush=True)
    ga_wavs = load_tar(args.ga_tar)
    print(f"[audio] reading {args.en_tar} ...", flush=True)
    en_wavs = load_tar(args.en_tar)

    cols = {k: [] for k in (
        "id", "split", "ga_wav", "en_wav", "ga_audio", "en_audio",
        "ga_raw", "ga_norm", "en_raw", "en_norm",
        "ga_gender", "en_gender", "ga_duration_s", "en_duration_s")}
    for index, (sid, g, e) in enumerate(chosen):
        ga_bytes, en_bytes = ga_wavs.get(g[WAV]), en_wavs.get(e[WAV])
        # A missing wav would silently shrink one language's column and break the pairing,
        # which is the one thing this whole design depends on. Fail loudly instead.
        assert ga_bytes is not None, f"sentence {sid}: {g[WAV]} missing from {args.ga_tar}"
        assert en_bytes is not None, f"sentence {sid}: {e[WAV]} missing from {args.en_tar}"
        ga_bytes, ga_frames, ga_sr = to_pcm16(ga_bytes)
        en_bytes, en_frames, en_sr = to_pcm16(en_bytes)
        cols["id"].append(sid)
        cols["split"].append("demo" if index < args.shots else "test")
        cols["ga_wav"].append(g[WAV])
        cols["en_wav"].append(e[WAV])
        cols["ga_audio"].append(ga_bytes)
        cols["en_audio"].append(en_bytes)
        cols["ga_raw"].append(g[RAW])
        cols["ga_norm"].append(g[NORM])
        cols["en_raw"].append(e[RAW])
        cols["en_norm"].append(e[NORM])
        cols["ga_gender"].append(g[GENDER])
        cols["en_gender"].append(e[GENDER])
        cols["ga_duration_s"].append(ga_frames / float(ga_sr))
        cols["en_duration_s"].append(en_frames / float(en_sr))

    pq.write_table(pa.table(cols), args.out)
    hours = (sum(cols["ga_duration_s"]) + sum(cols["en_duration_s"])) / 3600
    n_demo = sum(1 for s in cols["split"] if s == "demo")
    print(f"wrote {len(cols['id'])} sentence pairs to {args.out} "
          f"({n_demo} demo + {len(cols['id']) - n_demo} test; {hours:.2f} h of audio; "
          f"longest ga {max(cols['ga_duration_s']):.1f}s, "
          f"en {max(cols['en_duration_s']):.1f}s)")


if __name__ == "__main__":
    main()
