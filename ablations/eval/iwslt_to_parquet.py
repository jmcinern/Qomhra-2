#!/usr/bin/env python3
"""Pack the IWSLT2023 ga-eng dev set into ONE parquet (login node, LUMI).

The dev set is ~hundreds of small wavs. Staging them loose on /scratch stresses the
parallel-filesystem metadata servers (LUMI storage docs: small-files/inode quota), so we
fold the whole split into a single file: audio bytes + English reference + duration.

Source (clone + pin on the login node, which has network; compute nodes are offline):
  git clone https://github.com/shashwatup9k/iwslt2023_ga-eng <root>
  git -C <root> checkout 0384674

Pairing is positional: sorted(dev/wav/*.wav) lines up with dev/txt/dev.eng in order. A
count mismatch scrambles every reference, so we assert equal counts and fail loudly.

Usage (login node):
  python3 iwslt_to_parquet.py \
    --root /scratch/project_465002364/Qomhra-2/ablations/eval/data/iwslt2023_ga-eng \
    --split dev \
    --out  /scratch/project_465002364/Qomhra-2/ablations/eval/data/iwslt2023_dev.parquet
"""
import argparse
import glob
import os
import wave

import pyarrow as pa
import pyarrow.parquet as pq


def wav_duration_s(path):
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="cloned iwslt2023_ga-eng repo")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True, help="output .parquet")
    args = ap.parse_args()

    wav_dir = os.path.join(args.root, args.split, "wav")
    eng_path = os.path.join(args.root, args.split, "txt", f"{args.split}.eng")
    wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    with open(eng_path, encoding="utf-8") as f:
        refs = [ln.rstrip("\n") for ln in f]

    assert len(wavs) == len(refs), (
        f"count mismatch: {len(wavs)} wavs vs {len(refs)} eng lines — positional pairing "
        f"would scramble every reference. Check {wav_dir} and {eng_path}.")

    names, audio, engs, durs = [], [], [], []
    for path, ref in zip(wavs, refs):
        with open(path, "rb") as f:
            audio.append(f.read())
        names.append(os.path.basename(path))
        engs.append(ref)
        durs.append(wav_duration_s(path))

    table = pa.table({
        "name": names,
        "audio": audio,          # raw wav bytes
        "eng": engs,             # English reference translation
        "duration_s": durs,
    })
    pq.write_table(table, args.out)
    print(f"wrote {len(names)} utterances to {args.out} "
          f"(total {sum(durs)/3600:.2f} h, longest {max(durs):.1f}s)")


if __name__ == "__main__":
    main()
