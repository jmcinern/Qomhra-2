#!/usr/bin/env python3
"""
Ablations Job-1 — accounting for the tokenized speech units.

Walk <units-dir>/<dest_label>/*.npy, report per-label file count / total units /
hours (units / 180000 at 50 Hz), grand totals, and a few dtype/range sanity checks.
Writes meta.json next to the units. Run on LUMI (numpy only).

    python audio_units_meta.py --units-dir /scratch/project_465002364/audio/units_full
"""
import argparse
import json
import os
import sys

import numpy as np

UNITS_PER_HOUR = 180000  # 50 Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    by_label = {}          # label -> [files, units]
    total_files = total_units = 0
    bad_dtype = bad_range = 0
    checked = 0

    for label in sorted(os.listdir(args.units_dir)):
        ldir = os.path.join(args.units_dir, label)
        if not os.path.isdir(ldir):
            continue
        f_cnt = u_cnt = 0
        for name in os.listdir(ldir):
            if not name.endswith(".npy"):
                continue
            arr = np.load(os.path.join(ldir, name), mmap_mode="r")
            f_cnt += 1
            u_cnt += int(arr.shape[0])
            # sanity-check a small sample of files (full mmap scan is cheap for range)
            if checked < 200:
                if arr.dtype != np.uint16:
                    bad_dtype += 1
                if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= 1000):
                    bad_range += 1
                checked += 1
        by_label[label] = [f_cnt, u_cnt]
        total_files += f_cnt
        total_units += u_cnt

    meta = {
        "units_per_hour": UNITS_PER_HOUR,
        "total_files": total_files,
        "total_units": total_units,
        "total_hours": round(total_units / UNITS_PER_HOUR, 2),
        "sanity": {"checked": checked, "bad_dtype": bad_dtype, "bad_range": bad_range},
        "by_label": {
            k: {"files": v[0], "units": v[1], "hours": round(v[1] / UNITS_PER_HOUR, 2)}
            for k, v in sorted(by_label.items())
        },
    }

    out = args.out or os.path.join(args.units_dir, "meta.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"labels: {len(by_label)}  files: {total_files}  "
          f"units: {total_units:,}  hours: {meta['total_hours']}")
    print(f"sanity: checked={checked} bad_dtype={bad_dtype} bad_range={bad_range}")
    for k in sorted(by_label):
        c, u = by_label[k]
        print(f"  {k:40s} {c:6d}  {u/UNITS_PER_HOUR:8.1f} h")
    print(f"-> {out}")
    if bad_dtype or bad_range:
        print("WARN: sanity checks failed", file=sys.stderr)


if __name__ == "__main__":
    main()
