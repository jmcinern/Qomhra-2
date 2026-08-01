#!/usr/bin/env python3
"""Audit source clues and group leakage in the discrete-ASR utterance split."""

import argparse
import collections
import json
import os
import re

import pyarrow.parquet as pq


def family_and_group(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.startswith("teanglann_"):
        match = re.match(r"(teanglann_[^_]+)", stem)
        return "teanglann", match.group(1) if match else stem
    if "_grownups_" in stem:
        return "grownups", stem.split("_grownups_", 1)[0]
    if "_extended_corpus220525_" in stem:
        return "extended_corpus220525", stem.split(
            "_extended_corpus220525_", 1
        )[0]
    if "_corpus_beag_" in stem:
        return "corpus_beag", stem.split("_corpus_beag_", 1)[0]
    if stem.startswith("lnc_corpasbeag_"):
        return "corpus_beag", "lnc_corpasbeag"
    match = re.match(r"(iear\d+_spk\d+)", stem)
    if match:
        return "iear", match.group(1)
    match = re.match(r"([CMU]\d{4})-", stem)
    if match:
        return "CMU_segmented", match.group(1)
    if stem.startswith("dms_ga_"):
        return "dms_ga", re.sub(r"_\d+$", "", stem)
    if stem.startswith("seadna_"):
        match = re.match(r"(seadna_\d+)", stem)
        return "seadna", match.group(1) if match else "seadna"
    if stem.startswith("coc_"):
        return "coc", re.sub(r"_\d+$", "", stem)
    match = re.match(r"(\d+_\d+)_", stem)
    if match:
        return "numeric_recording", match.group(1)
    return stem.split("_", 1)[0], stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--sample", type=int, default=30)
    args = parser.parse_args()

    metadata = {}
    with open(args.metadata, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata[row["filename"]] = row["audio_filepath"]

    original = {}
    if args.source_manifest:
        with open(args.source_manifest, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                original[os.path.basename(row["audio_filepath"])] = row["audio_filepath"]

    by_split = collections.defaultdict(list)
    parquet = pq.ParquetFile(args.examples)
    for batch in parquet.iter_batches(columns=["filename", "split"]):
        for filename, split in zip(batch.column(0), batch.column(1)):
            by_split[split.as_py()].append(filename.as_py())

    train_groups = {family_and_group(name) for name in by_split["train"]}
    validation_groups = [family_and_group(name) for name in by_split["validation"]]
    family_counts = collections.Counter(family for family, _ in validation_groups)
    validation_unique_groups = set(validation_groups)
    crossed = validation_unique_groups & train_groups
    crossed_rows = sum(group in train_groups for group in validation_groups)

    print(f"train_rows={len(by_split['train'])}")
    print(f"validation_rows={len(by_split['validation'])}")
    print(f"validation_unique_name_groups={len(validation_unique_groups)}")
    print(
        "validation_groups_also_in_train="
        f"{len(crossed)}/{len(validation_unique_groups)} "
        f"({len(crossed) / max(len(validation_unique_groups), 1):.1%})"
    )
    print(
        "validation_rows_with_group_in_train="
        f"{crossed_rows}/{len(validation_groups)} "
        f"({crossed_rows / max(len(validation_groups), 1):.1%})"
    )
    print("validation_families:")
    for family, count in family_counts.most_common():
        print(f"  {family}\t{count}")

    print("validation_samples:")
    for filename in sorted(by_split["validation"])[: args.sample]:
        family, group = family_and_group(filename)
        print(
            json.dumps(
                {
                    "filename": filename,
                    "family": family,
                    "group": group,
                    "group_in_train": (family, group) in train_groups,
                    "packed_source_path": metadata.get(filename),
                    "setanta_source_path": original.get(filename),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
