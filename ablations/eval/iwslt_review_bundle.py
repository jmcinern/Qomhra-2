#!/usr/bin/env python3
"""Build a deterministic local/Hugging Face review bundle from IWSLT result JSONs."""

import argparse
import csv
import json
import os
import random


def load_result(spec):
    if "=" not in spec:
        raise ValueError(f"result must be LABEL=JSON_PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if label in document:
        source_label = label
    elif len(document) == 1:
        source_label = next(iter(document))
    else:
        raise ValueError(
            f"{path} has no top-level result named {label!r} and is not single-model")
    return label, os.path.abspath(path), source_label, document[source_label]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="IWSLT parquet containing WAV bytes")
    parser.add_argument("--result", action="append", required=True,
                        help="ordered output column as LABEL=JSON_PATH; repeat per model")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    loaded = [load_result(spec) for spec in args.result]
    labels = [label for label, _, _, _ in loaded]
    if len(labels) != len(set(labels)):
        raise ValueError("result labels must be unique")

    row_maps = {}
    reference_rows = None
    for label, _, _, result in loaded:
        rows = result["rows"]
        mapping = {row["name"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"duplicate utterance name in {label}")
        row_maps[label] = mapping
        if reference_rows is None:
            reference_rows = rows
        elif set(mapping) != {row["name"] for row in reference_rows}:
            raise ValueError(f"utterance set for {label} differs from the first result")

    names = [row["name"] for row in reference_rows]
    if not 0 < args.sample_size <= len(names):
        raise ValueError(f"sample size must be between 1 and {len(names)}")
    selected = sorted(random.Random(args.seed).sample(names, args.sample_size))

    import pyarrow.parquet as pq
    table = pq.read_table(args.data, columns=["name", "audio", "eng", "duration_s"])
    parquet_rows = {row["name"]: row for row in table.to_pylist() if row["name"] in selected}
    missing_audio = sorted(set(selected) - set(parquet_rows))
    if missing_audio:
        raise ValueError(f"selected audio missing from parquet: {missing_audio[:5]}")

    audio_dir = os.path.join(args.out, "test", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    model_fields = [field for label in labels
                    for field in (f"{label}_hyp", f"{label}_raw")]
    fields = (["item_no", "utterance_id", "audio_file", "duration_s",
               "english_reference"] + model_fields +
              ["annotation", "exclude_reason"])
    review_rows = []
    for item_no, name in enumerate(selected, start=1):
        source = parquet_rows[name]
        refs = {row_maps[label][name]["ref"] for label in labels}
        refs.add(source["eng"])
        if len(refs) != 1:
            raise ValueError(f"reference mismatch for {name}")
        with open(os.path.join(audio_dir, name), "wb") as handle:
            handle.write(source["audio"])
        row = {
            "item_no": item_no,
            "utterance_id": name,
            "audio_file": f"test/audio/{name}",
            "duration_s": round(source["duration_s"], 3),
            "english_reference": source["eng"],
            "annotation": "",
            "exclude_reason": "",
        }
        for label in labels:
            model_row = row_maps[label][name]
            row[f"{label}_hyp"] = model_row["hyp"]
            row[f"{label}_raw"] = model_row["raw"]
        review_rows.append(row)

    csv_path = os.path.join(args.out, "review.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    metadata_path = os.path.join(args.out, "test", "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        for row in review_rows:
            viewer_row = dict(row)
            viewer_row["file_name"] = viewer_row.pop("audio_file").removeprefix("test/")
            handle.write(json.dumps(viewer_row, ensure_ascii=False) + "\n")

    manifest = {
        "source_data": os.path.abspath(args.data),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "selection_method": "uniform sample without replacement, then filename sort",
        "selected_utterances": selected,
        "review_columns": fields,
        "results": {label: {"json": path, "source_result_label": source_label,
                            "run": result.get("run")}
                    for label, path, source_label, result in loaded},
    }
    with open(os.path.join(args.out, "selection.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    model_columns = "\n".join(
        f"- `{label}_hyp` and `{label}_raw`" for label in labels)
    readme = f"""---
license: cc-by-nc-sa-4.0
language:
- ga
- en
pretty_name: IWSLT Irish-English model-output review sample
size_categories:
- n<1K
---

# IWSLT Irish-English review sample

This private review bundle contains a deterministic sample of {args.sample_size} IWSLT
Irish-to-English dev utterances, their English references, and model outputs. Audio is
stored as 16-bit WAV and rendered by the Hugging Face AudioFolder dataset viewer.

## Model-output columns

{model_columns}

`annotation` and `exclude_reason` are blank reviewer fields. No Irish transcript or
Irish-versus-English overlap analysis is included in this bundle.

## Selection

Uniform sampling without replacement with seed `{args.seed}`, followed by filename sort.
Exact names and source-result provenance are recorded in `selection.json`.

## Licence and source

The source IWSLT Irish-English data is distributed under CC BY-NC-SA 4.0 by the Insight
Centre for Data Analytics / University of Galway and ADAPT Centre. This derived review
bundle retains that licence and is for non-commercial evaluation and annotation.
Source: https://github.com/shashwatup9k/iwslt2023_ga-eng
"""
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(readme)
    with open(os.path.join(args.out, ".gitattributes"), "w", encoding="utf-8") as handle:
        handle.write("*.wav filter=lfs diff=lfs merge=lfs -text\n")
    print(f"wrote {len(review_rows)} rows, {len(labels)} model-output pairs to {args.out}")


if __name__ == "__main__":
    main()
