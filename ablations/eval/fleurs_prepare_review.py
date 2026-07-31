#!/usr/bin/env python3
"""Flatten FLEURS evaluation JSONs into reviewable numeric and raw CSVs."""
import argparse
import csv
import glob
import json
from pathlib import Path


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--numbers-out", required=True)
    parser.add_argument("--raw-out", required=True)
    args = parser.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(glob.glob(pattern))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("no FLEURS JSON inputs matched")

    numbers = []
    raw = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        run = payload["run"]
        summary_row = {
            "model": run.get("label"),
            "checkpoint": run.get("checkpoint"),
            "n": run.get("n"),
            "source_file": str(Path(path).resolve()),
        }
        for condition, result in payload["conditions"].items():
            spec = result.get("spec", {})
            summary = result.get("summary", {})
            metric = spec.get("metric")
            if metric and metric in summary:
                summary_row[f"{condition}_{metric}"] = summary[metric]
            for secondary in ("wer", "cer", "chrf", "chrfpp"):
                if secondary in summary:
                    summary_row[f"{condition}_{secondary}"] = summary[secondary]
            diagnostics = summary.get("diagnostics", {})
            for key, value in diagnostics.items():
                if not isinstance(value, (dict, list)):
                    summary_row[f"{condition}_{key}"] = value
            for row in result.get("rows", []):
                item = {
                    "model": run.get("label"),
                    "checkpoint": run.get("checkpoint"),
                    "condition": condition,
                    "pair": spec.get("pair"),
                    "input_modality": spec.get("input_modality"),
                    "source_lang": spec.get("source_lang"),
                    "target_lang": spec.get("target_lang"),
                    "metric": metric,
                    "source_file": str(Path(path).resolve()),
                }
                for key, value in row.items():
                    item[key] = (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                raw.append(item)
        numbers.append(summary_row)

    write_csv(args.numbers_out, numbers)
    write_csv(args.raw_out, raw)


if __name__ == "__main__":
    main()
