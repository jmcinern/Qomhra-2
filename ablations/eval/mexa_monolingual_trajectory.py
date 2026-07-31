#!/usr/bin/env python3
"""Layer-wise diagnostics for all text/speech MEXA pairs.

Raw MEXA's mutual-nearest-neighbour test can remain zero while the paired item moves
substantially up the retrieval ranking.  This report therefore retains published raw
MEXA, and adds centered MEXA plus directional top-1/top-5 retrieval at every layer.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAIRS = (
    ("speech_en~text_en", "speech_en", "text_en"),
    ("speech_ga~text_ga", "speech_ga", "text_ga"),
    ("text_ga~text_en", "text_ga", "text_en"),
    ("speech_ga~speech_en", "speech_ga", "speech_en"),
    ("speech_ga~text_en", "speech_ga", "text_en"),
    ("speech_en~text_ga", "speech_en", "text_ga"),
)


def similarity(a, b, centered):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if centered:
        a = a - a.mean(axis=0, keepdims=True)
        b = b - b.mean(axis=0, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def mexa(matrix):
    diagonal = np.diag(matrix)
    off = matrix.copy()
    np.fill_diagonal(off, -np.inf)
    return float(((diagonal > off.max(axis=1)) & (diagonal > off.max(axis=0))).mean())


def mexa_at_k(matrix, k):
    """Fraction whose paired item ranks in the top-k in both directions."""
    row_order = np.argsort(-matrix, axis=1)
    column_order = np.argsort(-matrix, axis=0)
    row_ranks = np.empty(matrix.shape[0], dtype=np.int64)
    column_ranks = np.empty(matrix.shape[0], dtype=np.int64)
    for index in range(matrix.shape[0]):
        row_ranks[index] = int(np.flatnonzero(row_order[index] == index)[0]) + 1
        column_ranks[index] = int(np.flatnonzero(column_order[:, index] == index)[0]) + 1
    return float(((row_ranks <= k) & (column_ranks <= k)).mean())


def retrieval(matrix):
    order = np.argsort(-matrix, axis=1)
    ranks = np.asarray([
        int(np.flatnonzero(order[row] == row)[0]) + 1 for row in range(len(order))
    ])
    return float((ranks == 1).mean()), float((ranks <= 5).mean()), float(np.median(ranks))


def metadata(path):
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    with sidecar.open(encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(path):
    meta = metadata(path)
    data = np.load(path)
    rows = []
    for condition, left, right in PAIRS:
        a = data[f"{left}.weighted"]
        b = data[f"{right}.weighted"]
        for layer in range(a.shape[0]):
            raw = similarity(a[layer], b[layer], centered=False)
            centered = similarity(a[layer], b[layer], centered=True)
            top1_lr, top5_lr, median_lr = retrieval(centered)
            top1_rl, top5_rl, median_rl = retrieval(centered.T)
            rows.append({
                "label": meta.get("label", path.stem),
                "model": meta.get("model", path.stem),
                "step": meta.get("step"),
                "train_fraction": meta.get("train_fraction"),
                "condition": condition,
                "layer": layer,
                "n_sentences": a.shape[1],
                "raw_mexa": mexa(raw),
                "centered_mexa": mexa(centered),
                "centered_mexa_at10": mexa_at_k(centered, 10),
                "left_to_right_top1": top1_lr,
                "left_to_right_top5": top5_lr,
                "left_to_right_median_rank": median_lr,
                "right_to_left_top1": top1_rl,
                "right_to_left_top5": top5_rl,
                "right_to_left_median_rank": median_rl,
                "source": str(path),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.embeddings:
        print(f"[mono] {path}", flush=True)
        rows.extend(evaluate(path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[mono] wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
