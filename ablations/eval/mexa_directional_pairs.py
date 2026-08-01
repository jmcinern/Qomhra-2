#!/usr/bin/env python3
"""Directional retrieval diagnostics for all ordered MEXA object pairs.

MEXA itself is symmetric under matrix transpose. Retrieval is not, so this companion
reports both A->B and B->A using centered embeddings and the same 100 paired rows.
"""
import sys

import numpy as np


OBJECTS = ("text_ga", "text_en", "speech_ga", "speech_en")


def similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    if (a_norm == 0).any() or (b_norm == 0).any():
        return None
    a /= a_norm
    b /= b_norm
    return a @ b.T


def best_direction(a, b):
    best = None
    for layer in range(a.shape[0]):
        matrix = similarity(a[layer], b[layer])
        if matrix is None:
            continue
        order = np.argsort(-matrix, axis=1)
        ranks = np.asarray([
            int(np.flatnonzero(order[row] == row)[0]) + 1
            for row in range(len(order))
        ])
        result = (
            float((ranks == 1).mean()),
            float((ranks <= 5).mean()),
            float(np.median(ranks)),
            layer,
        )
        if best is None or result[:2] > best[:2]:
            best = result
    if best is None:
        raise RuntimeError("all layers collapsed after centering")
    return best


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: mexa_directional_pairs.py MODEL.npz [...]")
    for path in sys.argv[1:]:
        data = np.load(path)
        print(f"\n{path} [centered, weighted, best layer per direction]")
        print(f"{'direction':24s} {'top1':>6s} {'top5':>6s} {'median':>7s} {'layer':>6s}")
        for left in OBJECTS:
            for right in OBJECTS:
                if left == right:
                    continue
                top1, top5, median, layer = best_direction(
                    data[f"{left}.weighted"], data[f"{right}.weighted"]
                )
                print(
                    f"{left + '->' + right:24s} {top1:6.2f} {top5:6.2f} "
                    f"{median:7.1f} {layer:6d}"
                )


if __name__ == "__main__":
    main()
