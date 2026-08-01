#!/usr/bin/env python3
"""Decompose the gap between top-1 retrieval and the MEXA score on ONE matrix.

MEXA counts sentence i only if similarity[i][i] is the strict maximum of row i AND of
column i. Top-1 retrieval asks only about the row. So the two numbers differ by exactly
the sentences that win their row and then lose their column, and this prints that split
instead of leaving it to be inferred.

It also prints the hub structure: when every embedding is nearly identical, a handful of
text vectors become the nearest neighbour of many speech clips. Those hubs are what take
the column away from the sentences that legitimately won their row.
"""
import sys

import numpy as np


def sim(a, b):
    a = np.asarray(a, np.float64).copy()
    b = np.asarray(b, np.float64).copy()
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def decompose(matrix):
    n = matrix.shape[0]
    diagonal = np.diag(matrix)
    off = matrix.copy()
    np.fill_diagonal(off, -np.inf)
    row_ok = diagonal > off.max(axis=1)     # own transcript is the best match for this clip
    col_ok = diagonal > off.max(axis=0)     # this clip is the best match for its transcript
    return row_ok, col_ok


def main():
    path, left, right = sys.argv[1], sys.argv[2], sys.argv[3]
    pooling = sys.argv[4] if len(sys.argv) > 4 else "weighted"
    data = np.load(path)
    A, B = data[f"{left}.{pooling}"], data[f"{right}.{pooling}"]

    print(f"{path}\n{left} -> {right}  [{pooling}]\n")
    print(f"{'layer':>5} {'row wins':>9} {'col wins':>9} {'BOTH=MEXA':>10} "
          f"{'top hub':>8} {'hub owns':>9}")
    best = None
    for layer in range(A.shape[0]):
        matrix = sim(A[layer], B[layer])
        row_ok, col_ok = decompose(matrix)
        both = row_ok & col_ok
        # How concentrated are the nearest neighbours? If one text vector is the argmax
        # for many clips, it is a hub and it blocks every one of them but itself.
        nearest = matrix.argmax(axis=1)
        counts = np.bincount(nearest, minlength=matrix.shape[0])
        if best is None or row_ok.sum() > best[1]:
            best = (layer, int(row_ok.sum()), int(col_ok.sum()), int(both.sum()),
                    int(counts.max()), int(counts.argmax()))
        if row_ok.sum() or both.sum():
            print(f"{layer:>5} {row_ok.sum():>9} {col_ok.sum():>9} {both.sum():>10} "
                  f"{counts.argmax():>8} {counts.max():>9}")

    layer, rows, cols, both, hub_size, hub_id = best
    n = A.shape[1]
    print(f"\nbest row layer L{layer}:")
    print(f"  {rows}/{n} clips found their own transcript first   -> top1 = {rows / n:.2f}")
    print(f"  {cols}/{n} transcripts found their own clip first")
    print(f"  {both}/{n} did BOTH                                  -> MEXA = {both / n:.2f}")
    print(f"  so {rows - both} clip(s) won their row and then lost their column")
    print(f"  most-attracting text vector is sentence {hub_id}: it is the top match for "
          f"{hub_size}/{n} clips")


if __name__ == "__main__":
    main()
