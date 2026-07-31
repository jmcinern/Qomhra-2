#!/usr/bin/env python3
"""All six MEXA pairs with and without centering, so the gate can be read off directly.

Centering subtracts the mean of the 100 embeddings within each object before the cosine.
It removes the one direction every sequence shares and leaves the sentence-specific part.
It changes nothing about which sentences are parallel and it is applied identically to both
sides of every pair, so it cannot manufacture alignment that is not there -- the self-test's
random-matrix and collapsed-matrix cases still score ~0 under it.
"""
import sys

import numpy as np

CONDITIONS = (
    ("text_ga~text_en", "text_ga", "text_en"),
    ("speech_ga~text_ga", "speech_ga", "text_ga"),
    ("speech_en~text_en", "speech_en", "text_en"),
    ("speech_ga~text_en", "speech_ga", "text_en"),
    ("speech_en~text_ga", "speech_en", "text_ga"),
    ("speech_ga~speech_en", "speech_ga", "speech_en"),
)


def mexa(a, b):
    a = np.asarray(a, np.float64).copy()
    b = np.asarray(b, np.float64).copy()
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    if (a_norm == 0).any() or (b_norm == 0).any():
        return float("nan")
    a /= a_norm
    b /= b_norm
    m = a @ b.T
    d = np.diag(m)
    off = m.copy()
    np.fill_diagonal(off, -np.inf)
    return float(((d > off.max(1)) & (d > off.max(0))).mean())


def run(path, pooling="weighted"):
    data = np.load(path)
    print(f"\n{path}  [{pooling}]")
    print(f"  {'pair':22s} {'raw':>6} {'centered':>9}   best layer (centered)")
    results = []
    for name, left, right in CONDITIONS:
        A, B = data[f"{left}.{pooling}"], data[f"{right}.{pooling}"]
        raw_best = cent_best = 0.0
        cent_layer = 0
        for layer in range(A.shape[0]):
            a, b = A[layer].astype(np.float64), B[layer].astype(np.float64)
            raw_best = max(raw_best, mexa(a, b))
            c = mexa(a - a.mean(0, keepdims=True), b - b.mean(0, keepdims=True))
            if np.isfinite(c) and c > cent_best:
                cent_best, cent_layer = c, layer
        results.append((name, raw_best, cent_best, cent_layer))

    top = max(results, key=lambda r: r[2])[0]
    for name, raw_best, cent_best, cent_layer in results:
        mark = "  <-- highest" if name == top else ""
        print(f"  {name:22s} {raw_best:>6.2f} {cent_best:>9.2f}   L{cent_layer}{mark}")
    return top


def main():
    tops = {}
    for path in sys.argv[1:]:
        tops[path] = run(path)
    print()
    for path, top in tops.items():
        verdict = "GATE PASSES" if top == "speech_ga~text_ga" else "gate fails"
        print(f"{verdict}: highest centered pair in {path.split('/')[-1]} is {top}")


if __name__ == "__main__":
    main()
