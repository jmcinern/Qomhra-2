#!/usr/bin/env python3
"""Diagnostic: is the true counterpart merely not the STRICT mutual max, or nowhere at all?

MEXA is pass/fail per sentence -- a sentence counts only if its true counterpart is the
largest entry in both its row and its column. Retrieval rank is graded instead: if the true
match sits at rank 2 of 100, the representation carries alignment that MEXA's criterion
refuses to credit; if it sits near rank 50, there is nothing there at all.

That distinction is exactly what a MEXA score of ~0 cannot make on its own, and it is the
difference between "the probe is miswired" and "the criterion is too strict for this model".

Also reports the mean off-diagonal cosine within one side. Near +1.0 means the encoder maps
every sentence to nearly the same vector, which drives MEXA to 0 for a reason that has
nothing to do with cross-modal alignment.

    python mexa_retrieval_diag.py units.npz continuous.npz
"""
import sys

import numpy as np

CONDITIONS = (
    ("speech_ga->text_ga", "speech_ga", "text_ga"),
    ("speech_en->text_en", "speech_en", "text_en"),
    ("speech_ga->speech_en", "speech_ga", "speech_en"),
    ("text_ga->text_en", "text_ga", "text_en"),
)


def sim(a, b):
    a = np.asarray(a, np.float64).copy()
    b = np.asarray(b, np.float64).copy()
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def best_layer_stats(A, B):
    """Scan layers, keep the one with the best top-1 retrieval."""
    best = None
    for layer in range(A.shape[0]):
        matrix = sim(A[layer], B[layer])
        n = matrix.shape[0]
        order = np.argsort(-matrix, axis=1)
        rank = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
        stats = (layer, float((rank == 1).mean()), float((rank <= 5).mean()),
                 float(np.median(rank)))
        if best is None or stats[1] > best[1]:
            best = stats
    return best


def report(path, tag, pooling="weighted"):
    data = np.load(path)
    print(f"\n=== {tag}  [{pooling} pooling] ===")
    print(f"    {path}")
    for name, left, right in CONDITIONS:
        A, B = data[f"{left}.{pooling}"], data[f"{right}.{pooling}"]
        layer, top1, top5, median_rank = best_layer_stats(A, B)
        self_sim = sim(A[layer], A[layer])
        off = self_sim[~np.eye(len(self_sim), dtype=bool)]
        print(f"  {name:22s} best L{layer:2d}  top1={top1:.2f}  top5={top5:.2f}  "
              f"median_rank={median_rank:5.1f}/100")
        print(f"  {'':22s} {left} spread: mean off-diagonal cos = {off.mean():+.3f}")
    print(f"  {'':22s} chance: top1=0.01, top5=0.05, median_rank=50.5")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: mexa_retrieval_diag.py <npz> [<npz> ...]")
    for pooling in ("weighted", "lasttoken"):
        for path in sys.argv[1:]:
            report(path, path.split("/")[-2], pooling)


if __name__ == "__main__":
    main()
