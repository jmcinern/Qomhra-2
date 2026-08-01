#!/usr/bin/env python3
"""Why is the spread among unit-sequence embeddings so small?

Two candidates, separated here because they call for opposite responses.

(a) The 1000 unit token embeddings are themselves nearly identical, so a sequence of them
    cannot carry much. That would be a property of the trained model and no amount of
    post-processing fixes it.

(b) Ordinary anisotropy: decoder hidden states occupy a narrow cone, every sequence is
    dominated by one shared direction, and the sentence-specific part is a small residual
    on top. That is a property of the MEASUREMENT, and subtracting the mean recovers it.

Test for (a): pairwise cosine among the unit rows of the input embedding matrix, against a
same-sized sample of ordinary text rows as the control.

Test for (b): re-run retrieval and MEXA after centering each object's 100 embeddings. If the
signal jumps, the information was present and the common direction was hiding it.

    python mexa_collapse_diag.py <npz> [--checkpoint <path>]
"""
import argparse

import numpy as np

BASE_VOCAB = 151936
UNIT_COUNT = 1000


def sim(a, b):
    a = np.asarray(a, np.float64).copy()
    b = np.asarray(b, np.float64).copy()
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def offdiag_mean(x):
    s = sim(x, x)
    return float(s[~np.eye(len(s), dtype=bool)].mean())


def embedding_spread(checkpoint):
    """(a) Are the unit embeddings themselves collapsed, relative to text embeddings?"""
    import sys, os, torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
    from qomhra.checkpoint import load_for_eval

    model, _ = load_for_eval(checkpoint, device="cpu", dtype=torch.float32)
    weight = model.thinker.get_input_embeddings().weight.detach().float().numpy()
    units = weight[BASE_VOCAB:BASE_VOCAB + UNIT_COUNT]
    rng = np.random.default_rng(0)
    text = weight[rng.choice(BASE_VOCAB, UNIT_COUNT, replace=False)]

    print("\n(a) input embedding matrix")
    print(f"  {UNIT_COUNT} unit rows      : mean pairwise cos = {offdiag_mean(units):+.4f}   "
          f"mean norm = {np.linalg.norm(units, axis=1).mean():.4f}")
    print(f"  {UNIT_COUNT} text rows      : mean pairwise cos = {offdiag_mean(text):+.4f}   "
          f"mean norm = {np.linalg.norm(text, axis=1).mean():.4f}")
    print("  (if the unit number is far higher, the tokens themselves are the bottleneck)")
    del model


def centering(path, pooling="weighted"):
    """(b) Does subtracting the per-object mean recover retrieval and MEXA?"""
    data = np.load(path)
    print(f"\n(b) centering, {pooling} pooling, speech_ga vs text_ga")
    print(f"{'layer':>5} {'spread raw':>11} {'spread cent':>12} "
          f"{'top1 raw':>9} {'top1 cent':>10} {'MEXA raw':>9} {'MEXA cent':>10}")

    A, B = data[f"speech_ga.{pooling}"], data[f"text_ga.{pooling}"]
    best = None
    for layer in range(A.shape[0]):
        a, b = A[layer].astype(np.float64), B[layer].astype(np.float64)
        ac, bc = a - a.mean(0, keepdims=True), b - b.mean(0, keepdims=True)
        row = []
        for x, y in ((a, b), (ac, bc)):
            m = sim(x, y)
            n = m.shape[0]
            order = np.argsort(-m, axis=1)
            rank = np.array([int(np.where(order[i] == i)[0][0]) for i in range(n)])
            d = np.diag(m)
            off = m.copy(); np.fill_diagonal(off, -np.inf)
            mexa = float(((d > off.max(1)) & (d > off.max(0))).mean())
            row.append((float((rank == 0).mean()), mexa))
        entry = (layer, offdiag_mean(a), offdiag_mean(ac),
                 row[0][0], row[1][0], row[0][1], row[1][1])
        if best is None or entry[4] > best[4]:
            best = entry
        if layer % 6 == 0 or layer == A.shape[0] - 1:
            print(f"{entry[0]:>5} {entry[1]:>+11.3f} {entry[2]:>+12.3f} "
                  f"{entry[3]:>9.2f} {entry[4]:>10.2f} {entry[5]:>9.2f} {entry[6]:>10.2f}")
    print(f"\n  best centered layer L{best[0]}: spread {best[1]:+.3f} -> {best[2]:+.3f}, "
          f"top1 {best[3]:.2f} -> {best[4]:.2f}, MEXA {best[5]:.2f} -> {best[6]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pooling", default="weighted")
    args = ap.parse_args()
    centering(args.npz, args.pooling)
    if args.checkpoint:
        embedding_spread(args.checkpoint)


if __name__ == "__main__":
    main()
