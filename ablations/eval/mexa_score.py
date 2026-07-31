#!/usr/bin/env python3
"""MEXA scoring: parallel-sentence alignment inside a model's hidden states.

MEXA (arXiv 2410.05873, https://github.com/cisnlp/MEXA) is a label-free, generation-free
probe. Embed n parallel sentences twice - once per language, or here also once per MODALITY -
build the n x n cosine similarity matrix at each layer, and count a sentence as aligned only
if its true counterpart is the MUTUAL nearest neighbour: strictly the largest entry in its
row AND in its column. Score = passing / n, per layer.

The criterion and the cosine-in-float64 are ported from the reference compute_mexa.py rather
than reinvented. Two consequences worth stating because they shape the whole experiment:

  * The criterion is symmetric under transpose, so ga->en and en->ga are the SAME number.
    There is one cell per unordered pair, not two. (--self-test proves this.)
  * Requiring the mutual maximum, not merely a high similarity, is what makes the score
    robust to a model that maps everything into one blob: a degenerate representation with
    uniformly high similarities scores 0, not 1.

Six conditions come from the four embeddable objects (Irish/English x text/speech):

    text_ga~text_en     cross-lingual, text only
    speech_ga~text_ga   cross-modal, Irish
    speech_en~text_en   cross-modal, English (control)
    speech_ga~text_en   cross-modal AND cross-lingual
    speech_en~text_ga   cross-modal + cross-lingual, reverse
    speech_ga~speech_en cross-lingual, speech only (no generative counterpart)

Usage:
    python mexa_score.py --self-test
    python mexa_score.py --embeddings data/mexa_embeddings/*.npz --out mexa_long.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

# The four embeddable objects, and the six unordered pairs over them.
OBJECTS = ("text_ga", "text_en", "speech_ga", "speech_en")
POOLINGS = ("weighted", "lasttoken")
CONDITIONS = (
    ("text_ga~text_en", "text_ga", "text_en"),
    ("speech_ga~text_ga", "speech_ga", "text_ga"),
    ("speech_en~text_en", "speech_en", "text_en"),
    ("speech_ga~text_en", "speech_ga", "text_en"),
    ("speech_en~text_ga", "speech_en", "text_ga"),
    ("speech_ga~speech_en", "speech_ga", "speech_en"),
)


def mexa_pass_mask(matrix):
    """Per-sentence boolean: is the diagonal entry the strict max of its row AND its column?

    Kept separate from mexa() so tests (and later error analysis) can ask WHICH sentences
    aligned, not just how many.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError(f"expected a square matrix, got {matrix.shape}")
    diagonal = np.diag(matrix)
    off = matrix.copy()
    np.fill_diagonal(off, -np.inf)          # exclude the diagonal from both maxima
    row_ok = diagonal > off.max(axis=1)     # strictly greater, as in the reference
    col_ok = diagonal > off.max(axis=0)
    return row_ok & col_ok


def mexa(matrix):
    """Fraction of sentences whose diagonal entry is the strict max of its row AND column.

    Port of compute_mexa.py::mexa. The reference loops with np.delete; this is the vectorised
    equivalent and --self-test checks the two agree on random matrices.
    """
    mask = mexa_pass_mask(matrix)
    return float(np.count_nonzero(mask)) / mask.shape[0]


def _mexa_reference(matrix):
    """Literal transcription of the published loop. Used only by --self-test, as the oracle
    the vectorised mexa() above is checked against."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n = len(matrix)
    count = 0
    for i in range(n):
        diag_element = matrix[i][i]
        row = matrix[i]
        column = matrix[:, i]
        if diag_element > max(np.delete(row, i)):
            if diag_element > max(np.delete(column, i)):
                count += 1
    return count / n


def similarity_matrix(a, b):
    """Cosine similarity of every row of `a` against every row of `b`, in float64.

    The reference calls scipy.spatial.distance.cosine pairwise after astype(np.float64); a
    normalised matmul in the same dtype is the same quantity without the n^2 Python loop.
    float64 matters: embeddings are stored fp16, and the criterion is a STRICT comparison
    between numbers that can sit very close together.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def score_layer(a, b):
    """MEXA score for one layer: n x d embeddings on each side -> a number in [0, 1]."""
    return mexa(similarity_matrix(a, b))


# ---------------------------------------------------------------------------
# Self-tests. Nothing downstream is trustworthy until these pass, so they are a
# first-class part of the tool rather than a separate file.
# ---------------------------------------------------------------------------
def self_test():
    rng = np.random.default_rng(0)
    failures = []

    def check(name, condition, detail=""):
        print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        if not condition:
            failures.append(name)

    print("MEXA scoring self-test")

    # 1. Perfect alignment: every diagonal entry is the unique maximum of its row and column.
    identity = np.eye(100)
    check("identity matrix scores exactly 1.0", mexa(identity) == 1.0, f"{mexa(identity)}")

    # 2. One impostor entry. Note it fails TWO sentences, not one: entry [7, 20] beats
    #    sentence 7's diagonal in row 7, and simultaneously beats sentence 20's diagonal in
    #    column 20. That is the mutual-nearest-neighbour criterion doing its job - being a
    #    near-match to someone else's sentence disqualifies you too.
    broken = np.eye(50) * 0.9
    broken[7, 20] = 0.95
    failed = set(np.flatnonzero(~mexa_pass_mask(broken)).tolist())
    check("one impostor entry fails exactly the two sentences it touches",
          failed == {7, 20} and mexa(broken) == 48 / 50, f"failed={sorted(failed)}")

    # 2b. A pure row failure with no column side-effect: give sentence 3 a rival that is not
    #     any other sentence's diagonal competitor, by beating it in its row only.
    row_only = np.eye(50) * 0.9
    row_only[3, 11] = 0.95
    row_only[11, 11] = 0.99          # sentence 11 still wins its own column outright
    failed_row = set(np.flatnonzero(~mexa_pass_mask(row_only)).tolist())
    check("a row-only failure fails that sentence alone",
          failed_row == {3} and mexa(row_only) == 49 / 50, f"failed={sorted(failed_row)}")

    # 3. Chance level: unrelated embeddings must not score. With n=100 a mutual maximum by
    #    accident is rare, so this must land at or very near zero.
    random_scores = [mexa(rng.random((100, 100))) for _ in range(20)]
    check("random 100x100 matrices score ~0",
          max(random_scores) <= 0.05, f"max over 20 draws = {max(random_scores):.3f}")

    # 4. The symmetry claim the whole six-condition design rests on: swapping which language
    #    is the pivot cannot change the score, so ga->en and en->ga are ONE cell.
    transpose_ok = True
    for _ in range(20):
        m = rng.random((60, 60))
        m[np.arange(60), np.arange(60)] += rng.random(60)   # make some diagonals win
        if mexa(m) != mexa(m.T):
            transpose_ok = False
    check("transposing the matrix leaves the score identical (direction collapses)",
          transpose_ok)

    # 5. The vectorised implementation must agree with the published loop, exactly.
    oracle_ok = True
    for _ in range(20):
        m = rng.random((40, 40))
        m[np.arange(40), np.arange(40)] += rng.random(40)
        if mexa(m) != _mexa_reference(m):
            oracle_ok = False
    check("agrees with the published loop implementation on random matrices", oracle_ok)

    # 6. Degenerate representation: everything collapsed onto one point. Similarities are all
    #    equal, no diagonal is STRICTLY greater, so the score is 0 rather than 1. This is the
    #    property that lets MEXA give an honest answer for a broken checkpoint.
    collapsed = np.ones((100, 100))
    check("a fully collapsed representation scores 0, not 1", mexa(collapsed) == 0.0)

    # 7. End to end through similarity_matrix: identical embeddings on both sides.
    embeddings = rng.standard_normal((100, 64))
    check("identical embeddings on both sides score 1.0",
          score_layer(embeddings, embeddings) == 1.0)

    # 8. fp16 storage round-trip must not change the score - embeddings are saved fp16.
    a = rng.standard_normal((100, 64))
    b = a + 0.05 * rng.standard_normal((100, 64))
    check("fp16 round-trip does not change the score",
          score_layer(a, b) == score_layer(a.astype(np.float16), b.astype(np.float16)),
          f"{score_layer(a, b):.3f}")

    print(f"\n{'ALL TESTS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# Scoring saved embedding files.
# ---------------------------------------------------------------------------
def load_embeddings(path):
    """One .npz written by mexa_embed.py -> ({(object, pooling): [layers, n, d]}, metadata)."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(open(os.path.splitext(path)[0] + ".json", encoding="utf-8").read())
    arrays = {}
    for obj in OBJECTS:
        for pooling in POOLINGS:
            key = f"{obj}.{pooling}"
            if key not in data:
                raise SystemExit(f"{path}: missing '{key}' - was extraction interrupted?")
            arrays[(obj, pooling)] = data[key]
    return arrays, meta


def score_file(path):
    """All six conditions x both poolings x every layer for one model state."""
    arrays, meta = load_embeddings(path)
    n_layers, n_sents, hidden = arrays[("text_ga", "weighted")].shape
    rows = []
    for pooling in POOLINGS:
        for condition, left, right in CONDITIONS:
            a, b = arrays[(left, pooling)], arrays[(right, pooling)]
            if a.shape != b.shape:
                raise SystemExit(f"{path}: {left} is {a.shape} but {right} is {b.shape}; the "
                                 f"two sides of a condition must be row-for-row parallel")
            for layer in range(n_layers):
                rows.append({
                    "model": meta["model"],
                    "step": meta["step"],
                    "train_fraction": meta["train_fraction"],
                    "label": meta["label"],
                    "condition": condition,
                    "layer": layer,
                    "pooling": pooling,
                    "mexa": round(score_layer(a[layer], b[layer]), 6),
                    "n_sentences": n_sents,
                })
    print(f"[score] {os.path.basename(path)}: {n_layers} layers x {hidden}d, "
          f"n={n_sents}, {len(rows)} rows", flush=True)
    return rows


FIELDS = ["model", "step", "train_fraction", "label", "condition", "layer", "pooling",
          "mexa", "n_sentences"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="run the scoring unit tests and exit (do this before any GPU time)")
    ap.add_argument("--embeddings", nargs="*", default=[],
                    help=".npz files from mexa_embed.py (globs are expanded)")
    ap.add_argument("--out", default=None, help="long CSV: one row per model x condition "
                                                "x layer x pooling")
    ap.add_argument("--summary", default=None,
                    help="summary CSV: mean- and max-over-layers per model x condition x pooling")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.embeddings:
        ap.error("pass --self-test or --embeddings")

    paths = []
    for pattern in args.embeddings:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    rows = []
    for path in paths:
        rows.extend(score_file(path))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.out}")

    if args.summary:
        groups = {}
        for row in rows:
            key = (row["model"], row["step"], row["train_fraction"], row["label"],
                   row["condition"], row["pooling"])
            groups.setdefault(key, []).append(row["mexa"])
        with open(args.summary, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "step", "train_fraction", "label", "condition", "pooling",
                             "mexa_mean_over_layers", "mexa_max_over_layers", "best_layer",
                             "n_layers"])
            for key, values in sorted(groups.items()):
                writer.writerow(list(key) + [round(float(np.mean(values)), 6),
                                             round(float(np.max(values)), 6),
                                             int(np.argmax(values)), len(values)])
        print(f"wrote {len(groups)} rows to {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
