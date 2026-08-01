#!/usr/bin/env python3
"""Re-analyse a saved discrete-ASR retrieval score matrix. No GPU, no model.

`discrete_asr_retrieval.py` already stores the full N x N matrix of mean
transcript-token NLLs, so every statistic below is free to recompute.

Why re-analyse at all: ranking raw NLL within a row conflates two things --
how well the speech prefix predicts a transcript, and how likely that
transcript is a priori.  Candidate-intrinsic difficulty varies by whole nats
per token across transcripts and swamps the speech signal.  Subtracting each
candidate's column mean (its average NLL over all 100 prefixes) removes that
nuisance term and leaves the pointwise mutual information between prefix and
transcript, which is the quantity the experiment is actually asking about.

Also reports mean rank / AUC rather than only R@1.  With N=100 queries, R@1 has
almost no power: a genuine shift to median rank 40 still predicts only ~2-3%
R@1, which is indistinguishable from the 1% chance level at this sample size.
"""

import argparse
import json

import numpy as np


def rank_metrics(matrix):
    """Rank of the true candidate in each row; lower NLL ranks first."""
    n = matrix.shape[0]
    order = np.argsort(matrix, axis=1, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        ranks[i] = int(np.flatnonzero(order[i] == i)[0]) + 1
    # AUC = fraction of wrong candidates the true one beats.
    auc = float(np.mean((n - ranks) / (n - 1)))
    mean_rank = float(ranks.mean())
    # Under H0 ranks are uniform on 1..n.
    null_mean = (n + 1) / 2.0
    null_sd = np.sqrt((n * n - 1) / 12.0)
    z = (null_mean - mean_rank) / (null_sd / np.sqrt(n))
    return {
        "retrieval_at_1": float(np.mean(ranks == 1)),
        "retrieval_at_5": float(np.mean(ranks <= 5)),
        "retrieval_at_10": float(np.mean(ranks <= 10)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": mean_rank,
        "mean_rank_null": null_mean,
        "auc": auc,
        "mean_rank_z_vs_chance": float(z),
        "ranks": ranks.tolist(),
    }


def pmi_metrics(matrix):
    """Paired true-vs-mismatched NLL gap, in nats per transcript token.

    This is the headline effect size: how many nats of transcript-token loss the
    correct speech prefix saves relative to the average wrong prefix.  It is
    well defined even when retrieval is at chance.
    """
    n = matrix.shape[0]
    true_nll = np.diagonal(matrix).copy()
    off = matrix.copy()
    np.fill_diagonal(off, np.nan)
    # Mean over wrong PREFIXES for the same transcript (down the column): the
    # transcript is held fixed, so its intrinsic difficulty cancels exactly.
    mismatched_nll = np.nanmean(off, axis=0)
    gap = mismatched_nll - true_nll
    sd = float(gap.std(ddof=1))
    se = sd / np.sqrt(n)
    return {
        "mean_true_nll": float(true_nll.mean()),
        "mean_mismatched_nll": float(mismatched_nll.mean()),
        "mean_gap_nats_per_token": float(gap.mean()),
        "gap_sd": sd,
        "gap_se": se,
        "gap_t": float(gap.mean() / se) if se else float("nan"),
        "gap_positive_fraction": float(np.mean(gap > 0)),
    }


def bootstrap_auc(matrix, draws, seed):
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    base = np.asarray(rank_metrics(matrix)["ranks"])
    samples = np.empty(draws)
    for k in range(draws):
        pick = rng.integers(0, n, n)
        samples[k] = np.mean((n - base[pick]) / (n - 1))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def analyse(block, draws, seed):
    matrix = np.asarray(block["score_matrix"], dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(f"score matrix is {matrix.shape}, expected square")
    # Column-centred: remove each candidate transcript's intrinsic difficulty.
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    # Column-standardised: also remove differences in how much each candidate's
    # NLL varies across prefixes (length and token-count effects).
    scale = matrix.std(axis=0, keepdims=True)
    standardised = centred / np.where(scale > 0, scale, 1.0)
    return {
        "raw": rank_metrics(matrix),
        "column_centred": rank_metrics(centred),
        "column_standardised": rank_metrics(standardised),
        "pmi": pmi_metrics(matrix),
        "auc_ci95_raw": bootstrap_auc(matrix, draws, seed),
        "auc_ci95_column_centred": bootstrap_auc(centred, draws, seed),
    }


def line(name, block):
    rank = block["column_centred"]
    pmi = block["pmi"]
    return (
        f"{name:<10} raw R@1 {block['raw']['retrieval_at_1']:6.1%} "
        f"med {block['raw']['median_rank']:5.1f} | "
        f"centred R@1 {rank['retrieval_at_1']:6.1%} "
        f"med {rank['median_rank']:5.1f} "
        f"AUC {rank['auc']:.3f} z {rank['mean_rank_z_vs_chance']:5.2f} | "
        f"gap {pmi['mean_gap_nats_per_token']:+.4f} nats/tok "
        f"t {pmi['gap_t']:5.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="asr_retrieval_*.json written by discrete_asr_retrieval.py")
    ap.add_argument("--out")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)

    result = {"label": report.get("label"), "source": args.report}
    for arm in ("initial", "trained"):
        if arm in report and "score_matrix" in report[arm]:
            result[arm] = analyse(report[arm], args.bootstrap, args.seed)
            print(line(arm, result[arm]), flush=True)

    if "initial" in result and "trained" in result:
        result["delta"] = {
            key: (
                result["trained"]["column_centred"][key]
                - result["initial"]["column_centred"][key]
            )
            for key in ("retrieval_at_1", "auc", "mean_rank", "mean_reciprocal_rank")
        }
        result["delta"]["pmi_gap_nats_per_token"] = (
            result["trained"]["pmi"]["mean_gap_nats_per_token"]
            - result["initial"]["pmi"]["mean_gap_nats_per_token"]
        )
        print(json.dumps(result["delta"], indent=2), flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
