#!/usr/bin/env python3
"""100-way held-out transcript retrieval for discrete speech units.

For each held-out speech prefix, score every candidate transcript by mean
teacher-forced conditional NLL.  Retrieval@1 asks whether the true transcript
has the lowest NLL.  There is no generation, instruction template, truncation or
post-processing, so chance accuracy with 100 candidates is exactly 1%.

One invocation scores both the arm's initial state and its trained checkpoint.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, "..", "train"))
sys.path.insert(0, TRAIN)
from qomhra.model import get_model  # noqa: E402


def heldout_examples(path, n):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)

    # Select deterministically using only cheap scalar metadata. Converting the
    # two 1024-wide list columns for every row to Python ints consumed ~14 GB.
    validation_filenames = []
    for batch in parquet.iter_batches(
        columns=["split", "filename"], batch_size=65_536
    ):
        splits, filenames = batch.column(0), batch.column(1)
        for row_index in range(batch.num_rows):
            if splits[row_index].as_py() == "validation":
                validation_filenames.append(filenames[row_index].as_py())

    heldout_total = len(validation_filenames)
    if heldout_total < n:
        raise RuntimeError(f"only {heldout_total} validation rows, requested {n}")
    wanted = set(sorted(validation_filenames)[:n])

    # Scan bounded Arrow batches and convert token lists only for wanted rows.
    rows = []
    for batch in parquet.iter_batches(
        columns=["input_ids", "labels", "split", "filename"], batch_size=512
    ):
        ids_col, labels_col, split_col, filename_col = (
            batch.column(index) for index in range(4)
        )
        for row_index in range(batch.num_rows):
            if split_col[row_index].as_py() != "validation":
                continue
            filename = filename_col[row_index].as_py()
            if filename not in wanted:
                continue
            labels = np.asarray(labels_col[row_index].as_py(), dtype=np.int64)
            target_positions = np.flatnonzero(labels != -100)
            if not len(target_positions):
                raise RuntimeError(
                    f"held-out row {filename} has no transcript labels"
                )
            start = int(target_positions[0])
            ids = list(map(int, ids_col[row_index].as_py()))
            # Prefix ends at <|transcript_start|>; candidate includes transcript and EOS.
            rows.append(
                {
                    "filename": filename,
                    "prefix": ids[:start],
                    "transcript": ids[start:],
                }
            )
    rows.sort(key=lambda row: row["filename"])
    if len(rows) != n:
        raise RuntimeError(f"selected {n} validation rows but loaded {len(rows)}")
    return rows, heldout_total


def candidate_nll(model, prefix, candidates, batch_size, device):
    scores = []
    thinker = model.thinker
    for first in range(0, len(candidates), batch_size):
        batch_candidates = candidates[first:first + batch_size]
        sequences = [prefix + candidate for candidate in batch_candidates]
        labels = [
            [-100] * len(prefix) + candidate for candidate in batch_candidates
        ]
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        input_ids = torch.zeros(
            len(sequences), max_len, dtype=torch.long, device=device
        )
        targets = torch.full(
            (len(sequences), max_len), -100, dtype=torch.long, device=device
        )
        attention = torch.zeros(
            len(sequences), max_len, dtype=torch.long, device=device
        )
        for index, (seq, target) in enumerate(zip(sequences, labels)):
            length = len(seq)
            input_ids[index, :length] = torch.tensor(seq, device=device)
            targets[index, :length] = torch.tensor(target, device=device)
            attention[index, :length] = 1
        with torch.inference_mode():
            hidden = thinker.model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).last_hidden_state
            logits = thinker.lm_head(hidden[:, :-1])
            shifted_targets = targets[:, 1:]
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                shifted_targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(shifted_targets.shape)
            valid = shifted_targets.ne(-100)
            mean_loss = token_loss.sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        scores.extend(mean_loss.cpu().tolist())
        del input_ids, targets, attention, hidden, logits, token_loss
    return scores


def retrieval(model, examples, batch_size, device, label):
    candidates = [row["transcript"] for row in examples]
    matrix = []
    ranks = []
    margins = []
    for index, row in enumerate(examples):
        scores = candidate_nll(
            model, row["prefix"], candidates, batch_size, device
        )
        matrix.append(scores)
        order = np.argsort(scores)
        rank = int(np.flatnonzero(order == index)[0]) + 1
        ranks.append(rank)
        wrong_best = min(score for j, score in enumerate(scores) if j != index)
        margins.append(wrong_best - scores[index])
        if (index + 1) % 5 == 0 or index + 1 == len(examples):
            hits = sum(value == 1 for value in ranks)
            print(
                f"[{label}] {index + 1}/{len(examples)} "
                f"running R@1={hits / len(ranks):.1%}",
                flush=True,
            )
    return {
        "retrieval_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "retrieval_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])),
        "median_rank": float(np.median(ranks)),
        "mean_true_vs_best_wrong_margin_nll": float(np.mean(margins)),
        "ranks": ranks,
        "score_matrix": matrix,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(args.checkpoint, "config.yaml"))
    torch.manual_seed(int(cfg.seed))
    model, _ = get_model(cfg)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    examples, heldout_total = heldout_examples(args.data, args.n)

    initial = retrieval(model, examples, args.batch_size, "cuda", "initial")

    state = torch.load(
        os.path.join(args.checkpoint, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"trained checkpoint mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    del state
    torch.cuda.empty_cache()
    trained = retrieval(model, examples, args.batch_size, "cuda", "trained")

    report = {
        "label": args.label,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "heldout_total": heldout_total,
        "retrieval_n": args.n,
        "chance_retrieval_at_1": 1.0 / args.n,
        "selection": "first N validation rows by filename; same matrix before/after",
        "score": "mean transcript-token conditional NLL; lower is better",
        "initial": initial,
        "trained": trained,
        "delta_retrieval_at_1": (
            trained["retrieval_at_1"] - initial["retrieval_at_1"]
        ),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(
        f"{args.label}: R@1 {initial['retrieval_at_1']:.1%} -> "
        f"{trained['retrieval_at_1']:.1%} "
        f"(chance {1 / args.n:.1%}); wrote {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
