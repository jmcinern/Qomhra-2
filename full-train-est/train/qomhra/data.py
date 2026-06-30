"""Datasets for the training-time estimate.

`SyntheticTokenDataset` is the one used for the quick first number: it streams
fixed-length windows of uniformly-random token ids. There is no file I/O, so the
measured step time reflects the pure compute + RCCL communication path (the real
data loader can only be slower, never faster — this is the optimistic bound, which
is exactly what we want for a first GPU-hour estimate).

Each rank seeds its generator differently so the 8 GCDs don't train on identical
batches (mirrors data-parallel behaviour). Causal-LM labels = input_ids; the HF
model shifts them internally, so we do not shift here.

`MemmapTokenDataset` (stub, below) is the drop-in for when Agent2/Agent3 tokens
land — same yielded dict, so nothing in the train loop changes.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset


class SyntheticTokenDataset(IterableDataset):
    def __init__(self, vocab_size, seq_len, seed=0):
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        # Differentiate by global rank so ranks see different data.
        rank = int(os.environ.get("RANK", "0"))
        self.seed = int(seed) + 1000 * rank

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            ids = torch.randint(
                0, self.vocab_size, (self.seq_len,), generator=g, dtype=torch.long
            )
            yield {"input_ids": ids, "labels": ids.clone()}


class PackedTokenDataset(IterableDataset):
    """Yields fixed-length windows from a pre-packed 1-D token tensor.

    The whole (subset) corpus is small enough to hold in memory, so we build one
    flat `tokens` tensor once and chop it into `seq_len` windows. Each rank takes a
    strided shard of the windows (windows[rank::world_size]) and cycles forever, so
    timing runs of any `total_steps` work without an epoch boundary. Same yielded
    dict as the synthetic dataset, so the train loop is unchanged.
    """

    def __init__(self, tokens, seq_len, seed=0):
        self.tokens = tokens
        self.seq_len = int(seq_len)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.seed = int(seed)
        n_windows = self.tokens.numel() // self.seq_len
        # This rank's window indices (strided so ranks see disjoint data).
        self.idx = list(range(self.rank, n_windows, self.world))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + 1000 * self.rank)
        order = list(self.idx)
        while True:
            perm = torch.randperm(len(order), generator=g).tolist()
            for j in perm:
                start = order[j] * self.seq_len
                ids = self.tokens[start : start + self.seq_len].to(torch.long)
                yield {"input_ids": ids, "labels": ids.clone()}


def build_real_corpus(args):
    """Load the already-tokenized text subset + an audio subset (~1 h) from LUMI,
    offset audio into the combined vocab, shuffle docs together and concatenate
    into one flat token tensor. No tokenization happens here — these files were
    produced by Agent2 (text) and Agent3 (audio); we only read and pack them.
    """
    d = args.data
    sep_id = int(d.sep_id)
    rng = np.random.default_rng(int(args.seed))

    docs = []  # list of 1-D int64 numpy arrays (one per document/clip)

    # --- text: uint32 ids, documents separated by sep_id ---
    text = np.fromfile(d.text_bin, dtype=np.uint32).astype(np.int64)
    if d.text_token_budget:
        text = text[: int(d.text_token_budget)]
    # split on sep_id into docs (drop empty runs)
    bounds = np.flatnonzero(text == sep_id)
    prev = 0
    for b in bounds:
        if b > prev:
            docs.append(text[prev:b])
        prev = b + 1
    if prev < len(text):
        docs.append(text[prev:])
    n_text_docs, n_text_tok = len(docs), int(sum(len(x) for x in docs))

    # --- audio: uint16 units, one .npy per clip, offset into combined vocab ---
    audio_offset = int(d.audio_offset)
    budget = int(d.audio_token_budget)
    files = sorted(glob.glob(os.path.join(d.audio_units_dir, "**", "*.npy"),
                             recursive=True))
    n_audio_tok, n_audio_docs = 0, 0
    for f in files:
        if n_audio_tok >= budget:
            break
        units = np.load(f).astype(np.int64) + audio_offset
        docs.append(units)
        n_audio_tok += len(units)
        n_audio_docs += 1

    # --- shuffle docs together, concat with sep between, flatten ---
    order = rng.permutation(len(docs))
    parts = []
    for i in order:
        parts.append(docs[i])
        parts.append(np.array([sep_id], dtype=np.int64))
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[data] real corpus: text {n_text_tok:,} tok / {n_text_docs:,} docs"
              f" + audio {n_audio_tok:,} tok / {n_audio_docs} clips"
              f" (offset +{audio_offset}) -> {len(flat):,} packed tokens,"
              f" {len(flat) // int(d.seq_len):,} windows of {int(d.seq_len)}",
              flush=True)
    return torch.from_numpy(flat)


def get_dataloader(args, vocab_size):
    from torch.utils.data import DataLoader

    if args.data.source == "real":
        tokens = build_real_corpus(args)
        dataset = PackedTokenDataset(
            tokens, seq_len=args.data.seq_len, seed=args.seed
        )
    else:
        dataset = SyntheticTokenDataset(
            vocab_size=vocab_size, seq_len=args.data.seq_len, seed=args.seed
        )
    # num_workers=0: data is in-memory (real) or trivial to generate (synthetic),
    # so worker processes only add noise to the timing. pin_memory pairs with the
    # Accelerator's non_blocking H2D copies.
    return DataLoader(
        dataset,
        batch_size=args.data.micro_batch_size,
        num_workers=0,
        pin_memory=True,
    )
