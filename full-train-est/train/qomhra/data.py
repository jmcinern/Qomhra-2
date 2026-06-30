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


def get_dataloader(args, vocab_size):
    from torch.utils.data import DataLoader

    dataset = SyntheticTokenDataset(
        vocab_size=vocab_size, seq_len=args.data.seq_len, seed=args.seed
    )
    # num_workers=0: generation is trivial and keeps the synthetic path free of
    # dataloader-process noise in the timing. pin_memory pairs with the
    # Accelerator's non_blocking H2D copies.
    return DataLoader(
        dataset,
        batch_size=args.data.micro_batch_size,
        num_workers=0,
        pin_memory=True,
    )


# --- drop-in for real tokens (Agent2 text + Agent3 audio) -------------------
# class MemmapTokenDataset(IterableDataset):
#     """np.memmap Agent2 train.bin (uint32 text ids) + concatenated Agent3 .npy
#     audio units (offset by text_vocab), packed into seq_len windows. Same yielded
#     {input_ids, labels} dict as above so get_dataloader swaps with one line."""
#     ...
