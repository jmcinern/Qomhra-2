"""Datasets for the ablation CPT runs.

Three data sources, one per ablation:

  real   — text: pre-tokenized uint32 `.bin` (Qwen2.5-Omni tokenizer), docs split
           on sep_id, packed into fixed-length windows. Text + ASR transcripts.
  audio  — speech: WAVs -> WhisperFeatureExtractor mel, fed to Omni's native audio
           tower at train time (see model.py). Mel extraction happens in dataloader
           workers; we never cache encoder outputs, because the encoder is trained.
  mixed  — both: pure-text batches and pure-audio batches interleaved on a seeded
           schedule. Crucially the schedule is rank-INDEPENDENT: every rank runs the
           same modality on a given global step. FSDP requires an identical collective
           sequence across ranks, so a rank doing a text forward while another does an
           audio forward deadlocks on the first all-gather.

`SyntheticTokenDataset` is retained for pure timing runs (no file I/O).
"""
import csv
import glob
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset

SAMPLE_RATE = 16000


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

    The (subset) corpus is small enough to hold in memory, so we build one flat
    `tokens` tensor once and chop it into `seq_len` windows. Each rank takes a
    strided shard of the windows (windows[rank::world_size]) and cycles forever, so
    runs of any `total_steps` work without an epoch boundary.
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


def build_text_corpus(args):
    """Load the Omni-tokenized text/ASR `.bin`(s), shuffle docs, pack into one tensor.

    `text_bin` is a path or list of paths, each of which may be a **glob** — the
    tokenizer emits one shard per parquet row-group (`<source>__rgNN.bin`), so a
    source is `<source>__rg*.bin`, not a single file. No tokenization happens here;
    these were produced by ablations/text/tokenize_text.py with the Omni tokenizer.
    """
    d = args.data
    sep_id = int(d.sep_id)
    rng = np.random.default_rng(int(args.seed))

    patterns = d.text_bin
    if isinstance(patterns, str):
        patterns = [patterns]
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise FileNotFoundError(f"text_bin pattern matched no files: {pat}")
        paths.extend(hits)

    docs = []
    for path in paths:
        text = np.fromfile(path, dtype=np.uint32).astype(np.int64)
        bounds = np.flatnonzero(text == sep_id)
        prev = 0
        for b in bounds:
            if b > prev:
                docs.append(text[prev:b])
            prev = b + 1
        if prev < len(text):
            docs.append(text[prev:])

    order = rng.permutation(len(docs))
    parts = []
    n_tok = 0
    budget = int(d.text_token_budget) if d.get("text_token_budget", None) else 0
    for i in order:
        parts.append(docs[i])
        parts.append(np.array([sep_id], dtype=np.int64))
        n_tok += len(docs[i]) + 1
        if budget and n_tok >= budget:
            break
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[data] text corpus: {len(flat):,} tokens / {len(parts)//2:,} docs"
              f" from {len(paths)} shard(s) ->"
              f" {len(flat) // int(d.seq_len):,} windows of {int(d.seq_len)}",
              flush=True)
    return torch.from_numpy(flat)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def load_wav(path):
    """Return float32 mono @16 kHz in [-1,1]. scipy, because the container has no
    soundfile; the corpus is 16 kHz mono int16 PCM (see speech/speech-tknz.py)."""
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        wav = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        wav = (data.astype(np.float32) - 128.0) / 128.0
    else:
        wav = data.astype(np.float32)
    if sr != SAMPLE_RATE:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, SAMPLE_RATE
        ).numpy()
    return wav


class AudioClipDataset(IterableDataset):
    """Streams fixed-duration audio windows as mel features.

    Long recordings are cut into `chunk_s`-second windows (the audio tower is a
    quadratic-attention transformer, and a 30 s window is ~750 frames at 25 Hz —
    one window fills a comparable context to a text sequence). Each rank takes a
    strided shard of the file list and cycles forever, mirroring PackedTokenDataset.
    Mel extraction runs here, in the dataloader worker, so it overlaps with the GPU
    step; the encoder forward stays in the graph on the GPU because it is trained.
    """

    def __init__(self, manifest, audio_root, model_dir, chunk_s=30, seed=0,
                 min_chunk_s=2.0):
        self.rows = self._read_manifest(manifest, audio_root)
        if not self.rows:
            raise RuntimeError(f"no audio files resolved from manifest {manifest}")
        self.model_dir = model_dir
        self.chunk_s = int(chunk_s)
        self.min_samples = int(min_chunk_s * SAMPLE_RATE)
        self.seed = int(seed)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.idx = list(range(self.rank, len(self.rows), self.world))
        self._fe = None

        if self.rank == 0:
            print(f"[data] audio corpus: {len(self.rows):,} files "
                  f"({len(self.idx):,} on this rank), {self.chunk_s}s windows",
                  flush=True)

    @staticmethod
    def _read_manifest(manifest, audio_root):
        rows = []
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                p = os.path.join(audio_root, r["dest_label"], r["filename"])
                if os.path.isfile(p):
                    rows.append(p)
        return rows

    def _feature_extractor(self):
        # Built lazily so it lands inside the worker process, not the parent.
        if self._fe is None:
            from transformers import AutoFeatureExtractor
            fe = AutoFeatureExtractor.from_pretrained(self.model_dir)
            self._fe = getattr(fe, "feature_extractor", fe)
        return self._fe

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)
        order = self.idx[wid::nworkers]
        if not order:
            return

        fe = self._feature_extractor()
        rng = np.random.default_rng(self.seed + 1000 * self.rank + wid)
        n = self.chunk_s * SAMPLE_RATE

        while True:
            for j in rng.permutation(len(order)):
                try:
                    wav = load_wav(self.rows[order[j]])
                except Exception:
                    continue  # unreadable file: skip rather than kill the run
                for start in range(0, len(wav), n):
                    chunk = wav[start : start + n]
                    if len(chunk) < self.min_samples:
                        continue  # too short to have a next frame to predict
                    feats = fe(chunk, sampling_rate=SAMPLE_RATE,
                               return_attention_mask=True, return_tensors="pt")
                    mel = feats["input_features"][0]                 # (128, T)
                    flen = int(feats["feature_attention_mask"][0].sum())
                    yield {"input_features": mel[:, :flen], "feature_lens": flen}


def collate_audio(items):
    """Pad variable-length mel to the batch max; keep true lengths for the encoder.

    The audio tower packs the batch by `feature_lens` internally (model.py), so the
    padding here is never seen by the encoder — it only makes the batch stackable.
    """
    lens = torch.tensor([it["feature_lens"] for it in items], dtype=torch.long)
    n_mel = items[0]["input_features"].shape[0]
    out = torch.zeros(len(items), n_mel, int(lens.max()), dtype=torch.float32)
    for i, it in enumerate(items):
        f = it["input_features"]
        out[i, :, : f.shape[1]] = f
    return {"input_features": out, "feature_lens": lens}


class ModalityScheduler:
    """Which modality every rank runs on each global step.

    Seeded from `args.seed` alone — NOT the rank — so all ranks agree step-for-step.
    See the module docstring for why divergence deadlocks FSDP.
    """

    def __init__(self, total_steps, text_step_frac, seed=0):
        rng = np.random.default_rng(int(seed))
        n = int(total_steps)
        n_text = int(round(n * float(text_step_frac)))
        sched = np.array(["audio"] * n, dtype=object)
        sched[rng.choice(n, size=n_text, replace=False)] = "text"
        self.sched = sched

    def modality(self, step):
        return self.sched[step % len(self.sched)]


def get_dataloader(args, vocab_size):
    """Returns {modality: DataLoader} — one entry for text/audio, both for mixed."""
    from torch.utils.data import DataLoader

    d = args.data
    source = d.source
    loaders = {}

    if source in ("real", "mixed"):
        tokens = build_text_corpus(args)
        loaders["text"] = DataLoader(
            PackedTokenDataset(tokens, seq_len=d.seq_len, seed=args.seed),
            batch_size=d.micro_batch_size,
            num_workers=0,          # corpus is already in memory
            pin_memory=True,
        )

    if source in ("audio", "mixed"):
        loaders["audio"] = DataLoader(
            AudioClipDataset(
                manifest=d.audio_manifest,
                audio_root=d.audio_root,
                model_dir=args.model.base_model_id,
                chunk_s=d.audio_chunk_s,
                seed=args.seed,
            ),
            batch_size=d.audio_micro_batch_size,
            num_workers=int(d.get("audio_workers", 4)),   # mel extraction is CPU-bound
            pin_memory=True,
            collate_fn=collate_audio,
        )

    if source == "synthetic":
        loaders["text"] = DataLoader(
            SyntheticTokenDataset(vocab_size, seq_len=d.seq_len, seed=args.seed),
            batch_size=d.micro_batch_size,
            num_workers=0,
            pin_memory=True,
        )

    if not loaders:
        raise ValueError(f"unknown data.source: {source!r}")
    return loaders
