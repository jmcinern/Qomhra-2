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
# Imported at module scope, NOT lazily inside load_wav(). The python env is a
# squashfs on Lustre: a first-call import from inside a dataloader worker sends
# every worker on every rank into importlib's path crawl at once, and the metadata
# storm wedges the run before it can produce a single batch.
from scipy.io import wavfile
import torchaudio

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
        # torchaudio is imported at module scope for the same reason as scipy — and
        # more so here: this branch fires only on the ranks that happen to draw an
        # off-rate file, so a lazy import would stall those ranks alone and diverge
        # the collective schedule.
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
                    # The bare WhisperFeatureExtractor calls this `attention_mask`;
                    # only the Omni processor renames it `feature_attention_mask`.
                    # It marks the true mel frames inside Whisper's 30 s zero-padding.
                    mask = feats.get("feature_attention_mask")
                    if mask is None:
                        mask = feats["attention_mask"]
                    flen = int(mask[0].sum())
                    yield {"input_features": mel[:, :flen], "feature_lens": flen}


class AlignedClipDataset(IterableDataset):
    """(speech, transcript) pairs for ablation 4 — the ONLY aligned objective.

    Ablations 2/3 train audio by next-frame regression, which never connects speech to
    text: measured, the encoder hears Irish accurately and renders it through an English
    lexicon ("Rudaí deasa" -> "ready does the"). This dataset supplies the supervision
    that link needs — the audio and the words actually spoken in it.

    Reads aligned_setup.jsonl (fields: audio, duration_s, text). Unlike AudioClipDataset
    there is NO windowing: a transcript labels the whole utterance, so a clip cut in half
    would be paired with words that are not in it. Over-long clips are dropped instead
    (only 4 of 8,010 exceed 30 s).
    """

    @staticmethod
    def is_heldout(audio_path, mod=40):
        """Eval split, by md5 of the audio path — NOT row order, so it stays stable as
        the jsonl grows and training and eval agree without passing a list around.
        Mirrors eval/asr_baseline.py::is_heldout."""
        import hashlib
        return int(hashlib.md5(audio_path.encode()).hexdigest(), 16) % mod == 0

    def __init__(self, jsonl, audio_root, model_dir, max_s=30.0, seed=0,
                 exclude_heldout=True):
        import json
        self.rows = []
        n_held = 0
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["duration_s"] > max_s:
                    continue
                if exclude_heldout and self.is_heldout(r["audio"]):
                    n_held += 1
                    continue
                self.rows.append((os.path.join(audio_root, r["audio"]), r["text"]))
        if not self.rows:
            raise RuntimeError(f"no aligned rows from {jsonl}")
        if exclude_heldout and n_held == 0:
            raise RuntimeError("exclude_heldout=True but nothing was held out — the "
                               "eval split would be training data and every WER "
                               "would be memorisation")
        self.model_dir = model_dir
        self.seed = int(seed)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.idx = list(range(self.rank, len(self.rows), self.world))
        self._fe = None
        self._tok = None
        if self.rank == 0:
            print(f"[data] aligned corpus: {len(self.rows):,} utterances "
                  f"({len(self.idx):,} on this rank), {n_held:,} held out for eval",
                  flush=True)

    @staticmethod
    def _local_snapshot(model_dir):
        """Resolve a repo id to its on-disk snapshot.

        AutoTokenizer.from_pretrained("Qwen/...") runs an `is_base_mistral` check that
        calls the HF API, which the compute nodes cannot reach -> OfflineModeIsEnabled.
        A local path short-circuits it as `_is_local`. AutoFeatureExtractor does no such
        check, which is why the unlabelled-audio path never hit this.
        """
        if os.path.isdir(model_dir):
            return model_dir
        hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
        pat = os.path.join(hub, "models--" + model_dir.replace("/", "--"),
                           "snapshots", "*")
        snaps = sorted(glob.glob(pat))
        if not snaps:
            raise RuntimeError(f"no local snapshot for {model_dir!r} under {hub}; "
                               f"the tokenizer cannot be fetched on an offline node")
        return snaps[0]

    def _extractors(self):
        # Lazily, so they land in the worker process rather than the parent.
        if self._fe is None:
            from transformers import AutoFeatureExtractor, AutoTokenizer
            local = self._local_snapshot(self.model_dir)
            fe = AutoFeatureExtractor.from_pretrained(local)
            self._fe = getattr(fe, "feature_extractor", fe)
            self._tok = AutoTokenizer.from_pretrained(local)
            # <|im_end|>, not tok.eos_token_id: the latter is right on this checkpoint
            # but generation_config's is None, and the eval stops on <|im_end|>. Train
            # the terminator the eval actually looks for.
            self._eos_id = self._tok.convert_tokens_to_ids("<|im_end|>")
        return self._fe, self._tok

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)
        order = self.idx[wid::nworkers]
        if not order:
            return
        fe, tok = self._extractors()
        rng = np.random.default_rng(self.seed + 1000 * self.rank + wid)
        while True:
            for j in rng.permutation(len(order)):
                path, text = self.rows[order[j]]
                try:
                    wav = load_wav(path)
                except Exception:
                    continue  # unreadable file: skip rather than kill the run
                feats = fe(wav, sampling_rate=SAMPLE_RATE,
                           return_attention_mask=True, return_tensors="pt")
                mask = feats.get("feature_attention_mask")
                if mask is None:
                    mask = feats["attention_mask"]
                flen = int(mask[0].sum())
                # Append <|im_end|>: without a terminator the model transcribes the
                # utterance correctly and then never stops — measured, it produced
                # "Go raibh an bhíonn ag dul isteach ar bhealach. Go raibh. Go raibh.
                # Go raibh..." to the token budget, and every repeat scores as an
                # insertion (430% WER on an utterance it had largely got right). The
                # transcript is the whole target, so its end is a fact worth teaching.
                ids = torch.cat([tok(text, return_tensors="pt").input_ids[0],
                                 torch.tensor([self._eos_id])])
                yield {"input_features": feats["input_features"][0][:, :flen],
                       "feature_lens": flen,
                       "input_ids": ids}


def collate_aligned(items):
    """Pad mel and transcript ids independently; -100 marks non-target positions.

    The audio side is padded exactly as collate_audio does. The text side is padded with
    -100 rather than a pad id so cross_entropy ignores it — otherwise the model would be
    trained to predict padding, which is most of a short utterance's tail.
    """
    lens = torch.tensor([it["feature_lens"] for it in items], dtype=torch.long)
    n_mel = items[0]["input_features"].shape[0]
    feats = torch.zeros(len(items), n_mel, int(lens.max()), dtype=torch.float32)
    for i, it in enumerate(items):
        f = it["input_features"]
        feats[i, :, : f.shape[1]] = f

    tlens = [it["input_ids"].shape[0] for it in items]
    max_t = max(tlens)
    ids = torch.zeros(len(items), max_t, dtype=torch.long)
    labels = torch.full((len(items), max_t), -100, dtype=torch.long)
    for i, it in enumerate(items):
        n = tlens[i]
        ids[i, :n] = it["input_ids"]
        labels[i, :n] = it["input_ids"]
    return {"input_features": feats, "feature_lens": lens,
            "input_ids": ids, "labels": labels,
            "text_lens": torch.tensor(tlens, dtype=torch.long)}


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

    if source == "aligned":
        loaders["aligned"] = DataLoader(
            AlignedClipDataset(
                jsonl=d.aligned_jsonl,
                audio_root=d.aligned_root,
                model_dir=args.model.base_model_id,
                seed=args.seed,
            ),
            batch_size=int(d.get("aligned_micro_batch_size", 2)),
            num_workers=int(d.get("audio_workers", 4)),   # mel extraction is CPU-bound
            pin_memory=True,
            collate_fn=collate_aligned,
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
