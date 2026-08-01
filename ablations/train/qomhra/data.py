"""Datasets for the ablation CPT runs.

Four finite data-source modes:

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
  continuation — the final text/audio epoch slice plus supervised aligned batches.

`SyntheticTokenDataset` is retained for pure timing runs (no file I/O).
"""
import csv
import glob
import math
import os
import warnings

import numpy as np
import torch
from torch.utils.data import IterableDataset
# Imported at module scope, NOT lazily inside load_wav(). The python env is a
# squashfs on Lustre: a first-call import from inside a dataloader worker sends
# every worker on every rank into importlib's path crawl at once, and the metadata
# storm wedges the run before it can produce a single batch.
try:
    from scipy.io import wavfile
except ImportError:
    # Discrete-token jobs do not read WAVs. Their newer base container may be
    # binary-incompatible with SciPy left in the legacy squashfs overlay.
    wavfile = None
try:
    import torchaudio
except (ImportError, OSError):
    torchaudio = None

SAMPLE_RATE = 16000


def _epoch_step_slice(n_examples, examples_per_step, start, end):
    """Return an optimizer-step-aligned slice for a fractional epoch range."""
    n_steps = n_examples // examples_per_step
    first = int(round(n_steps * float(start)))
    last = int(round(n_steps * float(end)))
    if not 0.0 <= float(start) < float(end) <= 1.0:
        raise ValueError(f"epoch range must satisfy 0 <= start < end <= 1, got {start}:{end}")
    if last <= first:
        raise ValueError(
            f"epoch range {start}:{end} contains no optimizer step ({n_steps} total)"
        )
    return first * examples_per_step, last * examples_per_step


def _shard_and_pad(order, rank, world, local_multiple):
    """DistributedSampler semantics without a second Accelerate sharding layer.

    Every real item occurs once globally.  At most ``world * local_multiple - 1``
    leading items are repeated so every rank has the same number of complete optimizer
    steps; this padding is recorded in dataset diagnostics.
    """
    order = np.asarray(order, dtype=np.int64)
    original = len(order)
    per_rank = int(math.ceil(original / (world * local_multiple))) * local_multiple
    total = per_rank * world
    if total > original:
        order = np.resize(order, total)
    return order[rank:total:world], total - original


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

    The corpus is shuffled once, sharded across ranks, padded to complete optimizer
    steps, and yielded exactly once.  ``epoch_start/end`` selects a deterministic
    fraction of that order, allowing ablation 4 to continue from ablation 3's 90%
    checkpoint without replaying the first 90%.
    """

    def __init__(self, tokens, seq_len, batch_size, grad_acc, seed=0,
                 epoch_start=0.0, epoch_end=1.0, window_start=0,
                 window_count=None):
        self.tokens = tokens
        self.seq_len = int(seq_len)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.seed = int(seed)
        n_windows = self.tokens.numel() // self.seq_len
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n_windows)
        window_start = int(window_start)
        if window_start < 0 or window_start > len(order):
            raise ValueError(
                f"window_start must be in [0, {len(order)}], got {window_start}"
            )
        if window_count is not None:
            window_count = int(window_count)
            if window_count <= 0 or window_start + window_count > len(order):
                raise ValueError(
                    f"requested windows [{window_start:,}, "
                    f"{window_start + window_count:,}) from {len(order):,}"
                )
            order = order[window_start:window_start + window_count]
        else:
            order = order[window_start:]
        local_multiple = int(batch_size) * int(grad_acc)
        idx, self.padding = _shard_and_pad(
            order, self.rank, self.world, local_multiple
        )
        first, last = _epoch_step_slice(
            len(idx), local_multiple, epoch_start, epoch_end
        )
        self.idx = idx[first:last]
        self.examples_per_step = local_multiple

    def __len__(self):
        return len(self.idx)

    def __iter__(self):
        for j in self.idx:
            start = int(j) * self.seq_len
            ids = self.tokens[start : start + self.seq_len].to(torch.long)
            yield {"input_ids": ids, "labels": ids.clone()}


class PackedLabeledDataset(PackedTokenDataset):
    """Fixed windows with a separately prepared masked target stream.

    Discrete ASR uses one ordinary causal sequence,
    ``speech units + transcript marker + text``, but only transcript tokens carry
    labels.  Keeping input IDs and labels in parallel flat files preserves the
    packed, mask-free text fast path while making the unit prefix context rather
    than a prediction target.
    """

    def __init__(self, tokens, labels, *args, **kwargs):
        if tokens.numel() != labels.numel():
            raise ValueError(
                f"ASR token/label streams differ: {tokens.numel()} != {labels.numel()}"
            )
        self.labels = labels
        super().__init__(tokens, *args, **kwargs)

    def __iter__(self):
        for j in self.idx:
            start = int(j) * self.seq_len
            stop = start + self.seq_len
            yield {
                "input_ids": self.tokens[start:stop].to(torch.long),
                "labels": self.labels[start:stop].to(torch.long),
            }


class UnitASRExampleDataset(IterableDataset):
    """One complete unit-speech/transcript pair per rank and microbatch.

    Microbatch one needs no padding or attention mask, and every transcript sees
    its full speech prefix.  This also produces one update per eight utterances
    instead of compressing roughly 150 utterances into each packed update.
    """

    def __init__(self, parquet, batch_size, grad_acc, seed=0, split="train",
                 epoch_start=0.0, epoch_end=1.0):
        import pyarrow.parquet as pq

        cols = pq.read_table(
            parquet, columns=["input_ids", "labels", "split"]
        ).to_pydict()
        selected = [
            index for index, value in enumerate(cols["split"]) if value == split
        ]
        self.inputs = [cols["input_ids"][index] for index in selected]
        self.labels = [cols["labels"][index] for index in selected]
        if len(self.inputs) != len(self.labels) or not self.inputs:
            raise RuntimeError(f"invalid discrete-ASR example parquet: {parquet}")
        for index, (ids, labels) in enumerate(zip(self.inputs, self.labels)):
            if len(ids) != len(labels) or not any(label != -100 for label in labels):
                raise RuntimeError(f"invalid ASR example row {index} in {parquet}")

        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        local_multiple = int(batch_size) * int(grad_acc)
        rng = np.random.default_rng(int(seed))
        idx, self.padding = _shard_and_pad(
            rng.permutation(len(self.inputs)), self.rank, self.world, local_multiple
        )
        first, last = _epoch_step_slice(
            len(idx), local_multiple, epoch_start, epoch_end
        )
        self.idx = idx[first:last]
        self.examples_per_step = local_multiple
        if self.rank == 0:
            lengths = np.asarray([len(ids) for ids in self.inputs])
            print(
                f"[data] ASR examples: {len(self.inputs):,} complete utterances, "
                f"split={split}, "
                f"length mean={lengths.mean():.1f} "
                f"p95={np.percentile(lengths, 95):.0f} max={lengths.max():,}; "
                f"{len(self.idx):,}/rank, {self.padding} padding rows",
                flush=True,
            )

    def __len__(self):
        return len(self.idx)

    def __iter__(self):
        for index in self.idx:
            yield {
                "input_ids": torch.as_tensor(
                    self.inputs[int(index)], dtype=torch.long
                ),
                "labels": torch.as_tensor(
                    self.labels[int(index)], dtype=torch.long
                ),
            }


def build_text_corpus(args, token_budget=None):
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
    budget = (
        int(token_budget) if token_budget is not None
        else int(d.text_token_budget) if d.get("text_token_budget", None)
        else 0
    )
    for i in order:
        parts.append(docs[i])
        parts.append(np.array([sep_id], dtype=np.int64))
        n_tok += len(docs[i]) + 1
        if budget and n_tok >= budget:
            break
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if budget:
        if len(flat) < budget:
            raise RuntimeError(
                f"text corpus supplies {len(flat):,} tokens, below budget {budget:,}"
            )
        flat = flat[:budget]

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[data] text corpus: {len(flat):,} tokens / {len(parts)//2:,} docs"
              f" from {len(paths)} shard(s) ->"
              f" {len(flat) // int(d.seq_len):,} windows of {int(d.seq_len)}",
              flush=True)
    return torch.from_numpy(flat)


def build_unit_corpus(path, label, token_budget=None):
    """Memory-map a prepared uint32 discrete-speech token stream."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} unit token stream not found: {path}")
    tokens = np.memmap(path, dtype=np.uint32, mode="c")
    if not len(tokens):
        raise RuntimeError(f"{label} unit token stream is empty: {path}")
    if token_budget is not None:
        token_budget = int(token_budget)
        if len(tokens) < token_budget:
            raise RuntimeError(
                f"{label} unit stream supplies {len(tokens):,} tokens, "
                f"below budget {token_budget:,}"
            )
        tokens = tokens[:token_budget]
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[data] {label} units: {len(tokens):,} packed tokens from {path}",
              flush=True)
    return torch.from_numpy(tokens)


def build_asr_corpus(token_path, label_path, token_budget=None):
    """Load aligned packed input IDs (uint32) and masked labels (int32)."""
    tokens = build_unit_corpus(token_path, "ASR input", token_budget=token_budget)
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"ASR label stream not found: {label_path}")
    labels = np.memmap(label_path, dtype=np.int32, mode="c")
    if token_budget is not None:
        if len(labels) < int(token_budget):
            raise RuntimeError(
                f"ASR labels supply {len(labels):,} tokens, "
                f"below budget {int(token_budget):,}"
            )
        labels = labels[:int(token_budget)]
    if len(labels) != tokens.numel():
        raise RuntimeError(
            f"ASR streams differ: {tokens.numel():,} inputs vs {len(labels):,} labels"
        )
    # Do not scan the entire multi-GB label mmap on every distributed rank merely
    # for a diagnostic count. Artifact validation records the exact total once;
    # launch-time validation only needs to prove that sampled packed rows contain
    # targets. (The first 8192 rows span many independently packed utterances.)
    probe_tokens = min(len(labels), 8192 * 1024)
    n_target_probe = int((labels[:probe_tokens] != -100).sum())
    if not n_target_probe:
        raise RuntimeError(
            f"ASR label stream has no transcript targets in its first "
            f"{probe_tokens:,} tokens"
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"[data] ASR labels: {n_target_probe:,}/{probe_tokens:,} targets "
            f"in launch probe; {len(labels):,} packed tokens from {label_path}",
            flush=True,
        )
    return tokens, torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def load_wav(path):
    """Return float32 mono @16 kHz in [-1,1]. scipy, because the container has no
    soundfile; the corpus is 16 kHz mono int16 PCM (see speech/speech-tknz.py)."""
    scipy_wavfile = wavfile
    if scipy_wavfile is None:
        from scipy.io import wavfile as scipy_wavfile
    sr, data = scipy_wavfile.read(path)
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
        audio_lib = torchaudio
        if audio_lib is None:
            import torchaudio as audio_lib
        wav = audio_lib.functional.resample(
            torch.from_numpy(wav), sr, SAMPLE_RATE
        ).numpy()
    return wav


def encoder_frames(mel_frames):
    """Frames the Qwen audio tower emits for `mel_frames` mel frames.

    Verbatim from Qwen2_5OmniAudioEncoder._get_feat_extract_output_lengths (conv
    stride 2, then avg-pool stride 2). It is pure arithmetic on the length — no
    weights — so the loader can compute the exact label count a chunk needs without
    holding the tower, and mismatches surface here rather than as a silent one-frame
    shift in the training objective.
    """
    conv = (int(mel_frames) - 1) // 2 + 1
    return (conv - 2) // 2 + 1


def _stub_load_units(clip_id, start_s, dur_s, n_frames, n_units=1000, seed=0):
    """Random unit labels of the correct length. DRY-RUN ONLY.

    Lets the discrete-target branch be shape-tested before the real mHuBERT cache
    exists. Trains on noise, so loss should sit at log(n_units) ~ 6.9 and stay there —
    if a real run shows that, the units never got wired up.
    """
    rng = np.random.default_rng(hash((clip_id, int(start_s), seed)) % (2**32))
    return rng.integers(0, n_units, size=int(n_frames), dtype=np.int16)


def make_units_loader(cfg):
    """Resolve the `load_units(clip_id, start_s, dur_s) -> int16[L]` contract.

    `data.audio_units_loader: stub` gives random labels for dry runs; otherwise it is
    a path to the real implementation (ablations/speech/units_loader.py), imported by
    file so the training package keeps no dependency on the tokenizer environment.
    """
    spec = cfg.get("audio_units_loader", None)
    if spec in (None, False):
        return None
    if spec == "stub":
        return "stub"
    import importlib.util

    if not os.path.exists(spec):
        raise SystemExit(f"data.audio_units_loader: no such file {spec}")
    module_spec = importlib.util.spec_from_file_location("units_loader", spec)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, "load_units"):
        raise SystemExit(f"{spec} defines no `load_units` (see CODEX_BRIEF_units_loader.md)")
    return module.load_units


class AudioClipDataset(IterableDataset):
    """Streams fixed-duration audio windows as mel features.

    Long recordings are cut into `chunk_s`-second windows (the audio tower is a
    quadratic-attention transformer, and a 30 s window is ~750 frames at 25 Hz —
    one window fills a comparable context to a text sequence). Whole files are
    assigned to ranks by chunk count and yielded once. Mel extraction runs in loader
    workers so it overlaps with the GPU step; encoder forward remains on the GPU.
    """

    def __init__(self, manifest, audio_root, model_dir, batch_size, grad_acc,
                 chunk_s=30, seed=0, min_chunk_s=2.0,
                 epoch_start=0.0, epoch_end=1.0, units_loader=None, n_units=1000):
        self.units_loader = units_loader
        self.n_units = int(n_units)
        self.rows = self._read_manifest(manifest, audio_root, chunk_s, min_chunk_s)
        if not self.rows:
            raise RuntimeError(f"no audio files resolved from manifest {manifest}")
        self.model_dir = model_dir
        self.chunk_s = int(chunk_s)
        self.min_samples = int(min_chunk_s * SAMPLE_RATE)
        self.seed = int(seed)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self._fe = None

        counts = np.asarray([r[2] for r in self.rows], dtype=np.int64)
        bins = [[] for _ in range(self.world)]
        totals = np.zeros(self.world, dtype=np.int64)
        for file_idx in np.argsort(-counts, kind="stable"):
            dest = int(np.argmin(totals))
            bins[dest].append(int(file_idx))
            totals[dest] += counts[file_idx]

        local_multiple = int(batch_size) * int(grad_acc)
        per_rank = int(math.ceil(int(totals.max()) / local_multiple)) * local_multiple
        rng = np.random.default_rng(self.seed + self.rank)
        local_files = rng.permutation(bins[self.rank])
        file_ids = np.repeat(local_files, counts[local_files]).astype(np.int64)
        chunk_ids = np.concatenate([
            np.arange(counts[i], dtype=np.int32) for i in local_files
        ]) if len(local_files) else np.zeros(0, dtype=np.int32)
        real_count = len(file_ids)
        if not real_count:
            raise RuntimeError(f"rank {self.rank} received no audio chunks")
        if real_count < per_rank:
            pad = per_rank - real_count
            file_ids = np.concatenate([file_ids, np.resize(file_ids, pad)])
            chunk_ids = np.concatenate([chunk_ids, np.resize(chunk_ids, pad)])
        first, last = _epoch_step_slice(
            per_rank, local_multiple, epoch_start, epoch_end
        )
        self.file_ids = file_ids[first:last]
        self.chunk_ids = chunk_ids[first:last]
        self.padding = per_rank * self.world - int(counts.sum())
        self.examples_per_step = local_multiple

        if self.rank == 0:
            print(f"[data] audio corpus: {len(self.rows):,} files / "
                  f"{int(counts.sum()):,} chunks; {per_rank:,}/rank "
                  f"(+{self.padding:,} distributed padding), "
                  f"slice={epoch_start:.1%}:{epoch_end:.1%}",
                  flush=True)

    def __len__(self):
        return len(self.file_ids)

    @staticmethod
    def _read_manifest(manifest, audio_root, chunk_s, min_chunk_s):
        rows = []
        skipped_bad_size = []
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                p = os.path.join(audio_root, r["dest_label"], r["filename"])
                duration = float(r["duration_s"])
                # The corpus is 16 kHz mono int16 PCM.  A damaged RIFF header can
                # advertise a multi-hour data chunk even though the file ends after
                # a few minutes (for example TG4/5537750827001.wav).  Filter such
                # rows before rank assignment: skipping later in __iter__ would give
                # ranks different batch counts and hang the next collective.
                expected_pcm_bytes = duration * SAMPLE_RATE * 2
                try:
                    actual_bytes = os.path.getsize(p)
                except OSError:
                    skipped_bad_size.append((p, "missing"))
                    continue
                # Allow for rounded manifest durations, while catching files short
                # enough that their planned final chunk could disappear at runtime.
                if actual_bytes < 0.98 * expected_pcm_bytes:
                    skipped_bad_size.append(
                        (p, f"{actual_bytes} B vs ~{expected_pcm_bytes:.0f} B expected")
                    )
                    continue
                whole = int(duration // float(chunk_s))
                remainder = duration - whole * float(chunk_s)
                n_chunks = whole + int(remainder >= float(min_chunk_s))
                if n_chunks:
                    rows.append((p, duration, n_chunks))
        if skipped_bad_size and int(os.environ.get("RANK", "0")) == 0:
            preview = "; ".join(f"{p} ({why})" for p, why in skipped_bad_size[:5])
            suffix = " ..." if len(skipped_bad_size) > 5 else ""
            print(f"[data] skipped {len(skipped_bad_size):,} missing/truncated audio "
                  f"file(s): {preview}{suffix}", flush=True)
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
        bounds = np.linspace(0, len(self.file_ids), nworkers + 1, dtype=np.int64)
        first, last = int(bounds[wid]), int(bounds[wid + 1])
        if first == last:
            return

        fe = self._feature_extractor()
        n = self.chunk_s * SAMPLE_RATE
        cached_idx, wav = None, None
        last_good = None
        failed_files = set()
        warned = set()

        def example_at(pos):
            """Decode one planned chunk; callers decide how to recover on failure."""
            nonlocal cached_idx, wav
            file_idx = int(self.file_ids[pos])
            if file_idx in failed_files:
                raise RuntimeError("audio file previously failed to decode")
            if file_idx != cached_idx:
                try:
                    wav = load_wav(self.rows[file_idx][0])
                except Exception:
                    failed_files.add(file_idx)
                    cached_idx, wav = None, None
                    raise
                cached_idx = file_idx
            start = int(self.chunk_ids[pos]) * n
            chunk = wav[start : start + n]
            if len(chunk) < self.min_samples:
                raise RuntimeError("decoded audio is shorter than its manifest duration")
            feats = fe(chunk, sampling_rate=SAMPLE_RATE,
                       return_attention_mask=True, return_tensors="pt")
            mel = feats["input_features"][0]
            mask = feats.get("feature_attention_mask")
            if mask is None:
                mask = feats["attention_mask"]
            flen = int(mask[0].sum())
            example = {"input_features": mel[:, :flen], "feature_lens": flen}

            if self.units_loader is not None:
                path = self.rows[file_idx][0]
                start_s = int(self.chunk_ids[pos]) * float(self.chunk_s)
                dur_s = len(chunk) / SAMPLE_RATE
                want = encoder_frames(flen)
                if self.units_loader == "stub":
                    units = _stub_load_units(path, start_s, dur_s, want, self.n_units)
                else:
                    units = self.units_loader(path, start_s, dur_s)
                # Checked here, where the offending clip is still named. A silent
                # off-by-one would train the model on a shifted task and the loss
                # would still fall.
                if len(units) != want:
                    raise RuntimeError(
                        f"units length {len(units)} != {want} encoder frames for "
                        f"{path} @ {start_s:.1f}s (+{dur_s:.1f}s, {flen} mel frames)"
                    )
                example["unit_labels"] = torch.as_tensor(
                    np.asarray(units), dtype=torch.long
                )
            return example

        for pos in range(first, last):
            file_idx = int(self.file_ids[pos])
            try:
                example = example_at(pos)
                last_good = example
            except Exception as exc:
                path = self.rows[file_idx][0]
                if path not in warned:
                    warnings.warn(
                        f"skipping unusable audio at runtime: {path}: {exc}",
                        RuntimeWarning,
                    )
                    warned.add(path)

                # Ranks must yield identical numbers of batches.  If this worker has
                # already decoded something valid, substitute it for the bad chunk.
                # Otherwise scan forward once to seed a fallback.  This duplicates a
                # tiny amount of data but cannot desynchronise FSDP collectives.
                if last_good is None:
                    tried_files = {file_idx}
                    for fallback_pos in range(pos + 1, last):
                        fallback_idx = int(self.file_ids[fallback_pos])
                        if fallback_idx in tried_files:
                            continue
                        tried_files.add(fallback_idx)
                        try:
                            last_good = example_at(fallback_pos)
                            break
                        except Exception:
                            continue
                    if last_good is None:
                        raise RuntimeError(
                            f"worker {wid} could not find any usable fallback audio"
                        ) from exc
                example = last_good
            yield example


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

    def __init__(self, jsonl=None, audio_root=None, model_dir=None, parquet=None,
                 batch_size=4, grad_acc=1, max_s=30.0, seed=0,
                 exclude_heldout=True, epoch_start=0.0, epoch_end=1.0):
        self.rows = []
        n_held = 0
        if parquet:
            import pyarrow.parquet as pq
            cols = pq.read_table(
                parquet, columns=["audio_filepath", "duration", "text"]
            ).to_pydict()
            records = zip(cols["audio_filepath"], cols["duration"], cols["text"])
        elif jsonl:
            import json
            def json_records():
                with open(jsonl, encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        yield (os.path.join(audio_root, r["audio"]),
                               r["duration_s"], r["text"])
            records = json_records()
        else:
            raise ValueError("aligned dataset requires parquet or jsonl")
        for path, duration, text in records:
            if float(duration) > max_s:
                continue
            if exclude_heldout and self.is_heldout(path):
                n_held += 1
                continue
            self.rows.append((path, text))
        if not self.rows:
            raise RuntimeError(f"no aligned rows from {parquet or jsonl}")
        if exclude_heldout and n_held == 0:
            raise RuntimeError("exclude_heldout=True but nothing was held out — the "
                               "eval split would be training data and every WER "
                               "would be memorisation")
        self.model_dir = model_dir
        self.seed = int(seed)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        local_multiple = int(batch_size) * int(grad_acc)
        rng = np.random.default_rng(self.seed)
        idx, self.padding = _shard_and_pad(
            rng.permutation(len(self.rows)), self.rank, self.world, local_multiple
        )
        first, last = _epoch_step_slice(
            len(idx), local_multiple, epoch_start, epoch_end
        )
        self.idx = idx[first:last]
        self.examples_per_step = local_multiple
        self._fe = None
        self._tok = None
        if self.rank == 0:
            print(f"[data] aligned corpus: {len(self.rows):,} utterances, "
                  f"{len(self.idx):,}/rank in selected slice, "
                  f"{n_held:,} held out for eval",
                  flush=True)

    def __len__(self):
        return len(self.idx)

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
        bounds = np.linspace(0, len(self.idx), nworkers + 1, dtype=np.int64)
        first, last = int(bounds[wid]), int(bounds[wid + 1])
        order = self.idx[first:last]
        if not len(order):
            return
        fe, tok = self._extractors()
        for row_idx in order:
            path, text = self.rows[int(row_idx)]
            try:
                wav = load_wav(path)
            except Exception as exc:
                raise RuntimeError(f"failed to load aligned audio {path}") from exc
            feats = fe(wav, sampling_rate=SAMPLE_RATE,
                       return_attention_mask=True, return_tensors="pt")
            mask = feats.get("feature_attention_mask")
            if mask is None:
                mask = feats["attention_mask"]
            flen = int(mask[0].sum())
            # Teach the same terminator used by evaluation; otherwise a correct
            # transcript can continue into repetitions that score as insertions.
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
    batch = {"input_features": out, "feature_lens": lens}

    if "unit_labels" in items[0]:
        # Padded to the batch's encoder-frame max, matching the (B, F) frame_mask the
        # tower produces. -100 so cross_entropy ignores it even if the mask slips.
        n_frames = max(len(it["unit_labels"]) for it in items)
        units = torch.full((len(items), n_frames), -100, dtype=torch.long)
        for i, it in enumerate(items):
            u = it["unit_labels"]
            units[i, : len(u)] = u
        batch["unit_labels"] = units
    return batch


def get_dataloader(args, vocab_size):
    """Return finite per-modality loaders and optimizer-step capacities.

    These datasets already shard by global rank. Do not pass the loaders through
    ``accelerator.prepare`` or the corpus will be sharded a second time.
    """
    from torch.utils.data import DataLoader

    d = args.data
    source = d.source
    loaders = {}
    capacities = {}
    grad_acc = int(args.optim.grad_acc)
    epoch_start = float(d.get("epoch_start", 0.0))
    epoch_end = float(d.get("epoch_end", 1.0))

    if source in ("discrete_mixed", "discrete_mixed_asr"):
        from .plan import allocate_stream_steps

        weights = dict(d.stream_weights)
        if source == "discrete_mixed" and float(weights.get("asr", 0)) > 0:
            raise ValueError("discrete_mixed cannot include an ASR stream")
        allowed = {"text", "units", "asr"}
        unknown = set(weights) - allowed
        if unknown:
            raise ValueError(f"unknown discrete stream(s): {sorted(unknown)}")
        if source == "discrete_mixed_asr" and float(weights.get("asr", 0)) <= 0:
            raise ValueError("discrete_mixed_asr requires a positive ASR weight")

        world = int(os.environ.get("WORLD_SIZE", "1"))
        batch = int(d.micro_batch_size)
        tokens_per_step = int(d.seq_len) * batch * grad_acc * world
        total_steps = int(d.token_budget) // tokens_per_step
        step_counts = allocate_stream_steps(total_steps, weights)
        stream_budgets = {
            key: steps * tokens_per_step for key, steps in step_counts.items()
        }

        for key, token_budget in stream_budgets.items():
            window_count = token_budget // int(d.seq_len)
            if key == "text":
                tokens = build_unit_corpus(d.text_bin, "text")
                dataset = PackedTokenDataset(
                    tokens, seq_len=d.seq_len, batch_size=batch,
                    grad_acc=grad_acc, seed=args.seed,
                    epoch_start=epoch_start, epoch_end=epoch_end,
                    window_count=window_count,
                )
            elif key == "units":
                tokens = build_unit_corpus(
                    d.unit_train_bin, "train"
                )
                dataset = PackedTokenDataset(
                    tokens, seq_len=d.seq_len, batch_size=batch,
                    grad_acc=grad_acc, seed=args.seed + 1,
                    epoch_start=epoch_start, epoch_end=epoch_end,
                    window_count=window_count,
                )
            else:
                tokens, labels = build_asr_corpus(
                    d.asr_input_bin, d.asr_labels_bin,
                )
                dataset = PackedLabeledDataset(
                    tokens, labels, seq_len=d.seq_len, batch_size=batch,
                    grad_acc=grad_acc, seed=args.seed + 2,
                    epoch_start=epoch_start, epoch_end=epoch_end,
                    window_count=window_count,
                )
            loaders[key] = DataLoader(
                dataset, batch_size=batch, num_workers=0,
                pin_memory=True, drop_last=True,
            )
            capacities[key] = len(dataset) // (batch * grad_acc)

        if int(os.environ.get("RANK", "0")) == 0:
            realised = {
                key: capacities[key] / sum(capacities.values())
                for key in sorted(capacities)
            }
            print(
                f"[data] discrete stream steps={dict(sorted(capacities.items()))}; "
                f"realised={realised}; total={sum(capacities.values()):,}",
                flush=True,
            )
        return loaders, capacities

    if source in (
        "real", "mixed", "continuation", "units", "unit_asr",
        "unit_asr_examples",
    ):
        if source == "units":
            tokens = build_unit_corpus(d.unit_train_bin, "train")
            dataset = PackedTokenDataset(
                tokens, seq_len=d.seq_len, batch_size=d.micro_batch_size,
                grad_acc=grad_acc, seed=args.seed,
                epoch_start=epoch_start, epoch_end=epoch_end,
            )
        elif source == "unit_asr":
            tokens, labels = build_asr_corpus(
                d.asr_input_bin, d.asr_labels_bin
            )
            dataset = PackedLabeledDataset(
                tokens, labels, seq_len=d.seq_len,
                batch_size=d.micro_batch_size, grad_acc=grad_acc,
                seed=args.seed, epoch_start=epoch_start, epoch_end=epoch_end,
            )
        elif source == "unit_asr_examples":
            if int(d.micro_batch_size) != 1:
                raise ValueError(
                    "unit_asr_examples requires micro_batch_size=1"
                )
            dataset = UnitASRExampleDataset(
                d.asr_examples_parquet,
                batch_size=d.micro_batch_size,
                grad_acc=grad_acc,
                seed=args.seed,
                split=d.get("asr_split", "train"),
                epoch_start=epoch_start,
                epoch_end=epoch_end,
            )
        else:
            tokens = build_text_corpus(args)
            dataset = PackedTokenDataset(
                tokens, seq_len=d.seq_len, batch_size=d.micro_batch_size,
                grad_acc=grad_acc, seed=args.seed,
                epoch_start=epoch_start, epoch_end=epoch_end,
            )
        loaders["text"] = DataLoader(
            dataset,
            batch_size=d.micro_batch_size,
            num_workers=0,          # corpus is already in memory
            pin_memory=True,
            drop_last=True,
        )
        capacities["text"] = len(dataset) // (int(d.micro_batch_size) * grad_acc)

    if source in ("audio", "mixed", "continuation"):
        dataset = AudioClipDataset(
            manifest=d.audio_manifest,
            audio_root=d.audio_root,
            model_dir=args.model.base_model_id,
            batch_size=d.audio_micro_batch_size,
            grad_acc=grad_acc,
            chunk_s=d.audio_chunk_s,
            seed=args.seed,
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            units_loader=make_units_loader(d),
            n_units=int(args.get("task", {}).get("audio_n_units", 1000)),
        )
        loaders["audio"] = DataLoader(
            dataset,
            batch_size=d.audio_micro_batch_size,
            num_workers=int(d.get("audio_workers", 4)),   # mel extraction is CPU-bound
            pin_memory=True,
            collate_fn=collate_audio,
            drop_last=True,
            persistent_workers=int(d.get("audio_workers", 4)) > 0,
        )
        capacities["audio"] = len(dataset) // (
            int(d.audio_micro_batch_size) * grad_acc
        )

    if source in ("aligned", "continuation"):
        aligned_batch = int(d.get("aligned_micro_batch_size", 2))
        dataset = AlignedClipDataset(
            jsonl=d.get("aligned_jsonl", None),
            parquet=d.get("aligned_parquet", None),
            audio_root=d.get("aligned_root", None),
            model_dir=args.model.base_model_id,
            batch_size=aligned_batch,
            grad_acc=grad_acc,
            seed=args.seed + 17,
        )
        loaders["aligned"] = DataLoader(
            dataset,
            batch_size=aligned_batch,
            num_workers=int(d.get("audio_workers", 4)),   # mel extraction is CPU-bound
            pin_memory=True,
            collate_fn=collate_aligned,
            drop_last=True,
            persistent_workers=int(d.get("audio_workers", 4)) > 0,
        )
        capacities["aligned"] = len(dataset) // (aligned_batch * grad_acc)

    if source == "synthetic":
        loaders["text"] = DataLoader(
            SyntheticTokenDataset(vocab_size, seq_len=d.seq_len, seed=args.seed),
            batch_size=d.micro_batch_size,
            num_workers=0,
            pin_memory=True,
        )
        max_steps = args.optim.get("max_steps", None)
        if max_steps is None:
            raise ValueError("synthetic data requires optim.max_steps")
        capacities["text"] = int(max_steps)

    if not loaders:
        raise ValueError(f"unknown data.source: {source!r}")
    return loaders, capacities


def get_validation_dataloaders(args):
    """Build a fixed held-out suite matching the configured discrete mixture."""
    from torch.utils.data import DataLoader
    from .plan import allocate_stream_steps

    d = args.data
    if d.source not in ("discrete_mixed", "discrete_mixed_asr"):
        if d.source != "units":
            return {}
        weights = {"units": 1.0}
    else:
        weights = dict(d.stream_weights)

    batch = int(d.micro_batch_size)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    total_steps = int(d.get("validation_steps", 128))
    counts = allocate_stream_steps(total_steps, weights)
    loaders = {}
    for key, steps in counts.items():
        global_windows = int(steps) * batch * world
        if key == "text":
            tokens = build_unit_corpus(d.text_bin, "text validation source")
            total_windows = tokens.numel() // int(d.seq_len)
            dataset = PackedTokenDataset(
                tokens, seq_len=d.seq_len, batch_size=batch, grad_acc=1,
                seed=args.seed, window_start=total_windows - global_windows,
                window_count=global_windows,
            )
        elif key == "units":
            tokens = build_unit_corpus(d.unit_validation_bin, "unit validation")
            dataset = PackedTokenDataset(
                tokens, seq_len=d.seq_len, batch_size=batch, grad_acc=1,
                seed=args.seed + 991, window_count=global_windows,
            )
        else:
            tokens, labels = build_asr_corpus(
                d.asr_validation_input_bin, d.asr_validation_labels_bin
            )
            dataset = PackedLabeledDataset(
                tokens, labels, seq_len=d.seq_len, batch_size=batch, grad_acc=1,
                seed=args.seed + 992, window_count=global_windows,
            )
        loaders[key] = DataLoader(
            dataset, batch_size=batch, num_workers=0,
            pin_memory=True, drop_last=True,
        )
    return loaders
