#!/usr/bin/env python3
"""MEXA embedding extraction: four parallel object types, every layer, one model load.

MEXA scores alignment from HIDDEN STATES, so this script never generates. It runs forward
passes only, which is what lets it produce a meaningful number for checkpoints that score
zero on every generative benchmark: babbling, prompt format and stop tokens cannot reach it.

Four embeddable objects per model, all over the SAME 100 parallel FLEURS sentences:

    text_ga    Irish sentence, bare
    text_en    English sentence, bare
    speech_ga  Irish recording of that sentence
    speech_en  English recording of that sentence

The six MEXA conditions are then unordered pairs over these four; pairing happens in
mexa_score.py, so every condition is guaranteed to come from one model load.

Pooling follows the reference embed_extractor.py exactly:
  * weighted  - position-weighted average, token at 1-based position i weighted by i:
                sum(h_i * i) / sum(i). Emphasises later tokens.
  * lasttoken - the final real position.
Both are computed in the same pass and both are reported, so we can show the choice of
pooling does not drive the result.

The one extension that is OURS, not the paper's: MEXA targets decoder-only TEXT models. For
audio we wrap the waveform in <|audio_bos|><|AUDIO|><|audio_eos|>, and the processor expands
<|AUDIO|> into N placeholder positions. We pool over THOSE POSITIONS ONLY - never the bos/eos
markers, which carry no content - with the position weights restarting at 1 inside the span,
which is the faithful analogue of the text formula.

TWO SPEECH PATHS, and they are not interchangeable
--------------------------------------------------
`--speech continuous` (default) is the above: waveform -> audio tower -> decoder.
`--speech units` feeds mHuBERT unit IDs straight in as ordinary token ids and never runs
the audio tower at all.

Which one is correct depends entirely on the checkpoint. The discrete ablations do not
train the audio tower - it is dead weight they never touch - so measuring them through it
measures the stock base model's audio tower, not what training changed. Conversely the
superseded continuous-audio runs have no unit embeddings to look at. Pick the path that
matches how the model was trained; the sidecar json records which was used.

The unit sequence is `<speech_start> deduped_units <speech_end>`, the prefix half of the
training contract in prepare_fleurs_discrete_asr.py. We pool over the units only, so the
markers get no weight - the same rule as the audio placeholder span.

Usage (one model per invocation, or several to amortise nothing but the data load):
  python mexa_embed.py --data data/fleurs_parallel_test.parquet --n 100 \
      --models base=base text-10pct=/scratch/.../step_000819 --out data/mexa_embeddings

  python mexa_embed.py --data data/fleurs_parallel_test.parquet --n 100 \
      --speech units --unit-parquet data/fleurs_mexa_units.parquet \
      --models asr250h=/scratch/.../step_023863 --out data/mexa_embeddings_units
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

# MIOpen's conv is broken in the LUMI rocm6.2 container (it writes its kernel DBs as
# DIRECTORIES, so every conv returns miopenStatusInternalError) and the audio tower's conv1d
# dies on it. Must be set before the first conv. Same line, same reason, as iwslt_st.py.
torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iwslt_st import AUDIO_MARKERS, TARGET_SR, decode_audio  # noqa: E402

SNAPSHOT_GLOB = ("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")
OBJECTS = ("text_ga", "text_en", "speech_ga", "speech_en")

# Discrete-speech vocabulary, copied from prepare_fleurs_discrete_asr.py. These are the ids
# the ablations were trained on; nothing here may drift from that file.
BASE_VOCAB = 151936
UNIT_COUNT = 1000
SPEECH_START = BASE_VOCAB + UNIT_COUNT          # 152936
SPEECH_END = SPEECH_START + 1                   # 152937
TRANSCRIPT_START = SPEECH_END + 1               # 152938


# ---------------------------------------------------------------------------
# Data: the same 100 sentences for every model and every condition.
# ---------------------------------------------------------------------------
def load_sentences(parquet_path, n):
    """First n sentence ids of the FLEURS parallel file, sorted, from the TEST rows.

    Taking test rows (not the 3 demo rows FLEURS holds out) keeps this set a strict subset of
    FLEURS' scored set, so the MEXA-to-FLEURS validation join later is on the SAME sentences
    rather than merely the same corpus.

    Text is the RAW transcription column, the same choice for both languages.
    """
    import pyarrow.parquet as pq
    records = pq.read_table(parquet_path).to_pylist()
    test = sorted((r for r in records if r["split"] == "test"), key=lambda r: r["id"])
    if len(test) < n:
        raise SystemExit(f"{parquet_path}: only {len(test)} test rows, need {n}")
    chosen = test[:n]
    print(f"[data] {len(records)} rows in file, {len(test)} test, using first {n} "
          f"(sentence ids {chosen[0]['id']}..{chosen[-1]['id']})", flush=True)
    return chosen


def build_objects(rows):
    """The four parallel object lists, row-for-row aligned by construction."""
    return {
        "text_ga": [r["ga_raw"] for r in rows],
        "text_en": [r["en_raw"] for r in rows],
        "speech_ga": [decode_audio(r["ga_audio"]) for r in rows],
        "speech_en": [decode_audio(r["en_audio"]) for r in rows],
    }


def build_objects_units(rows, unit_parquet):
    """Same four lists, but the speech sides are unit-id arrays instead of waveforms.

    Keyed on sentence id and language, and every requested id must be present: a silent
    fallback here would misalign the rows and the score would be meaningless rather than
    merely wrong.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(unit_parquet, columns=["id", "lang", "units"]).to_pylist()
    store = {(r["id"], r["lang"]): np.asarray(r["units"], dtype=np.int64) for r in table}

    speech = {}
    for lang, obj in (("ga", "speech_ga"), ("en", "speech_en")):
        arrays = []
        for r in rows:
            units = store.get((r["id"], lang))
            if units is None:
                raise SystemExit(
                    f"{unit_parquet}: no units for sentence {r['id']} ({lang}). Run "
                    f"fleurs_mexa_units.py over the same --data and --n as this script.")
            if units.size == 0:
                raise SystemExit(f"sentence {r['id']} ({lang}): empty unit sequence")
            if units.min() < 0 or units.max() >= UNIT_COUNT:
                raise SystemExit(f"sentence {r['id']} ({lang}): unit ids outside "
                                 f"[0, {UNIT_COUNT}) - wrong codebook?")
            arrays.append(units)
        speech[obj] = arrays

    lengths = [len(a) for a in speech["speech_ga"]] + [len(a) for a in speech["speech_en"]]
    print(f"[units] {len(rows)} sentences x 2 langs, dedup length "
          f"min={min(lengths)} median={int(np.median(lengths))} max={max(lengths)}",
          flush=True)
    return {
        "text_ga": [r["ga_raw"] for r in rows],
        "text_en": [r["en_raw"] for r in rows],
        **speech,
    }


# ---------------------------------------------------------------------------
# Pooling. Both formulas are the reference implementation's, restricted to a span.
# ---------------------------------------------------------------------------
def pool(hidden, start, end):
    """hidden: [seq, d] for one layer. Pool positions [start, end) two ways.

    Returns (weighted, lasttoken) as float32 numpy. Weights are 1..L over the span, which for
    text (span = whole sequence) is exactly embed_extractor.py's arange(1, seq_len+1), and for
    audio restarts inside the audio placeholder span so the markers get no weight at all.
    """
    span = hidden[start:end].to(torch.float32)
    length = span.shape[0]
    weights = torch.arange(1, length + 1, device=span.device, dtype=torch.float32)
    weighted = (span * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
    return weighted.cpu().numpy(), span[-1].cpu().numpy()


def audio_span(input_ids, audio_token_id):
    """[start, end) of the audio placeholder positions the processor expanded <|AUDIO|> into.

    Asserts the positions are contiguous: one clip per forward pass, so a gap would mean the
    processor laid the sequence out differently than assumed and the pooling would be wrong.
    """
    positions = (input_ids == audio_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise SystemExit("no <|AUDIO|> placeholder positions in the tokenised prompt - the "
                         "processor did not expand the audio marker")
    start, end = int(positions[0]), int(positions[-1]) + 1
    if end - start != positions.numel():
        raise SystemExit("audio placeholder positions are not contiguous; pooling assumes "
                         "one contiguous span per clip")
    return start, end


# ---------------------------------------------------------------------------
# Extraction.
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_one(thinker, proc, item, kind, device, audio_token_id,
              unit_representation="pooled_units"):
    """One sentence or one clip -> (weighted, lasttoken), each [n_layers+1, d] float32."""
    if kind == "text":
        # Bare sentence: no chat template, no instruction, no demos. Every position is the
        # sentence's own, so the span is the whole sequence.
        inputs = proc(text=item, return_tensors="pt").to(device)
        start, end = 0, inputs["input_ids"].shape[1]
    elif kind == "units":
        # The original unit MEXA pools the content-unit positions. The ASR-boundary
        # variant instead mirrors the trained causal prefix and extracts the hidden
        # state at <transcript_start>, i.e. the exact state whose logits predict the
        # first transcript token. Cap at the trained 1024-token context without using
        # the reference transcript length (which would leak the answer into the input).
        units = (item + BASE_VOCAB).tolist()
        if unit_representation == "asr_boundary":
            units = units[:1021]  # start + units + end + transcript_start <= 1024
            seq = [SPEECH_START] + units + [SPEECH_END, TRANSCRIPT_START]
        else:
            seq = [SPEECH_START] + units + [SPEECH_END]
        ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        inputs = {"input_ids": ids}
        if unit_representation == "asr_boundary":
            # A one-position span makes weighted and last-token pooling identical on
            # the speech side while retaining both text-pooling controls downstream.
            start, end = ids.shape[1] - 1, ids.shape[1]
        else:
            # Units only; the two markers carry no content, exactly as with audio.
            start, end = 1, ids.shape[1] - 1
    else:
        inputs = proc(text=AUDIO_MARKERS, audio=[item], sampling_rate=TARGET_SR,
                      return_tensors="pt", padding=True).to(device)
        start, end = audio_span(inputs["input_ids"][0], audio_token_id)

    out = thinker(**inputs, output_hidden_states=True, return_dict=True)
    weighted, lasttoken = [], []
    for layer_hidden in out.hidden_states:            # n_layers + 1, includes the embeddings
        w, l = pool(layer_hidden[0], start, end)
        weighted.append(w)
        lasttoken.append(l)
    return np.stack(weighted), np.stack(lasttoken)


def check_unit_vocab(thinker, label, unit_representation="pooled_units"):
    """The unit ids must be real rows of the embedding matrix, not out-of-range indices.

    A stock Qwen2.5-Omni-3B has 151936 rows and knows nothing about units, so asking it for
    row 152936 is an indexing error at best and garbage at worst. Only checkpoints trained
    with the expanded vocabulary can be scored this way, and this is where that is enforced.
    """
    rows = thinker.get_input_embeddings().weight.shape[0]
    required_id = (
        TRANSCRIPT_START
        if unit_representation == "asr_boundary"
        else SPEECH_END
    )
    if rows <= required_id:
        raise SystemExit(
            f"[{label}] embedding matrix has {rows} rows, but this unit representation "
            f"needs at least {required_id + 1}. This checkpoint was not trained with the expanded speech "
            f"vocabulary, so it has no unit embeddings to measure. Use --speech continuous, "
            f"or score a discrete checkpoint.")
    print(f"[{label}] vocab {rows} rows; units {BASE_VOCAB}..{BASE_VOCAB + UNIT_COUNT - 1}, "
          f"markers {SPEECH_START}/{SPEECH_END}", flush=True)


def extract(thinker, proc, objects, device, label, speech_mode="continuous",
            unit_representation="pooled_units"):
    """All four object types for one model. Returns {'obj.pooling': [layers, n, d] fp16}."""
    audio_token_id = proc.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    if audio_token_id is None or audio_token_id == proc.tokenizer.unk_token_id:
        raise SystemExit("tokenizer does not know <|AUDIO|>")
    if speech_mode == "units":
        check_unit_vocab(thinker, label, unit_representation)

    speech_kind = "units" if speech_mode == "units" else "audio"
    arrays, shapes = {}, {}
    for obj in OBJECTS:
        kind = "text" if obj.startswith("text") else speech_kind
        started = time.time()
        weighted, lasttoken = [], []
        for index, item in enumerate(objects[obj], start=1):
            w, l = embed_one(
                thinker, proc, item, kind, device, audio_token_id,
                unit_representation=unit_representation,
            )
            weighted.append(w)
            lasttoken.append(l)
            if index % 25 == 0 or index == len(objects[obj]):
                print(f"[{label}/{obj}] {index}/{len(objects[obj])} "
                      f"({(time.time() - started) / index:.2f}s each)", flush=True)
        # [n, layers, d] -> [layers, n, d]: scoring works one layer at a time.
        for pooling, stack in (("weighted", weighted), ("lasttoken", lasttoken)):
            array = np.stack(stack).transpose(1, 0, 2)
            check_no_nan(label, obj, pooling, array)
            arrays[f"{obj}.{pooling}"] = array.astype(np.float16)
            shapes[f"{obj}.{pooling}"] = array.shape
    return arrays, shapes


def check_no_nan(label, obj, pooling, array):
    """On ROCm a NaN'd audio path has NO visible symptom in this script - there is no text to
    decode, so it would quietly produce a wrong number instead of a wall of '!'. Fail loudly.

    Also catches all-zero vectors, which cosine similarity cannot normalise.
    """
    if not np.isfinite(array).all():
        bad = np.argwhere(~np.isfinite(array).all(axis=-1))
        raise SystemExit(
            f"[{label}] {obj}.{pooling}: non-finite values at (layer, sentence) {bad[:5].tolist()}"
            f" - almost certainly ROCm SDPA on masked audio. Load with attn_implementation="
            f"'eager' and check torch.backends.cudnn.enabled is False. Refusing to save.")
    norms = np.linalg.norm(array, axis=-1)
    if (norms == 0).any():
        raise SystemExit(f"[{label}] {obj}.{pooling}: zero-norm embeddings; cosine is undefined")


# ---------------------------------------------------------------------------
def save(out_dir, label, arrays, shapes, meta):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{label}.npz")
    np.savez(npz_path, **arrays)
    meta = dict(meta, shapes={k: list(v) for k, v in shapes.items()})
    with open(os.path.join(out_dir, f"{label}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(npz_path) / 1e6
    print(f"[{label}] wrote {npz_path} ({size_mb:.0f} MB) + sidecar json", flush=True)


def parse_model_spec(spec):
    """'label=path' with path 'base' meaning the stock Qwen2.5-Omni-3B snapshot."""
    if "=" not in spec:
        raise SystemExit(f"--models expects label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="fleurs_parallel_test.parquet")
    ap.add_argument("--models", nargs="+", required=True, help="label=path (path 'base' = "
                                                               "stock Qwen2.5-Omni-3B)")
    ap.add_argument("--n", type=int, default=100, help="parallel sentences (MEXA default 100)")
    ap.add_argument("--baseline", default=None, help="override the base snapshot path")
    ap.add_argument("--out", required=True, help="directory for the .npz + .json per model")
    ap.add_argument("--recipe", default=None, help="recipe name recorded in the sidecar "
                                                   "(text/speech/both/aligned/base)")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--train-fraction", type=float, default=None,
                    help="global training progress in [0,1]; aligned's phase maps to 0.9-1.0")
    ap.add_argument("--speech", default="continuous", choices=["continuous", "units"],
                    help="continuous = waveform through the audio tower (the original path); "
                         "units = mHuBERT unit ids as ordinary tokens, tower never runs")
    ap.add_argument("--unit-parquet", default=None,
                    help="required with --speech units: output of fleurs_mexa_units.py attach")
    ap.add_argument(
        "--unit-representation",
        default="pooled_units",
        choices=["pooled_units", "asr_boundary"],
        help="pooled_units = pool content-unit states; asr_boundary = append "
             "<transcript_start> and extract that causal boundary state",
    )
    args = ap.parse_args()

    if args.speech == "units" and not args.unit_parquet:
        ap.error("--speech units requires --unit-parquet")
    if args.speech != "units" and args.unit_representation != "pooled_units":
        ap.error("--unit-representation applies only with --speech units")

    if args.baseline is None:
        import glob
        snaps = glob.glob(SNAPSHOT_GLOB)
        if not snaps:
            raise SystemExit("no local Qwen2.5-Omni-3B snapshot; pass --baseline")
        args.baseline = snaps[0]

    rows = load_sentences(args.data, args.n)
    objects = (build_objects_units(rows, args.unit_parquet) if args.speech == "units"
               else build_objects(rows))
    sentence_ids = [r["id"] for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(args.baseline)

    for spec in args.models:
        label, path = parse_model_spec(spec)
        model_step = args.step
        model_train_fraction = args.train_fraction
        if path == "base":
            print(f"[{label}] loading stock Qwen2.5-Omni-3B ...", flush=True)
            from generate_compare import load_baseline
            # eager, NOT sdpa: masked audio on ROCm SDPA returns NaN.
            thinker = load_baseline(args.baseline, device, dtype, attn="eager")
            model_path = args.baseline
        else:
            print(f"[{label}] loading {path} ...", flush=True)
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
            from qomhra.checkpoint import load_for_eval
            model, _ = load_for_eval(path, device=device, dtype=dtype)
            thinker = model.thinker
            model_path = path
            meta_path = os.path.join(path, "meta.json")
            if os.path.isfile(meta_path):
                checkpoint_meta = json.load(open(meta_path, encoding="utf-8"))
                if model_step is None:
                    model_step = checkpoint_meta.get("step")
                if model_train_fraction is None:
                    model_train_fraction = checkpoint_meta.get("epoch_progress")

        started = time.time()
        arrays, shapes = extract(
            thinker, proc, objects, device, label, args.speech,
            unit_representation=args.unit_representation,
        )
        layers, n_sents, hidden = shapes["text_ga.weighted"]
        print(f"[{label}] {layers} hidden-state tensors (n_layers+1) x {hidden}d, "
              f"n={n_sents}, {time.time() - started:.0f}s", flush=True)

        save(args.out, label, arrays, shapes, {
            "label": label,
            "model": args.recipe or label.split("-")[0],
            "step": model_step,
            "train_fraction": model_train_fraction,
            "model_path": os.path.abspath(model_path),
            "data": os.path.abspath(args.data),
            "n_sentences": n_sents,
            "sentence_ids": sentence_ids,
            "text_column": "raw_transcription",
            "n_hidden_state_tensors": layers,
            "hidden_size": hidden,
            "poolings": ["weighted", "lasttoken"],
            "speech_mode": args.speech,
            "speech_input": ("mHuBERT unit ids as token ids; audio tower never runs"
                             if args.speech == "units" else
                             "waveform through the audio tower"),
            "unit_parquet": (os.path.abspath(args.unit_parquet)
                             if args.speech == "units" else None),
            "unit_contract": ("<speech_start> dedup_units <speech_end>, pooled over units "
                              "only" if args.speech == "units"
                              and args.unit_representation == "pooled_units" else
                              "<speech_start> dedup_units <speech_end> "
                              "<transcript_start>, boundary state only"
                              if args.speech == "units" else None),
            "unit_representation": (
                args.unit_representation if args.speech == "units" else None
            ),
            "unit_context_limit": (
                1024 if args.speech == "units"
                and args.unit_representation == "asr_boundary" else None
            ),
            "unit_token_start": BASE_VOCAB if args.speech == "units" else None,
            "audio_markers": None if args.speech == "units" else AUDIO_MARKERS,
            "target_sample_rate_hz": None if args.speech == "units" else TARGET_SR,
            "generation": "none - forward passes only",
            "attention": ("eager" if path == "base" else
                          "eager audio tower; decoder uses checkpoint configuration"),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "dtype": str(dtype),
            "device": device,
        })

        del thinker
        if path != "base":
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
