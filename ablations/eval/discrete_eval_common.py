#!/usr/bin/env python3
"""Shared pieces for the unit-in, zero-shot, no-post-processing text evals.

Speech reaches the model as mHuBERT unit ids offset into the expanded vocabulary and
wrapped in the sentinels the discrete-ASR runs were trained on:

    <|speech_start|> units <|speech_end|> <|transcript_start|> text <|im_end|>

Nothing here rewrites, truncates or cleans a hypothesis. The only transformation applied
to a decode is dropping tokens at or after the trained EOS, which is where the model
said it stopped -- not an edit to what it said. Everything else is reported raw so the
smoke gate (0 empty, 0 cap-hits, 0 repetition loops, EOS on every item) is checkable
from the output file alone.
"""
import os
import math

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

BASE_VOCAB = 151936
UNIT_COUNT = 1000
SPEECH_START = BASE_VOCAB + UNIT_COUNT      # 152936
SPEECH_END = SPEECH_START + 1               # 152937
TRANSCRIPT_START = SPEECH_END + 1           # 152938
EXPANDED_VOCAB = TRANSCRIPT_START + 1       # 152939

# Literal spellings, for models without the expanded vocabulary. Base's embedding
# stops at 151936, so the three sentinels have no ids it can consume; it takes the
# same prompt string through its own BPE instead, with the tower's audio markers
# occupying the slot the unit block occupies. TRANSCRIPT_START has no native Qwen
# equivalent, so it becomes a plain cue rather than an unrepresentable token.
AUDIO_MARKERS = "<|audio_bos|><|AUDIO|><|audio_eos|>"
SPEECH_START_TEXT = "<|speech_start|>"
SPEECH_END_TEXT = "<|speech_end|>"
TRANSCRIPT_START_TEXT = "TRANSCRIPT:"

# Sentinels plus the trailing answer cue; the slack a prompt must leave for generation.
SENTINEL_TOKENS = 3


def reference_token_budget(tokenizer, reference, multiplier=1.25, minimum=8,
                           hard_max=None):
    """Generation ceiling derived from the known reference length.

    The extra token is reserved for a structural EOS. ``hard_max`` is only a
    safety ceiling; pass 0/None for no additional fixed cap.
    """
    reference_tokens = len(tokenizer.encode(
        reference or "", add_special_tokens=False))
    budget = max(int(minimum), math.ceil(reference_tokens * multiplier) + 1)
    if hard_max:
        budget = min(budget, int(hard_max))
    return reference_tokens, budget



def base_snapshot(model_id="Qwen/Qwen2.5-Omni-3B"):
    """Locate the local base snapshot, which is where the tokenizer lives.

    A training checkpoint directory holds config.yaml + pytorch_model.bin only -- it is
    not an HF model directory, so AutoTokenizer cannot read it. Unit ids sit above the
    tokenizer's vocabulary and are handled by id, so the unmodified base tokenizer is the
    right one.
    """
    import glob
    if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id
    pattern = os.path.join(os.environ["HF_HOME"], "hub",
                           "models--" + model_id.replace("/", "--"), "snapshots", "*")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise RuntimeError(f"no local snapshot for {model_id}")
    return hits[0]


def speech_span(units):
    """Unit ids (0-999) -> <|speech_start|> offset-units <|speech_end|>."""
    units = np.asarray(units, dtype=np.int64)
    if len(units) and (units.min() < 0 or units.max() >= UNIT_COUNT):
        raise ValueError(f"unit ids outside [0, {UNIT_COUNT}): "
                         f"[{units.min()}, {units.max()}]")
    return ([SPEECH_START] + (units + BASE_VOCAB).tolist() + [SPEECH_END])


def chunk_units(units, chunk_size):
    """Split one clip's unit sequence into <= chunk_size windows, in order.

    CLUAS clips are 40-93 s, which is 1,500-3,400 units after dedup -- past the 1024
    context every ablation trains at. Windowing is done on units rather than on the
    waveform so the clip is tokenized once and the window size stays a decode-time knob.
    """
    units = np.asarray(units, dtype=np.int64)
    if chunk_size <= 0 or len(units) <= chunk_size:
        return [units]
    return [units[i:i + chunk_size] for i in range(0, len(units), chunk_size)]


def units_budget(seq_len, max_new_tokens, question_tokens):
    """Largest unit window that still leaves room for the question and the answer."""
    return seq_len - max_new_tokens - question_tokens - SENTINEL_TOKENS


def repetition_report(ids, n=5, threshold=3):
    """Flag a decode as looping without altering it.

    Two independent signals: an n-gram of token ids emitted `threshold`+ times, or a
    long decode made of very few distinct tokens. Reported, never acted on.
    """
    ids = list(ids)
    counts = {}
    for i in range(len(ids) - n + 1):
        gram = tuple(ids[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    top = max(counts.values()) if counts else 0
    unique_ratio = len(set(ids)) / max(len(ids), 1)
    return {
        "max_ngram_repeat": top,
        "unique_token_ratio": unique_ratio,
        "looping": bool(top >= threshold or (len(ids) > 20 and unique_ratio < 0.3)),
    }


class StopOnSubstrings(StoppingCriteria):
    """Halts generation once the decoded suffix contains one of the given strings.

    A decoding-time choice, the same kind as choosing eos_token_id or wording the prompt:
    it tells the model where to stop, it does not touch what the model already said.
    Different from post-processing, which edits a decode that has already finished.
    Batch size 1 only, which is what every eval here uses.
    """

    def __init__(self, tokenizer, prompt_len, stop_strings):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stop_strings = list(stop_strings)
        self.triggered = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.triggered is not None:
            return True
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        for s in self.stop_strings:
            if s in text:
                self.triggered = s
                return True
        return False


@torch.no_grad()
def _normalise_eos(eos_id):
    """Return a non-empty list while keeping old scalar callers compatible."""
    values = eos_id if isinstance(eos_id, (list, tuple, set)) else [eos_id]
    values = [int(value) for value in values]
    if not values:
        raise ValueError("at least one EOS id is required")
    return list(dict.fromkeys(values))


def _token_name(tokenizer, token_id):
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return token if token is not None else f"<id:{token_id}>"


def decode(thinker, tokenizer, prompt_ids, eos_id, max_new_tokens, device="cuda",
          stop_strings=None):
    """Greedy decode of one prompt. Returns the raw decode and its stop diagnostics."""
    eos_ids = _normalise_eos(eos_id)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    criteria = None
    if stop_strings:
        criteria = StoppingCriteriaList(
            [StopOnSubstrings(tokenizer, prompt.shape[1], stop_strings)])
    generated = thinker.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=eos_ids[0],
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=criteria,
    )
    out = generated.sequences[0, prompt.shape[1]:].tolist()
    return _decode_report(tokenizer, out, eos_ids, len(prompt_ids),
                          scores=generated.scores, stop_strings=stop_strings)


def _mean_logprob(scores, ids):
    """Mean log-probability the model assigned to the tokens it actually emitted.

    Used to choose between the answers different audio windows produce. It ranks whole
    decodes against each other and never alters one, so it stays inside the
    no-post-processing rule. Length-normalised, otherwise short answers always win.
    """
    if not scores or not ids:
        return None
    total = 0.0
    for step, token in enumerate(ids):
        if step >= len(scores):
            break
        logits = scores[step][0].float()
        total += float(torch.log_softmax(logits, dim=-1)[token])
    return total / min(len(ids), len(scores))


def _decode_report(tokenizer, out, eos_id, prompt_len, scores=None, stop_strings=None):
    eos_ids = _normalise_eos(eos_id)
    eos_hits = [(out.index(value), value) for value in eos_ids if value in out]
    eos_index, emitted_eos_id = (min(eos_hits) if eos_hits else (None, None))
    eos_emitted = emitted_eos_id is not None
    answer_ids = out[:eos_index] if eos_emitted else out
    # Unit ids have no text form; keep them visible as a count rather than dropping
    # them silently, because a text eval emitting speech units is a real finding.
    unit_ids = [t for t in answer_ids if t >= len(tokenizer)]
    text_ids = [t for t in answer_ids if t < len(tokenizer)]
    # The accepted terminal EOS has already been removed by token position. Keep any
    # other generated special token visible; silently cleaning it would repair output.
    text = tokenizer.decode(text_ids, skip_special_tokens=False)
    # Preserve a human-readable decode before any structural-boundary removal.
    # Expanded unit ids have no tokenizer string, so their exact values remain in
    # raw_output_ids and are counted separately.
    raw_known_ids = [t for t in out if t < len(tokenizer)]
    raw_text = tokenizer.decode(raw_known_ids, skip_special_tokens=False)
    stop_reason = "eos" if eos_emitted else "max_new_tokens"
    matched_stop_string = None
    if not eos_emitted and stop_strings:
        # Cut at the earliest boundary, the same treatment as the eos_id cut above --
        # the point where the model itself moved on, not an edit to what it said.
        hits = [(text.find(s), s) for s in stop_strings if s in text]
        if hits:
            idx, matched_stop_string = min(hits, key=lambda h: h[0])
            text = text[:idx]
            stop_reason = "stop_string"
    return {
        "hypothesis": text,
        "raw_output_ids": out,
        "raw_output_text": raw_text,
        "answer_ids": answer_ids,
        "generated_tokens": len(out),
        "eos_emitted": eos_emitted,
        "accepted_eos_ids": eos_ids,
        "emitted_stop_id": emitted_eos_id,
        "emitted_stop_token": (
            _token_name(tokenizer, emitted_eos_id)
            if emitted_eos_id is not None else None
        ),
        "stop_reason": stop_reason,
        "matched_stop_string": matched_stop_string,
        "unit_ids_in_answer": len(unit_ids),
        # The base_raw counterpart of unit_ids_in_answer. A model with the expanded
        # vocabulary answers in the wrong modality by emitting unit ids; base, which
        # sees the sentinels as ordinary BPE subwords ("<","|","speech","_start",...),
        # does it by writing the delimiters back out as text. Same failure, different
        # spelling, so it has to be counted or one arm looks cleaner than it is.
        # Recorded, never stripped -- the contract is no post-processing.
        "sentinel_text_in_answer": sum(
            text.count(marker)
            for marker in (SPEECH_START_TEXT, SPEECH_END_TEXT,
                           TRANSCRIPT_START_TEXT)
        ),
        "prompt_tokens": prompt_len,
        "mean_logprob": _mean_logprob(scores, answer_ids),
        **repetition_report(answer_ids),
    }


@torch.no_grad()
def decode_prepared(thinker, tokenizer, inputs, eos_id, max_new_tokens, stop_strings=None):
    """Same greedy decode and same diagnostics, for inputs the processor built.

    Used when speech goes through the Qwen audio tower instead of arriving as units:
    `inputs` then carries input_features alongside input_ids, so the prompt cannot be a
    plain id list. Reporting is identical so tower and unit runs are directly comparable.
    """
    eos_ids = _normalise_eos(eos_id)
    prompt_len = inputs["input_ids"].shape[1]
    criteria = None
    if stop_strings:
        criteria = StoppingCriteriaList(
            [StopOnSubstrings(tokenizer, prompt_len, stop_strings)])
    generated = thinker.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=eos_ids[0],
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=criteria,
    )
    out = generated.sequences[0, prompt_len:].tolist()
    return _decode_report(tokenizer, out, eos_ids, prompt_len,
                          scores=generated.scores, stop_strings=stop_strings)


def smoke_gate(rows):
    """The four conditions from handoff 2. Counts only -- the user reads the outputs."""
    empty = sum(1 for r in rows if not r["hypothesis"].strip())
    cap = sum(1 for r in rows if r["stop_reason"] == "max_new_tokens")
    natural_eos = sum(1 for r in rows if r["stop_reason"] == "eos")
    backup_stop = sum(1 for r in rows if r["stop_reason"] == "stop_string")
    accepted_stop = natural_eos + backup_stop
    looping = sum(1 for r in rows if r["looping"])
    no_eos = sum(1 for r in rows if not r["eos_emitted"])
    return {
        "items": len(rows),
        "empty_outputs": empty,
        "hit_token_cap": cap,
        "repetition_loops": looping,
        "accepted_stop": accepted_stop,
        "natural_eos": natural_eos,
        "backup_stop_string": backup_stop,
        "missing_eos": no_eos,
        # A declared backup marker is an accepted decoding boundary, but is
        # deliberately counted separately from a model-emitted trained EOS.
        "passed": empty == 0 and cap == 0 and looping == 0
                  and accepted_stop == len(rows),
    }


def print_smoke(name, gate, rows, limit=10):
    print("\n" + "=" * 72)
    print(f"SMOKE — {name}")
    print("=" * 72)
    for key, value in gate.items():
        print(f"  {key:20s} {value}")
    print("\n" + "-" * 72)
    print("RAW OUTPUTS (unedited)")
    print("-" * 72)
    for row in rows[:limit]:
        head = row.get("item", "")
        print(f"\n[{head}] stop={row['stop_reason']} tokens={row['generated_tokens']} "
              f"prompt={row['prompt_tokens']}")
        if row.get("emitted_stop_token"):
            print(f"  emitted_stop={row['emitted_stop_token']!r} "
                  f"(id={row['emitted_stop_id']})")
        elif row.get("matched_stop_string"):
            print(f"  backup_stop={row['matched_stop_string']!r}")
        print(f"  {row['hypothesis']!r}")
