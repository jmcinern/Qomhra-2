#!/usr/bin/env python3
"""Shape/semantics probe for the discrete-target audio objective (task.audio_target=units).

CPU only, no checkpoint, no mHuBERT cache — it exercises the parts of the units path
that can be wrong independently of the real data:

  1. encoder_frames() matches Qwen2_5OmniAudioEncoder._get_feat_extract_output_lengths
  2. collate_audio pads unit_labels to the frame max with -100
  3. the CE loss, unit_acc, unit_repeat_acc and unit_pred_entropy are computed on the
     right positions, with the t+n shift applied in the right direction
  4. a one-frame label misalignment RAISES rather than training on a shifted task

Run:  python ablations/train/_probe_units.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

# data.py imports torchaudio at module scope for wav decoding. Nothing here decodes
# audio, so stand in a placeholder rather than require the LUMI env on a laptop.
for _dep in ("torchaudio",):
    try:
        __import__(_dep)
    except ImportError:
        import importlib.machinery
        import types

        _mod = types.ModuleType(_dep)
        # transformers probes optional deps with find_spec, which rejects __spec__=None.
        _mod.__spec__ = importlib.machinery.ModuleSpec(_dep, None)
        sys.modules[_dep] = _mod

from qomhra.data import collate_audio, encoder_frames, _stub_load_units
from qomhra.model import OmniThinkerCPT, _marginal_entropy

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def stub_model(horizon=1, n_units=1000):
    """An OmniThinkerCPT with only the fields the loss helpers touch — building the
    real one would download and load 3B params."""
    m = object.__new__(OmniThinkerCPT)
    m.audio_target = "units"
    m.audio_horizon = horizon
    m.n_units = n_units
    return m


# 1 ---------------------------------------------------------------------------
print("\n[1] encoder_frames parity with transformers")
try:
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
    )

    lens = torch.tensor([1, 2, 3, 7, 100, 999, 3000, 3001])
    _, want = Qwen2_5OmniAudioEncoder._get_feat_extract_output_lengths(None, lens)
    got = torch.tensor([encoder_frames(int(x)) for x in lens])
    check("matches _get_feat_extract_output_lengths", bool((got == want).all()),
          f"{got.tolist()} vs {want.tolist()}")
except ImportError as exc:
    print(f"  SKIP  transformers unavailable ({exc})")

# 30 s at 100 Hz mel -> the ~750 frames the docstrings quote
check("30 s chunk lands near 750 frames", 740 <= encoder_frames(3000) <= 760,
      f"{encoder_frames(3000)} frames")

# 2 ---------------------------------------------------------------------------
print("\n[2] collate_audio with unit labels")
items = []
for flen in (3000, 1200):
    n = encoder_frames(flen)
    items.append({
        "input_features": torch.randn(128, flen),
        "feature_lens": flen,
        "unit_labels": torch.as_tensor(
            np.asarray(_stub_load_units("clip", 0, 30.0, n)), dtype=torch.long
        ),
    })
batch = collate_audio(items)
n_max = encoder_frames(3000)
check("unit_labels shape is (B, F_max)", tuple(batch["unit_labels"].shape) == (2, n_max),
      str(tuple(batch["unit_labels"].shape)))
check("short item is -100 padded",
      bool((batch["unit_labels"][1, encoder_frames(1200):] == -100).all()))
check("valid region carries real labels",
      bool((batch["unit_labels"][1, : encoder_frames(1200)] >= 0).all()))
check("no unit_labels key when the loader is off",
      "unit_labels" not in collate_audio(
          [{k: v for k, v in it.items() if k != "unit_labels"} for it in items]))

# 3 ---------------------------------------------------------------------------
print("\n[3] loss / metrics on a hand-built batch")
# Two items, 6 frames each; item 1 has only 4 real frames.
units = torch.tensor([
    [5, 5, 7, 7, 9, 9],
    [1, 2, 3, 4, -100, -100],
])
frame_mask = torch.tensor([
    [True] * 6,
    [True] * 4 + [False] * 2,
])
m = stub_model(horizon=1)
# A head that predicts the CURRENT unit every time — i.e. exactly the repeat baseline.
logits = torch.zeros(2, 6, m.n_units)
for b in range(2):
    for t in range(6):
        if units[b, t] >= 0:
            logits[b, t, units[b, t]] = 20.0
n = m.audio_horizon
valid = frame_mask[:, :-n] & frame_mask[:, n:]
out = m._audio_loss_units(logits[:, :-n], valid, units, frame_mask)

# item0: 5 pairs, of which (5->5) and (7->7) and (9 is last, dropped) repeat -> 5,7,9 at
# t=0,2,4 repeat into t=1,3,5 => 3 of 5.  item1: 3 valid pairs, none repeat.
check("unit_repeat_acc == copy-baseline accuracy",
      abs(float(out["unit_repeat_acc"]) - 3 / 8) < 1e-6,
      f"{float(out['unit_repeat_acc']):.4f} (expected 0.3750)")
check("a copy head scores exactly the repeat baseline",
      abs(float(out["unit_acc"]) - float(out["unit_repeat_acc"])) < 1e-6,
      f"unit_acc={float(out['unit_acc']):.4f}")
check("padded positions excluded", int(out["n_audio_frames"]) == 8,
      f"{int(out['n_audio_frames'])} (expected 8 = 5 + 3)")

# A uniform head should sit at ln(n_units).
flat = torch.zeros(2, 6, m.n_units)
out_flat = m._audio_loss_units(flat[:, :-n], valid, units, frame_mask)
check("uniform logits give loss = ln(1000)",
      abs(float(out_flat["loss"]) - np.log(1000)) < 1e-3,
      f"{float(out_flat['loss']):.4f} vs {np.log(1000):.4f}")

# Direction of the shift: a head that predicts the NEXT unit should be perfect.
oracle = torch.zeros(2, 6, m.n_units)
for b in range(2):
    for t in range(5):
        if units[b, t + 1] >= 0:
            oracle[b, t, units[b, t + 1]] = 20.0
out_oracle = m._audio_loss_units(oracle[:, :-n], valid, units, frame_mask)
check("oracle next-unit head scores 1.0 (shift direction correct)",
      abs(float(out_oracle["unit_acc"]) - 1.0) < 1e-6,
      f"{float(out_oracle['unit_acc']):.4f}")

# Collapse canary.
check("entropy of a single-class prediction is 0",
      abs(float(_marginal_entropy(torch.zeros(50, dtype=torch.long), 1000))) < 1e-6)
check("entropy of a uniform prediction approaches ln(k)",
      abs(float(_marginal_entropy(torch.arange(1000), 1000)) - np.log(1000)) < 1e-3)

# Horizon 2 drops one more pair per item.
m2 = stub_model(horizon=2)
valid2 = frame_mask[:, :-2] & frame_mask[:, 2:]
out2 = m2._audio_loss_units(logits[:, :-2], valid2, units, frame_mask)
check("horizon=2 keeps 4 + 2 pairs", int(out2["n_audio_frames"]) == 6,
      f"{int(out2['n_audio_frames'])}")

# 4 ---------------------------------------------------------------------------
print("\n[4] misalignment is loud, not silent")
try:
    m._audio_loss_units(logits[:, :-n], valid, units[:, :-1], frame_mask)
    check("off-by-one label length raises", False, "it did NOT raise")
except RuntimeError as exc:
    check("off-by-one label length raises", "!=" in str(exc), str(exc)[:70])

try:
    m._audio_loss_units(logits[:, :-n], valid, None, frame_mask)
    check("missing unit_labels raises", False, "it did NOT raise")
except RuntimeError as exc:
    check("missing unit_labels raises", "unit_labels" in str(exc), str(exc)[:70])

# -----------------------------------------------------------------------------
print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
