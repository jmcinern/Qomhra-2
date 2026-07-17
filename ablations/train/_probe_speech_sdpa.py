"""Can the speech branch drop eager now that KV heads match?

The audio + aligned branches pin eager in two places (model.py):
  1. _force_eager_attention(thinker.audio_tower) — the tower forward went all-NaN
     under sdpa (measured: 3,072,000/3,072,000 elements).
  2. _eager_attention(thinker.model) in _forward_audio/_forward_aligned — the DECODER
     backward NaN'd whenever a padding mask was present.

Both were diagnosed before we knew GQA was silently forcing the math/broadcast path on
every call. A synthetic probe shows EFFICIENT+repeat_kv+masked is clean (fwd finite,
grads finite), so the recorded cause may be incomplete. But synthetic tensors use a bool
mask; the real tower builds a float -inf additive mask, where a fully-masked row makes
softmax NaN on its own. Only the real model settles it.

Runs the REAL ragged audio batch through both branches, eager vs sdpa, and checks the
BACKWARD — the forward was never the broken part.
"""
import contextlib

import torch
from hydra import compose, initialize

import qomhra.model as M

# A deliberately RAGGED batch: unequal clip lengths -> padding mask. Equal-length
# batches never triggered the NaN, which is why it hid for so long.
LENS = [750, 375]


def build_batch(device):
    n_mel, hop = 128, 2   # mel frames per encoder frame
    feats = [torch.randn(n_mel, L * hop, device=device) for L in LENS]
    T = max(f.shape[1] for f in feats)
    mel = torch.zeros(len(feats), n_mel, T, device=device)
    for i, f in enumerate(feats):
        mel[i, :, :f.shape[1]] = f
    lens = torch.tensor([f.shape[1] for f in feats], device=device)
    return mel.to(torch.bfloat16), lens


def check(model, mel, lens, use_sdpa, tower_sdpa):
    real_ctx = M._eager_attention
    if use_sdpa:
        M._eager_attention = lambda module: contextlib.nullcontext()
    if tower_sdpa:
        for m in model.thinker.audio_tower.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None:
                cfg._attn_implementation = "sdpa"
    try:
        model.zero_grad(set_to_none=True)
        out = model(input_features=mel, feature_lens=lens)
        loss = out["loss"]
        fwd_ok = torch.isfinite(loss).all().item()
        loss.backward()
        torch.cuda.synchronize()
        tower = [p for n, p in model.thinker.audio_tower.named_parameters()
                 if p.grad is not None]
        nan = sum(1 for p in tower if not torch.isfinite(p.grad).all())
        return (f"loss={loss.item():.4f} fwd={'finite' if fwd_ok else 'NaN'} "
                f"tower_grads: {nan}/{len(tower)} NaN")
    except Exception as e:
        return f"FAIL {type(e).__name__}: {str(e)[:70]}"
    finally:
        M._eager_attention = real_ctx
        M._force_eager_attention(model.thinker.audio_tower)


def main():
    with initialize(config_path="qomhra/configs", version_base="1.1"):
        args = compose(config_name="omni_speech")

    model, _ = M.get_model(args)
    model = model.cuda().to(torch.bfloat16)
    mel, lens = build_batch("cuda")
    print(f"ragged batch: encoder frames {LENS} (mel {tuple(mel.shape)})\n")

    for tower_sdpa in (False, True):
        for dec_sdpa in (False, True):
            tag = (f"tower={'sdpa ' if tower_sdpa else 'eager'} "
                   f"decoder={'sdpa ' if dec_sdpa else 'eager'}")
            print(f"{tag} -> {check(model, mel, lens, dec_sdpa, tower_sdpa)}", flush=True)


if __name__ == "__main__":
    main()
