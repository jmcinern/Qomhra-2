"""Single-GCD, no-FSDP reproduction of the speech NaN — the fast debug loop.

The full rig needs 8 GCDs, an salloc and ~4 min to tell us anything. This does the
same forward/backward/update on one GCD in ~90 s, with the instrumentation the real
loop can't afford, so lr/dtype/loss hypotheses can be bisected in minutes.

It answers two questions that the 8-GCD run conflates:

  --mode forward   Freeze everything, run N batches, no optimizer. If a NaN shows up
                   here it is DATA-dependent, not an lr blowup — the prime candidate
                   being a fully-masked attention row: the tower runs eager attention
                   (ROCm SDPA NaNs it, see model.py) and eager builds a float -inf
                   additive mask, so a row with no valid keys softmaxes to NaN.

  --mode train     Real optimizer, real lr, per-step grad norms split by submodule
                   (audio_tower / decoder / audio_head) plus the tower's param absmax.
                   If the tower's grad norm spikes a step or two before the loss goes
                   NaN, it is the update destroying the encoder — bisect lr from there.

Both modes stop at the first non-finite and print a stage-by-stage breakdown (mel ->
tower out -> decoder hidden -> pred -> loss) so the NaN's origin is named, not guessed.

Usage (inside an allocation; see probe.sh):
    python _probe_steps.py --mode train --steps 30 optim.lr=3e-5
Any trailing args are Hydra overrides on the legacy/omni_speech config.
"""
import argparse
import os
import sys

import torch
from hydra import compose, initialize_config_dir

from qomhra.data import get_dataloader
from qomhra.model import apply_freezing, get_model

HERE = os.path.dirname(os.path.abspath(__file__))


def finite(t):
    return bool(torch.isfinite(t).all())


def describe(name, t):
    t = t.float()
    bad = (~torch.isfinite(t)).sum().item()
    return (f"{name:<16} shape={tuple(t.shape)} finite={t.numel()-bad}/{t.numel()} "
            f"absmax={t.abs()[torch.isfinite(t)].max().item() if bad < t.numel() else float('nan'):.4g}")


def stage_breakdown(model, batch):
    """Re-run the audio path stage by stage and name the first stage that goes bad."""
    m = model.thinker
    feats, lens = batch["input_features"], batch["feature_lens"]
    print("  " + describe("mel", feats))
    with torch.no_grad():
        embeds, frame_mask = model._encode_audio(m.audio_tower, feats, lens)
        print("  " + describe("tower_out", embeds))
        print(f"  frame_mask       valid rows per item: {frame_mask.sum(1).tolist()}")
        if not finite(embeds):
            print("  --> FIRST BAD STAGE: audio_tower output")
            return
        hidden = m.model(inputs_embeds=embeds, attention_mask=frame_mask.long(),
                         use_cache=False).last_hidden_state
        print("  " + describe("decoder_hidden", hidden))
        if not finite(hidden):
            print("  --> FIRST BAD STAGE: text decoder")
            return
        pred = model.audio_head(hidden)
        print("  " + describe("pred", pred))
        print("  --> FIRST BAD STAGE: audio_head / loss")


def grad_norms(model):
    """Grad norm per submodule group — where the explosion lives."""
    groups = {
        "audio_tower": model.thinker.audio_tower,
        "decoder": model.thinker.model,
        "audio_head": model.audio_head,
    }
    out = {}
    for name, mod in groups.items():
        sq, nonfinite = 0.0, 0
        for p in mod.parameters():
            if p.grad is None:
                continue
            g = p.grad.float()
            if not torch.isfinite(g).all():
                nonfinite += 1
            sq += float(g[torch.isfinite(g)].pow(2).sum())
        out[name] = (sq ** 0.5, nonfinite)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["forward", "backward", "lens", "anomaly", "train"], default="train")
    ap.add_argument("--steps", type=int, default=30)
    # All 4.04B params with fp32 Adam state is ~64 GB — it OOMs a single GCD. `tower`
    # trains only the audio tower + head (~0.65B, fits easily). The decoder's params
    # are frozen but it still runs forward/backward, so gradient reaches the tower
    # through it exactly as in the real run — and the decoder is not a suspect anyway
    # (the text ablation is stable at this lr). This isolates the hypothesis: is the
    # tower's own update destroying it?
    ap.add_argument("--trainable", choices=["all", "tower"], default="tower")
    args, overrides = ap.parse_known_args()

    with initialize_config_dir(config_dir=os.path.join(HERE, "qomhra", "configs"),
                               version_base=None):
        cfg = compose(config_name="legacy/omni_speech", overrides=overrides)

    # No FSDP, no accelerate. Params stay fp32 and compute runs under bf16 autocast —
    # the same arithmetic FSDP's MixedPrecision(param_dtype=bf16) gives each rank,
    # minus the sharding and the grad all-reduce. That is exactly the point: if the
    # NaN reproduces here, FSDP is not the culprit.
    dev = torch.device("cuda:0")
    torch.manual_seed(cfg.seed)

    model, _ = get_model(cfg)
    print(apply_freezing(model, cfg), flush=True)
    if args.trainable == "tower":
        for p in model.thinker.model.parameters():
            p.requires_grad_(False)
        model.thinker.lm_head.requires_grad_(False)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[probe] decoder params frozen; training tower+head only ({n/1e9:.2f}B)")
    model.to(dev).train()

    loader = get_dataloader(cfg, vocab_size=None)["audio"]
    it = iter(loader)

    if args.mode == "forward":
        # Frozen: any NaN here is the data/mask, not the update.
        print(f"[probe] forward-only sweep, {args.steps} batches, no optimizer\n", flush=True)
        for step in range(1, args.steps + 1):
            batch = {k: v.to(dev) for k, v in next(it).items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            loss, var = float(out["loss_audio"]), float(out["audio_target_var"])
            lens = batch["feature_lens"].tolist()
            ok = loss == loss and var == var
            print(f"step {step:>3} loss {loss:.4f} var {var:.4f} "
                  f"frames {out['n_audio_frames']:>5} mel_lens {lens}"
                  f"{'' if ok else '   <-- NON-FINITE'}", flush=True)
            if not ok:
                print("\n[probe] stage breakdown on the offending batch:")
                stage_breakdown(model, batch)
                sys.exit(1)
        print("\n[probe] forward sweep CLEAN — the NaN needs the optimizer.")
        return

    if args.mode == "anomaly":
        # Run the REAL model on the REAL batches with autograd anomaly detection on.
        # When a backward produces the first NaN, torch raises and names the forward
        # op that created the offending grad_fn — which beats any amount of guessing
        # about where in tower/decoder/loss it lives. Slow, so only used on the one
        # batch that fails.
        torch.autograd.set_detect_anomaly(True)
        print(f"[probe] anomaly detection over {args.steps} batches\n", flush=True)
        for step in range(1, args.steps + 1):
            batch = {k: v.to(dev) for k, v in next(it).items()}
            model.zero_grad(set_to_none=True)
            print(f"step {step:>3} mel_lens {batch['feature_lens'].tolist()}", flush=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            out["loss"].backward()
        return

    if args.mode == "lens":
        # Sweep single-clip mel lengths through the tower and report, per length,
        # whether the FORWARD has NaNs (and where) and whether the BACKWARD does.
        # A forward that is finite where we look but NaN in the padding rows is the
        # signature of a fully-masked attention row: our frame_mask drops those rows
        # from the loss, so the loss is finite, but the backward still multiplies the
        # NaN softmax by a zero grad — and 0 * NaN = NaN, which is why only the
        # gradients blow up.
        tower = model.thinker.audio_tower
        print("[probe] per-batch forward/backward check on the audio tower\n", flush=True)
        for combo in [[3000], [1997], [3000, 3000], [3000, 2664], [2664, 3000],
                      [1997, 3000], [3000, 1997], [2000, 3000], [1000, 3000],
                      [2900, 3000]]:
            L = max(combo)
            feats = torch.zeros(len(combo), 128, L, device=dev)
            for i, c in enumerate(combo):
                feats[i, :, :c] = torch.randn(128, c, device=dev) * 0.5
            lens = torch.tensor(combo, device=dev)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeds, fmask = model._encode_audio(tower, feats, lens)
                nan_out = int((~torch.isfinite(embeds)).sum())
                loss = embeds[fmask].float().pow(2).mean()
            loss.backward()
            nf = sum(1 for p in tower.parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all())
            print(f"  mel_lens {str(combo):<14} -> frames {fmask.sum(1).tolist()} "
                  f"| fwd NaNs {nan_out:>7} | bwd nonfinite grads {nf:>3}"
                  f"{'   <-- BAD' if nf else ''}", flush=True)
        return

    if args.mode == "backward":
        # Forward + backward on a PRISTINE model, no optimizer, grads zeroed each
        # batch. The weights never change, so any non-finite grad here is caused by
        # the batch alone. Ragged batches (unequal feature_lens) are the suspects.
        print(f"[probe] backward-only sweep, {args.steps} batches, weights frozen in "
              f"time (no optimizer)\n", flush=True)
        bad = 0
        for step in range(1, args.steps + 1):
            batch = {k: v.to(dev) for k, v in next(it).items()}
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            out["loss"].backward()
            gn = grad_norms(model)
            lens = batch["feature_lens"].tolist()
            ragged = len(set(lens)) > 1
            nf = gn["audio_tower"][1]
            print(f"step {step:>3} loss {float(out['loss_audio']):.4f} "
                  f"mel_lens {str(lens):<14} {'RAGGED' if ragged else 'square':<7} "
                  f"tower_grad {gn['audio_tower'][0]:.3g} nonfinite_params={nf}"
                  f"{'   <-- NaN GRAD' if nf else ''}", flush=True)
            bad += bool(nf)
        print(f"\n[probe] {bad}/{args.steps} batches produced non-finite tower grads.")
        return

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.optim.lr, betas=tuple(cfg.optim.betas),
                            weight_decay=cfg.optim.weight_decay)
    print(f"[probe] train loop, {args.steps} steps, lr={cfg.optim.lr}, "
          f"clip={cfg.optim.grad_clip}\n", flush=True)

    for step in range(1, args.steps + 1):
        batch = {k: v.to(dev) for k, v in next(it).items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        out["loss"].backward()

        gn = grad_norms(model)
        pre_clip = torch.nn.utils.clip_grad_norm_(trainable, cfg.optim.grad_clip)
        opt.step()
        model.update_ema()      # no-op unless target_detach=ema; mirrors train_utils

        loss, var = float(out["loss_audio"]), float(out["audio_target_var"])
        tower_absmax = max(float(p.detach().abs().max())
                           for p in model.thinker.audio_tower.parameters())
        ok = loss == loss and var == var
        print(
            f"step {step:>3} loss {loss:.4f} var {var:.4f} | grad tower "
            f"{gn['audio_tower'][0]:.3g}(nf={gn['audio_tower'][1]}) dec "
            f"{gn['decoder'][0]:.3g}(nf={gn['decoder'][1]}) head "
            f"{gn['audio_head'][0]:.3g} | total {float(pre_clip):.3g} | "
            f"tower|w|max {tower_absmax:.3g}"
            f"{'' if ok else '   <-- NON-FINITE'}", flush=True)
        if not ok:
            print("\n[probe] stage breakdown on the offending batch:")
            stage_breakdown(model, batch)
            sys.exit(1)

    print("\n[probe] train loop CLEAN at this lr — raise --steps or change the lever.")


if __name__ == "__main__":
    main()
