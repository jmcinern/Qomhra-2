"""Model construction for the ablation CPT runs.

We continued-pretrain **Qwen2.5-Omni-3B's Thinker** (no Talker — that is
post-finetuning scope). Two modalities, one backbone:

  text   ids --embed_tokens--> decoder --lm_head--> shifted CE          (NTP on text)
  audio  wav --mel--> audio_tower --> 25 Hz x 2048-d embeddings
                 --(as inputs_embeds)--> decoder --audio_head--> next-frame
                 regression against the *detached* encoder embeddings   (NTP on audio hidden)

Speech goes through Omni's **native audio path**, not mHuBERT discrete units, so
the encoder + adapter are exercised and trained. The audio tower's output width
(2048) equals the thinker's hidden size, so its embeddings drop straight into the
decoder's input slots — one audio frame occupies one context slot, exactly like a
text token.

The audio target is the next frame's encoder embedding with a **stop-grad**. The
encoder is trainable in the speech/both ablations, so without the detach the
model could minimise the loss by collapsing the encoder's output distribution
rather than by predicting anything. `target_detach: ema` swaps in a frozen EMA
copy of the encoder for the targets if plain detach still drifts.

No vocab resize: the Qwen2.5 tokenizer's native ids are used as-is (the old
shared-vocab mHuBERT scheme grew the embedding table; the native audio path does
not need any new rows).
"""
import contextlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_thinker(m):
    """Load ONLY the Thinker out of the Qwen2.5-Omni checkpoint.

    We deliberately do not go through Qwen2_5OmniForConditionalGeneration: its
    `from_pretrained` also builds the Talker + Token2Wav and `torch.load`s the TTS
    speaker dict, which transformers refuses on the container's torch 2.5 (the
    CVE-2025-32434 guard demands >= 2.6). None of that is used here — we train the
    Thinker only. The Thinker class declares `base_model_prefix = "thinker"`, so it
    maps the checkpoint's `thinker.*` keys itself; we just have to hand it the
    nested thinker_config rather than the full Omni config.
    """
    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    cfg = AutoConfig.from_pretrained(m.base_model_id)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        m.base_model_id,
        config=cfg.thinker_config,
        attn_implementation=m.attn_implementation,
    )

    # The Thinker also carries a vision tower. These ablations are text+speech only,
    # so it would never receive input — but under `freeze: all` it would still take
    # optimizer state and get sharded/all-gathered every step. Drop it (~0.7B params).
    if getattr(thinker, "visual", None) is not None:
        thinker.visual = None

    # The audio tower returns all-NaN under SDPA on ROCm (verified: every one of
    # 3,072,000 output elements NaN with sdpa, clean under eager, in both fp32 and
    # bf16, with correctly-loaded weights). Its attention builds a float -inf
    # additive mask, which the ROCm SDPA kernel turns into NaN. The text decoder is
    # unaffected, so pin eager on the tower only and leave the decoder on SDPA.
    _force_eager_attention(thinker.audio_tower)

    return thinker


def _force_eager_attention(module):
    """Set `_attn_implementation = "eager"` on a submodule tree's configs.

    transformers dispatches attention per-module off its own config, so overriding
    the tower's config (and its layers') is enough — the decoder keeps SDPA.
    """
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None:
            cfg._attn_implementation = "eager"


@contextlib.contextmanager
def _eager_attention(module):
    """Temporarily run `module` on eager attention, restoring the original after.

    ROCm's SDPA returns **NaN from its backward** whenever an attention mask is
    present — the forward is finite, so the loss looks healthy and only the
    gradients are poisoned. Autograd anomaly detection names it outright:
    `ScaledDotProductEfficientAttentionBackward0 returned nan values`.

    That is why only *ragged* audio batches broke: a batch of equal-length clips
    needs no padding mask and trains fine, while one short clip introduces the mask
    and NaNs every tower gradient on that step (measured: 488/488 params, at any lr).
    Text is unaffected — packed token batches carry no padding mask — so the decoder
    keeps fast SDPA there and only the audio branch, whose sequences are ~750 frames
    (vs 4096 for text), pays for eager.
    """
    saved = []
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None:
            saved.append((cfg, cfg._attn_implementation))
            cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        for cfg, impl in saved:
            cfg._attn_implementation = impl


class OmniThinkerCPT(nn.Module):
    """Thinker + a next-audio-frame regression head, with a modality-branching forward.

    The batch decides the branch: an `input_features` key means an audio batch, an
    `input_ids` key means a text batch. (No string "modality" field — those collate
    into lists and would have to be unpacked on every step.) Under FSDP all ranks
    must take the *same* branch on a given step; the mixed-modality schedule in
    data.py guarantees that.
    """

    def __init__(self, thinker, task_cfg):
        super().__init__()
        self.thinker = thinker
        self.audio_loss = task_cfg.get("audio_loss", "cosine")
        self.target_detach = task_cfg.get("target_detach", "detach")

        # Read the width off the embedding table rather than a nested config attr —
        # the thinker's config nests hidden_size under text_config.
        hidden = thinker.get_input_embeddings().weight.shape[1]
        head_dim = int(task_cfg.get("audio_head_dim", None) or hidden)
        self.audio_head = nn.Linear(hidden, head_dim)

        # Target encoder for the regression targets. `frozen` keeps the pretrained
        # tower fixed; `ema` lets it track the live tower slowly. Either way it is
        # held inside a plain list, NOT assigned as a submodule: nn.Module would
        # register it, and FSDP's auto-wrap policy — which keys off the layer CLASS —
        # would then shard the copy's Qwen2_5OmniAudioEncoderLayers exactly as it
        # shards the live tower's. `update_ema` would be lerping a rank's empty shard
        # against a full tensor ("size of tensor a (0) must match b (491520)"). Hidden
        # from .modules()/.parameters(), it stays whole and unsharded on every rank —
        # it is frozen and only ~0.65B params in bf16, so replicating it is cheap.
        self.ema_decay = float(task_cfg.get("ema_decay", 0.999))
        if self.target_detach in ("ema", "frozen"):
            tower = copy.deepcopy(thinker.audio_tower).to(torch.bfloat16)
            for p in tower.parameters():
                p.requires_grad_(False)
            tower.eval()
            self._target = [tower]
        else:
            self._target = []

    # -- audio encoder ---------------------------------------------------------
    def _encode_audio(self, tower, input_features, feature_lens):
        """Run the audio tower over a padded mel batch -> (B, F, D) + valid-frame mask.

        The tower wants the batch *packed*: valid mel frames concatenated along time
        into (n_mel, sum_T), with the per-item lengths passed alongside. Its output is
        likewise packed (sum_F, D), so we re-split it into a padded (B, F, D). This
        mirrors `speech/speech-tknz.py::encode`, but with grad enabled.
        """
        # The dataloader hands us fp32 mel; under FSDP mixed precision the tower's
        # weights are bf16, and the first conv will not cast for us.
        input_features = input_features.to(dtype=next(tower.parameters()).dtype)

        lens = feature_lens.tolist()
        packed = torch.cat(
            [input_features[i, :, : lens[i]] for i in range(input_features.shape[0])],
            dim=1,
        )

        out_lengths = tower._get_feat_extract_output_lengths(feature_lens)
        if isinstance(out_lengths, (tuple, list)):
            aftercnn_lens, out_lens = out_lengths
        else:
            aftercnn_lens, out_lens = feature_lens, out_lengths

        out = tower(packed, feature_lens=feature_lens, aftercnn_lens=aftercnn_lens)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        # packed (sum_F, D) -> padded (B, F, D)
        sizes = out_lens.tolist()
        chunks = torch.split(hidden, sizes, dim=0)
        embeds = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)
        max_f = embeds.shape[1]
        frame_mask = (
            torch.arange(max_f, device=embeds.device)[None, :]
            < out_lens.to(embeds.device)[:, None]
        )
        return embeds, frame_mask

    def _forward_audio(self, input_features, feature_lens):
        embeds, frame_mask = self._encode_audio(
            self.thinker.audio_tower, input_features, feature_lens
        )

        # The audio branch always carries a padding mask (clips have unequal lengths),
        # which is exactly the case ROCm's SDPA backward NaNs on. See _eager_attention.
        with _eager_attention(self.thinker.model):
            hidden = self.thinker.model(
                inputs_embeds=embeds,
                attention_mask=frame_mask.long(),
                use_cache=False,
            ).last_hidden_state
        pred = self.audio_head(hidden)[:, :-1]           # predict frame t+1 from t

        tower = self.target_tower
        if tower is not None:
            # Hidden from the module tree, so Accelerate never moved it — place it on
            # the rank's device on first use.
            if next(tower.parameters()).device != embeds.device:
                tower.to(embeds.device)
            with torch.no_grad():
                target_all, _ = self._encode_audio(tower, input_features, feature_lens)
        else:
            target_all = embeds
        target = target_all[:, 1:].detach()               # stop-grad: see module docstring

        # A position is a valid training target only if both it and its successor are
        # real frames — this drops padding AND the final real frame (no successor).
        valid = frame_mask[:, :-1] & frame_mask[:, 1:]
        if valid.sum() == 0:
            raise RuntimeError("audio batch has no frame with a successor; "
                               "clips are too short for next-frame prediction")

        pred, target = pred[valid], target[valid]
        if self.audio_loss == "cosine":
            loss = (1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()
        else:
            loss = F.mse_loss(pred.float(), target.float())

        return {
            "loss": loss,
            "loss_audio": loss.detach(),
            # Collapse canary: if the encoder degenerates to a constant, the targets
            # lose variance and the loss goes to zero for the wrong reason.
            "audio_target_var": target.float().var(dim=0).mean().detach(),
            "n_audio_frames": int(valid.sum()),
        }

    def _forward_aligned(self, input_features, feature_lens, input_ids, labels,
                         text_lens=None, **_):
        """Ablation 4: predict the transcript FROM the speech. The only branch that
        connects the two modalities.

        Audio frames are prefixed to the transcript embeddings and the loss is taken on
        the text positions only, so gradient reaches the encoder solely through "did
        these frames let the decoder produce these words". That is the supervision
        next-frame regression cannot provide: measured, regression leaves the encoder
        rendering Irish through an English lexicon ("Rudaí deasa" -> "ready does the").

        No loss on the audio positions: there is no ground-truth token for a mel frame,
        and predicting the next audio frame here is exactly the objective we are trying
        to complement.
        """
        embeds, frame_mask = self._encode_audio(
            self.thinker.audio_tower, input_features, feature_lens
        )
        tok_embeds = self.thinker.model.embed_tokens(input_ids).to(embeds.dtype)
        inputs = torch.cat([embeds, tok_embeds], dim=1)

        text_mask = labels.ne(-100)
        attn = torch.cat([frame_mask, text_mask], dim=1).long()

        # Ragged by construction (clips and transcripts both vary), which is the case
        # ROCm's SDPA backward NaNs on. See _eager_attention.
        with _eager_attention(self.thinker.model):
            hidden = self.thinker.model(
                inputs_embeds=inputs, attention_mask=attn, use_cache=False,
            ).last_hidden_state

        # Predict token t+1 from position t. The last audio frame is the position that
        # predicts the FIRST token, so the text logits start one before the text block.
        n_audio = embeds.shape[1]
        logits = self.thinker.lm_head(hidden[:, n_audio - 1 : -1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return {
            "loss": loss,
            "loss_aligned": loss.detach(),
            "n_text_tokens": int(text_mask.sum()),
            "n_audio_frames": int(frame_mask.sum()),
        }

    def _forward_text(self, input_ids, labels=None, **_):
        hidden = self.thinker.model(
            input_ids=input_ids, use_cache=False
        ).last_hidden_state
        logits = self.thinker.lm_head(hidden)
        if labels is None:
            labels = input_ids
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
        )
        return {
            "loss": loss,
            "loss_text": loss.detach(),
            "n_text_tokens": int(labels[:, 1:].numel()),
        }

    def forward(self, **batch):
        # Aligned batches carry BOTH mel and ids, so they must be tested before the
        # audio-only check or they would silently take the regression branch and the
        # transcripts would be ignored.
        if "input_features" in batch and "input_ids" in batch:
            return self._forward_aligned(**batch)
        if "input_features" in batch:
            return self._forward_audio(batch["input_features"], batch["feature_lens"])
        return self._forward_text(**batch)

    @property
    def target_tower(self):
        """The (unsharded, frozen) target encoder, or None under plain `detach`."""
        return self._target[0] if self._target else None

    @torch.no_grad()
    def update_ema(self):
        """Track the live encoder with the target copy (no-op unless target_detach=ema).

        Under FSDP the live tower's parameters are sharded, so a rank holds only a
        slice of each one while the target is whole. `summon_full_params` gathers the
        tower for the duration of the lerp; without it the shapes do not line up.
        """
        if self.target_detach != "ema" or not self._target:
            return
        live = self.thinker.audio_tower
        ctx = contextlib.nullcontext()
        if any(hasattr(m, "_fsdp_wrapped_module") for m in live.modules()):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            ctx = FSDP.summon_full_params(live, writeback=False, recurse=True)
        with ctx:
            for tgt, src in zip(self._target[0].parameters(), live.parameters()):
                if tgt.shape != src.shape:
                    return          # still sharded: skip rather than corrupt the target
                tgt.lerp_(src.detach().to(tgt.dtype), 1.0 - self.ema_decay)


def apply_freezing(model, args):
    """Set requires_grad per the ablation's `freeze.mode`; return a short summary.

    `llm_only` (the Text ablation, per ablations/README): train the LLM decoder,
    lm_head and embeddings; freeze the audio tower.
    `all` (Speech and Text+Speech): everything trains.

    Must be identical on every rank — a divergent frozen-param set breaks FSDP's
    all-gathers.
    """
    mode = args.get("freeze", {}).get("mode", "all")
    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "llm_only":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.thinker.model.parameters():
            p.requires_grad_(True)
        for p in model.thinker.lm_head.parameters():
            p.requires_grad_(True)
        model.thinker.get_input_embeddings().weight.requires_grad_(True)
    else:
        raise ValueError(f"unknown freeze.mode: {mode!r} (expected all | llm_only)")

    # The EMA target tower is never trained, whatever the mode.
    if getattr(model, "target_tower", None) is not None:
        for p in model.target_tower.parameters():
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"[freeze] mode={mode}: {trainable/1e9:.2f}B / {total/1e9:.2f}B params trainable"


def get_model(args):
    m = args.model
    task = args.get("task", {})

    thinker = _load_thinker(m)
    thinker.config.use_cache = False

    model = OmniThinkerCPT(thinker, task)

    # Ablation 4 continues from the `both` checkpoint rather than from stock Omni, so
    # the aligned phase starts where ablation 3 left off. Loaded before FSDP wraps the
    # model and before gradient checkpointing, so the state dict keys still match.
    init_from = m.get("init_from", None)
    if init_from:
        import os as _os
        path = _os.path.join(init_from, "pytorch_model.bin")
        if not _os.path.exists(path):
            raise SystemExit(f"model.init_from: no pytorch_model.bin under {init_from}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Loud, because silently starting from stock weights would look like "the
        # aligned data did nothing" rather than "the checkpoint never loaded".
        print(f"[init] resumed from {init_from} "
              f"({len(missing)} missing, {len(unexpected)} unexpected keys)", flush=True)
        if len(missing) > 50:
            raise SystemExit(f"init_from loaded almost nothing — {len(missing)} missing "
                             f"keys, e.g. {missing[:5]}. Refusing to train from stock "
                             f"weights while claiming to resume.")

    if m.gradient_checkpointing:
        thinker.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    vocab = thinker.get_input_embeddings().weight.shape[0]
    return model, vocab
