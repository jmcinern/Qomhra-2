"""Model construction for the training-time estimate.

Quick first number: build a Qwen3-8B *architecture* with random weights (same
FLOPs/step as the real checkpoint, but no 16 GB download and no offline-cache
wrangling). The embedding table is sized for the combined text+audio vocab so the
step cost matches the eventual multimodal run.

Combined vocab id space (one shared embedding table):
  text  ids : [0, text_vocab)            Qwen3 native (sep/eos = 151643)
  audio ids : [text_vocab, text_vocab + audio_vocab)   mHuBERT units offset
  specials  : the next `num_special` ids (audio-BOS/EOS, modality-switch, ...)
  padded up to a multiple of `pad_vocab_to_multiple_of` for matmul efficiency.

Real-weights path (later, for a loss curve): swap `Qwen3ForCausalLM(config)` for
`Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")` then
`model.resize_token_embeddings(combined_vocab, pad_to_multiple_of=128)`.
"""
import math


def combined_vocab_size(m):
    raw = int(m.text_vocab) + int(m.audio_vocab) + int(m.num_special)
    mult = int(m.pad_vocab_to_multiple_of)
    return math.ceil(raw / mult) * mult


def get_model(args):
    from transformers import AutoModelForCausalLM, Qwen3Config

    m = args.model
    vocab = combined_vocab_size(m)

    # Liger fused kernels (config-gated). Patches the Qwen3 module classes BEFORE
    # instantiation: fused RMSNorm/RoPE/SwiGLU kill most of the elementwise bucket,
    # and fused_linear_cross_entropy never materialises the 152,960-vocab logits
    # (frees ~10 GB -> bigger micro-batch, and folds the LM-head+CE into one kernel).
    if m.get("use_liger", False):
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(
            rope=True,
            rms_norm=True,
            swiglu=True,
            fused_linear_cross_entropy=True,
            cross_entropy=False,  # mutually exclusive with the fused linear CE
        )

    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=m.hidden_size,
        intermediate_size=m.intermediate_size,
        num_hidden_layers=m.num_hidden_layers,
        num_attention_heads=m.num_attention_heads,
        num_key_value_heads=m.num_key_value_heads,
        head_dim=m.head_dim,
        max_position_embeddings=m.max_position_embeddings,
        rms_norm_eps=m.rms_norm_eps,
        rope_theta=m.rope_theta,
        tie_word_embeddings=m.tie_word_embeddings,
        use_cache=False,  # training; KV-cache off (and required for grad checkpointing)
    )

    # from_config honors attn_implementation (config kwarg can be ignored when
    # instantiating the class directly).
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation=m.attn_implementation
    )

    if m.gradient_checkpointing:
        # use_reentrant=False is the FSDP-friendly variant.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, vocab
