"""What does the REAL text forward pass to SDPA, and why does torch reject flash?

Standalone facts so far: flash works on this MI250X at the exact training shape
(0.06 GiB vs math 3.50 GiB), `_forward_text` passes no attention_mask, and the decoder
is left on sdpa. Yet the baseline profile shows math (bmm + softmax) and forcing
[flash, efficient] raises "No available kernel". One of those beliefs is wrong.

So: intercept the first real SDPA call, print its arguments, and ask torch itself --
with debug=True, which prints the rejection REASON to stderr -- why flash is refused.
"""
import torch
import torch.nn.functional as F
from hydra import compose, initialize

_real_sdpa = F.scaled_dot_product_attention
_seen = []


def _spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    if not _seen:
        _seen.append(True)
        print("\n=== FIRST REAL SDPA CALL FROM THE TEXT DECODER", flush=True)
        print(f"  q={tuple(q.shape)} dtype={q.dtype} contiguous={q.is_contiguous()}")
        print(f"  attn_mask={'None' if attn_mask is None else tuple(attn_mask.shape)}"
              f"{'' if attn_mask is None else f' dtype={attn_mask.dtype}'}")
        print(f"  is_causal={is_causal} dropout_p={dropout_p}")
        from torch.backends.cuda import (SDPAParams, can_use_efficient_attention,
                                         can_use_flash_attention)
        p = SDPAParams(q, k, v, attn_mask, dropout_p, is_causal, False)
        # debug=True makes torch print exactly which constraint failed.
        print(f"  can_use_flash     = {can_use_flash_attention(p, True)}", flush=True)
        print(f"  can_use_efficient = {can_use_efficient_attention(p, True)}", flush=True)
        print("===\n", flush=True)
    return _real_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, **kw)


def main():
    F.scaled_dot_product_attention = _spy
    # transformers binds the symbol at import time in some versions; patch there too.
    import transformers.integrations.sdpa_attention as sa
    if hasattr(sa, "scaled_dot_product_attention"):
        sa.scaled_dot_product_attention = _spy

    with initialize(config_path="qomhra/configs", version_base="1.1"):
        args = compose(config_name="omni_text")

    from qomhra.model import get_model
    model, _ = get_model(args)
    model = model.cuda().to(torch.bfloat16)
    model.gradient_checkpointing_disable() if hasattr(
        model, "gradient_checkpointing_disable") else None

    ids = torch.randint(0, 150000, (1, args.data.seq_len), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=ids, labels=ids.clone())
    print(f"peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
          f"(math would be ~3.5 GiB of score matrices alone)")


if __name__ == "__main__":
    main()
