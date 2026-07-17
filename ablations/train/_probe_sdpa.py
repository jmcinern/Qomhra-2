"""Which SDPA backends actually exist on this ROCm build, and why not the fast ones?

Arm E (sdpa_backends=[flash,efficient]) died with "No available kernel", and the
baseline profile shows attention on the math backend: explicit bmm + a full
(B, heads, 4096, 4096) score matrix per layer. That is both ~27% of CUDA time AND the
reason gradient_checkpointing cannot be turned off (59 GiB of activations in one
forward). So this is the highest-value question in the whole optimisation pass.

Tests the real training shape, and reports torch's own reason for rejecting a backend.
"""
import torch
from torch.backends.cuda import (SDPAParams, can_use_efficient_attention,
                                 can_use_flash_attention)
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as sdpa

# Qwen2.5-Omni-3B thinker: hidden 2048, 16 heads -> head_dim 128, seq 4096 packed.
B, H, S, D = 1, 16, 4096, 128


def main():
    print(f"torch {torch.__version__} | gpu {torch.cuda.get_device_name(0)}")
    print(f"built with flash={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
          f"math={torch.backends.cuda.math_sdp_enabled()}")

    q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))

    # torch's own verdict, with the rejection reason printed to stderr by debug=True.
    p = SDPAParams(q, k, v, None, 0.0, True, False)
    print(f"\ncan_use_flash_attention     = {can_use_flash_attention(p, True)}")
    print(f"can_use_efficient_attention = {can_use_efficient_attention(p, True)}")

    for name, backend in [("flash", SDPBackend.FLASH_ATTENTION),
                          ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("math", SDPBackend.MATH)]:
        torch.cuda.reset_peak_memory_stats()
        try:
            with sdpa_kernel([backend]):
                o = sdpa(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            # Peak is the point: math must hold S*S per head; flash never does.
            print(f"{name:10s} OK   out={tuple(o.shape)} "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:5.2f} GiB")
        except Exception as e:
            print(f"{name:10s} FAIL {type(e).__name__}: {str(e)[:90]}")


if __name__ == "__main__":
    main()
