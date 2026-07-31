"""Does repeat_kv also fix the ROCm masked-batch NaN, or is the mask itself the problem?

The audio + aligned branches pin EAGER attention because ROCm's SDPA returns NaN from
its BACKWARD whenever an attention mask is present (forward finite, gradients poisoned —
measured 488/488 tower params NaN). That diagnosis was made before we knew GQA was
silently in play on every one of those calls, forcing the math/broadcast path.

So the recorded cause ("SDPA + mask") may really be "SDPA + mask + GQA". If repeat_kv
fixes it too, the speech ablations get the fused kernel as well.

Isolates the two variables at the real audio shape: GQA vs repeated KV, x mask vs no
mask, checking the BACKWARD (the forward was never the broken part).
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Thinker decoder shape; ~750 frames is a 30s audio clip (vs 4096 packed text).
B, HQ, HKV, S, D = 2, 16, 2, 750, 128


def repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def run(gqa, masked, backend):
    torch.manual_seed(0)
    q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    kk, vv = (k, v) if gqa else (repeat_kv(k, HQ // HKV), repeat_kv(v, HQ // HKV))
    kw = {"enable_gqa": True} if gqa else {}

    mask = None
    if masked:
        # A ragged batch: clip 2 is half padding. This is exactly what collate_audio
        # produces and what the NaN was traced to.
        m = torch.ones(B, S, dtype=torch.bool, device="cuda")
        m[1, S // 2:] = False
        mask = m[:, None, None, :].expand(B, 1, S, S)

    try:
        with sdpa_kernel([backend]):
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, **kw)
        o.sum().backward()
        torch.cuda.synchronize()
        # Padded rows legitimately produce NaN grads in q; check k/v, which the tower
        # actually trains through.
        bad = [n for n, t in (("q", q), ("k", k), ("v", v))
               if t.grad is not None and not torch.isfinite(t.grad).all()]
        fwd = "finite" if torch.isfinite(o).all() else "NaN"
        return f"fwd={fwd:6s} grads={'NaN in ' + ','.join(bad) if bad else 'all finite'}"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {str(e)[:60]}"


def main():
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"shape q={B}x{HQ}x{S}x{D} kv_heads={HKV}\n")
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH):
        for gqa in (True, False):
            for masked in (True, False):
                tag = f"{backend.name:20s} {'gqa' if gqa else 'repeat_kv':10s} " \
                      f"{'masked' if masked else 'nomask':7s}"
                print(f"{tag} -> {run(gqa, masked, backend)}", flush=True)
        print()


if __name__ == "__main__":
    main()
