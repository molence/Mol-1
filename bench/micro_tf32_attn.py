"""TF32 Triton attention vs FP32 SDPA at the diffusion/confidence shapes: error and time."""
import torch
import torch.nn.functional as F
from mol.model.modules.triton_attention import tf32_attention, tf32_attention_ok

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

def t(fn, n=50):
    for _ in range(5): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1e3

g = torch.Generator(device="cuda").manual_seed(0)
cases = {
    "token": ((5, 644, 16, 48), (1, 16, 644, 644)),
    "atom": ((1024, 32, 32), (1024, 32, 128)),
    "conf": ((16, 644, 24), (16, 644, 644)),
}
for name, (qs, bs) in cases.items():
    if name == "token":
        qkv = [torch.randn(qs, device="cuda", generator=g).transpose(1, 2) * 0.4 for _ in range(3)]
        bias = torch.randn(bs, device="cuda", generator=g) * 2
    elif name == "atom":
        q = torch.randn(qs, device="cuda", generator=g) * 0.4
        k = torch.randn(1024, 128, 32, device="cuda", generator=g) * 0.4
        v = torch.randn(1024, 128, 32, device="cuda", generator=g)
        qkv = [q, k, v]
        bias = torch.randn(bs, device="cuda", generator=g) * 2
        bias[..., 100:] = -1e9  # masked keys
    else:
        qkv = [torch.randn(qs, device="cuda", generator=g) * 0.4 for _ in range(3)]
        bias = torch.randn(bs, device="cuda", generator=g) * 2
    q, k, v = qkv
    assert tf32_attention_ok(q, k, v, bias)
    ref64 = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=bias.double(), scale=1.0)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
    out = tf32_attention(q, k, v, bias)
    print(f"{name}: fp32 sdpa err {(ref.double()-ref64).abs().max():.2e}  tf32 triton err {(out.double()-ref64).abs().max():.2e} (mean {(out.double()-ref64).abs().mean():.2e})")
    print(f"   sdpa fp32 {t(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)):.1f} us   triton tf32 {t(lambda: tf32_attention(q, k, v, bias)):.1f} us")
