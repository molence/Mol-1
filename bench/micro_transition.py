"""Microbenchmark the pair Transition (LN -> SwiGLU MLP, n=4) at trunk shapes."""

import torch
import torch.nn.functional as F

from micro_tri import bench

dev = "cuda"


def make(c):
    w_a = torch.randn(4 * c, c, device=dev, dtype=torch.bfloat16) / c**0.5
    w_b = torch.randn(4 * c, c, device=dev, dtype=torch.bfloat16) / c**0.5
    w_o = torch.randn(c, 4 * c, device=dev, dtype=torch.bfloat16) / (4 * c) ** 0.5
    return w_a, w_b, w_o


def ref(x, w_a, w_b, w_o, rows):
    out = torch.empty_like(x)
    for s in range(0, x.shape[0], rows):
        y = F.layer_norm(x[s : s + rows], x.shape[-1:])
        a = F.silu(F.linear(y, w_a))
        b = F.linear(y, w_b)
        b *= a
        out[s : s + rows] = F.linear(b, w_o)
    return out


@torch.compile(dynamic=True)
def silu_mul(ab):
    a, b = ab.chunk(2, dim=-1)
    return F.silu(a) * b


def fused(x, w_ab, w_o, rows):
    out = torch.empty_like(x)
    for s in range(0, x.shape[0], rows):
        y = F.layer_norm(x[s : s + rows], x.shape[-1:])
        out[s : s + rows] = F.linear(silu_mul(F.linear(y, w_ab)), w_o)
    return out


if __name__ == "__main__":
    for n, c in ((648, 320), (648, 448)):
        x = torch.randn(n * n, c, device=dev, dtype=torch.bfloat16)
        w_a, w_b, w_o = make(c)
        w_ab = torch.cat([w_a, w_b], 0)
        flops = 2 * n * n * c * 4 * c * 3
        for rows in (n * n, 128 * n, 64 * n, 32 * n, 16384, 8192):
            with torch.no_grad():
                t0 = bench(lambda: ref(x, w_a, w_b, w_o, rows))
                t1 = bench(lambda: fused(x, w_ab, w_o, rows))
            print(f"N={n} C={c} rows/chunk={rows}: ref {t0:.3f} ms ({flops / t0 / 1e9:.0f} TF)  fused {t1:.3f} ms ({flops / t1 / 1e9:.0f} TF)")
        with torch.no_grad():
            r0, r1 = ref(x, w_a, w_b, w_o, 128 * n), fused(x, w_ab, w_o, 16384)
        print("  max abs diff", (r0.float() - r1.float()).abs().max().item(), "ref max", r0.float().abs().max().item())
