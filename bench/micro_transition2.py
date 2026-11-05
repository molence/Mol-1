"""Transition variants: current module vs fused-weight GEMM + Triton SwiGLU with L2-sized row chunks."""

import os
import sys
import warnings

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from mol.model.modules.primitives import Transition  # noqa: E402
from mol.model.layer_norm import triton_layer_norm as tln  # noqa: E402

from micro_tri import bench  # noqa: E402


@triton.jit
def _silu_mul(AB, Y, M, F_: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = (m < M)[:, None] & (f < F_)[None, :]
    row = m.to(tl.int64)[:, None] * (2 * F_)
    a = tl.load(AB + row + f[None, :], mask=mask, other=0.0).to(tl.float32)
    b = tl.load(AB + row + F_ + f[None, :], mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(a) * b
    tl.store(Y + m.to(tl.int64)[:, None] * F_ + f[None, :], y.to(Y.dtype.element_ty), mask=mask)


def silu_mul(ab):
    m, f2 = ab.shape
    f = f2 // 2
    y = torch.empty((m, f), device=ab.device, dtype=ab.dtype)
    _silu_mul[(triton.cdiv(m, 32), triton.cdiv(f, 128))](ab, y, m, f, BLOCK_M=32, BLOCK_F=128, num_warps=4)
    return y


def fused(mod, w_ab, w_o, x, rows):
    c = x.shape[-1]
    x2 = x.reshape(-1, c)
    out = torch.empty_like(x2)
    for s in range(0, x2.shape[0], rows):
        y = tln.layer_norm(x2[s : s + rows], mod.layernorm1.weight, mod.layernorm1.bias, 1e-5) if x2.shape[0] >= 65536 else mod.layernorm1(x2[s : s + rows])
        torch.mm(silu_mul(torch.mm(y, w_ab.t())), w_o.t(), out=out[s : s + rows])
    return out.view(x.shape)


if __name__ == "__main__":
    for n, c in ((648, 320), (648, 448)):
        mod = Transition(c_in=c, n=4).cuda().eval()
        for p in mod.parameters():
            torch.nn.init.normal_(p, std=0.05)
        x = torch.randn(n, n, c, device="cuda", dtype=torch.bfloat16)
        w_ab = torch.cat([mod.linear_no_bias_a.weight, mod.linear_no_bias_b.weight], 0).bfloat16()
        w_o = mod.linear_no_bias.weight.bfloat16()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = mod(x, chunk_size=128).float()
            t0 = bench(lambda: mod(x, chunk_size=128))
            t1 = bench(lambda: mod(x, chunk_size=None))
            print(f"N={n} C={c}: module chunk128 {t0:.3f} ms | module unchunked {t1:.3f} ms")
            for rows in (8192, 16384, 32768, 65536, n * n):
                with torch.autocast("cuda", enabled=False):
                    t = bench(lambda: fused(mod, w_ab, w_o, x, rows))
                    err = (fused(mod, w_ab, w_o, x, rows).float() - ref).abs().max().item()
                print(f"   fused rows={rows}: {t:.3f} ms  maxerr {err:.3g} (ref max {ref.abs().max().item():.3g})")
