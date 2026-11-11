"""Microbenchmark cuEquivariance triangle kernels at Mol-1 trunk shapes."""

import math
import warnings

import torch
from cuequivariance_torch.primitives.triangle import triangle_attention, triangle_multiplicative_update

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"


def bench(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def tri_att(N, H, D, dtype, bias_dtype):
    q = torch.randn(1, N, H, N, D, device=dev, dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    bias = torch.randn(1, 1, H, N, N, device=dev, dtype=bias_dtype)
    mask = torch.ones(1, N, 1, 1, N, device=dev, dtype=torch.bool)
    with torch.no_grad():
        return bench(lambda: triangle_attention(q, k, v, bias, mask=mask, scale=1 / math.sqrt(D)))


def tri_mul(N, C, dtype):
    x = torch.randn(1, N, N, C, device=dev, dtype=dtype)
    mask = torch.ones(1, N, N, device=dev, dtype=dtype)
    w = lambda *s: (torch.randn(*s, device=dev, dtype=dtype) / math.sqrt(s[-1]))
    args = dict(
        mask=mask, norm_in_weight=torch.ones(C, device=dev, dtype=dtype), norm_in_bias=torch.zeros(C, device=dev, dtype=dtype),
        p_in_weight=w(2 * C, C), g_in_weight=w(2 * C, C),
        norm_out_weight=torch.ones(C, device=dev, dtype=dtype), norm_out_bias=torch.zeros(C, device=dev, dtype=dtype),
        p_out_weight=w(C, C), g_out_weight=w(C, C), eps=1e-5,
    )
    with torch.no_grad():
        return bench(lambda: triangle_multiplicative_update(x, direction="outgoing", **args))


if __name__ == "__main__":
    print("cc", torch.cuda.get_device_capability())
    for N in (644, 648):
        for H, D in ((16, 20), (16, 32), (8, 40)):
            for dt, bdt in ((torch.bfloat16, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)):
                try:
                    t = tri_att(N, H, D, dt, bdt)
                    fl = 4 * N**3 * H * D
                    print(f"tri_att N={N} H={H} D={D} {dt} bias={bdt}: {t:.3f} ms  {fl / t / 1e9:.0f} TFLOPS")
                except Exception as ex:
                    print(f"tri_att N={N} H={H} D={D} {dt}: FAIL {type(ex).__name__}: {str(ex)[:120]}")
    for N in (644, 648):
        for C in (320, 448):
            for dt in (torch.bfloat16, torch.float32):
                t = tri_mul(N, C, dt)
                fl = 2 * N**3 * C + 2 * N * N * C * (4 * C) + 2 * N * N * C * 2 * C
                print(f"tri_mul N={N} C={C} {dt}: {t:.3f} ms  {fl / t / 1e9:.0f} TFLOPS")
