"""Sweep tile configs of cuEquivariance's gated dual GEMM beyond its tuning list, vs cuBLAS."""

import itertools
import warnings

import torch
from cuequivariance_ops.triton.utils import Precision
from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper as gg

from micro_tri import bench

warnings.filterwarnings("ignore")
dev = "cuda"
M = 648 * 648
for K, N, two in ((320, 640, False), (320, 320, True), (448, 896, False), (448, 448, True)):
    x1 = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    x2 = torch.randn_like(x1) if two else None
    w1 = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K**0.5
    w2 = torch.randn_like(w1)
    mask = None if two else torch.ones(M, device=dev, dtype=torch.bfloat16)
    tr = not two
    fl = 2 * M * K * N * 2
    res = []
    for tm, tn, tk, ns, nw in itertools.product((64, 128, 256), (64, 128, 256), (32, 64, 128), (2, 3, 4), (4, 8)):
        if N % tn or K % tk:
            continue
        try:
            t = bench(lambda: gg(x1, x2, w1, w2, None, None, mask, tr, TILE_M=tm, TILE_N=tn, TILE_K=tk,
                                 num_stages=ns, num_warps=nw, precision=Precision.DEFAULT.value), 5)
            res.append((t, (tm, tn, tk, ns, nw)))
        except Exception:
            pass
    res.sort()
    cub = bench(lambda: torch.mm(x1, torch.cat([w1, w2]).t()), 5)
    print(f"K={K} N={N} two={two}: best {res[0][0]:.3f} ms {fl / res[0][0] / 1e9:.0f} TF {res[0][1]} | 2nd {res[1][0]:.3f} {res[1][1]} | cuBLAS plain GEMM {cub:.3f} ms {fl / cub / 1e9:.0f} TF")
