"""Triangle attention at trunk shapes: cuEquivariance vs torch SDPA backends."""

import math
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cuequivariance_torch.primitives.triangle import triangle_attention
from micro_tri import bench

warnings.filterwarnings("ignore")
dev = "cuda"

for N, H, D in ((648, 10, 32), (648, 14, 32), (1024, 10, 32)):
    q = torch.randn(N, H, N, D, device=dev, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    bias = torch.randn(1, H, N, N, device=dev, dtype=torch.bfloat16)
    mask = torch.ones(N, 1, 1, N, device=dev, dtype=torch.bool)
    flops = 4 * N**3 * H * D
    with torch.no_grad():
        ref = triangle_attention(q[None], k[None], v[None], bias[None], mask=mask[None], scale=1 / math.sqrt(D))[0]
        t = bench(lambda: triangle_attention(q[None], k[None], v[None], bias[None], mask=mask[None], scale=1 / math.sqrt(D)))
        print(f"N={N} H={H} D={D} cueq: {t:.3f} ms {flops / t / 1e9:.0f} TF")
        for name, be in (("cudnn", SDPBackend.CUDNN_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel([be]):
                    fn = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias.expand(N, H, N, N))
                    out = fn()
                    t = bench(fn)
                err = (out.float() - ref.float()).abs().max().item()
                print(f"N={N} H={H} D={D} {name}: {t:.3f} ms {flops / t / 1e9:.0f} TF  maxerr {err:.3g}")
            except Exception as ex:
                print(f"N={N} H={H} D={D} {name}: FAIL {str(ex)[:150]}")
