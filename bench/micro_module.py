"""Profile one trunk module (TriangleAttention / TriangleMultiplication / Transition) in isolation.

Usage: python bench/micro_module.py [att|mul|trans] [N] [C]
"""

import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mol.model.modules.primitives import Transition  # noqa: E402
from mol.model.triangular.triangular import TriangleAttention, TriangleMultiplicationOutgoing  # noqa: E402

from micro_tri import bench  # noqa: E402

kind = sys.argv[1] if len(sys.argv) > 1 else "att"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 648
C = int(sys.argv[3]) if len(sys.argv) > 3 else 320
torch.backends.cuda.matmul.allow_tf32 = True

if kind == "att":
    mod = TriangleAttention(c_in=C, c_hidden=32, no_heads=C // 32)
    call = lambda m, z: m(z, mask=None, chunk_size=1024, triangle_attention="cuequivariance", inplace_safe=True)
elif kind == "mul":
    mod = TriangleMultiplicationOutgoing(c_z=C, c_hidden=C)
    call = lambda m, z: m(z, mask=None, inplace_safe=True, _add_with_inplace=True, triangle_multiplicative="cuequivariance")
else:
    mod = Transition(c_in=C, n=4)
    call = lambda m, z: m(z, chunk_size=128)
mod = mod.cuda().eval()
for p in mod.parameters():
    torch.nn.init.normal_(p, std=0.02)
z = torch.randn(N, N, C, device="cuda", dtype=torch.bfloat16)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    t = bench(lambda: call(mod, z))
    print(f"{kind} N={N} C={C}: {t:.3f} ms")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            call(mod, z)
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=90))
