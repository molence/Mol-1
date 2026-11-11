"""LayerNorm: Mol-1 fast LN extension vs torch native at Mol-1 shapes."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mol.model.layer_norm.layer_norm import FusedLayerNorm  # noqa: E402

from micro_tri import bench  # noqa: E402

@torch.compile(dynamic=False)
def ln_compiled(x, w, b):
    return F.layer_norm(x.float(), x.shape[-1:], w.float(), b.float(), 1e-5).to(x.dtype)


for rows, c, dt in ((648 * 648, 320, torch.bfloat16), (648 * 648, 448, torch.bfloat16), (648 * 648, 128, torch.bfloat16),
                    (648 * 648, 320, torch.float32), (5 * 644, 768, torch.float32), (5 * 4372, 128, torch.float32)):
    x = torch.randn(rows, c, device="cuda", dtype=dt)
    fl = FusedLayerNorm(c).cuda()
    w, b = fl.weight.to(dt), fl.bias.to(dt)
    with torch.no_grad():
        t0 = bench(lambda: fl(x), 20)
        t1 = bench(lambda: F.layer_norm(x, (c,), w, b, 1e-5), 20)
        t2 = bench(lambda: ln_compiled(x, w, b), 20)
        err2 = (fl(x).float() - ln_compiled(x, w, b).float()).abs().max().item()
        err = (fl(x).float() - F.layer_norm(x, (c,), w, b, 1e-5).float()).abs().max().item()
    gb = 2 * x.numel() * x.element_size() / 1e9
    print(f"rows={rows} C={c} {dt}: fastLN {t0 * 1e3:.1f} us ({gb / t0:.2f} TB/s)  torch {t1 * 1e3:.1f} us  compiled {t2 * 1e3:.1f} us ({gb / t2:.2f} TB/s)  maxdiff {err:.2g} / {err2:.2g}")
