"""Tune the Triton LayerNorm (rows per program, warps) against the fast LN extension."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mol.model.layer_norm.layer_norm import FusedLayerNorm
from mol.model.layer_norm import triton_layer_norm as tln
from micro_tri import bench

for rows, c, dt in ((648 * 648, 320, torch.bfloat16), (648 * 648, 448, torch.bfloat16), (648 * 648, 128, torch.bfloat16),
                    (648 * 648, 64, torch.bfloat16), (648 * 648, 320, torch.float32), (648 * 648, 448, torch.float32)):
    x = torch.randn(rows, c, device="cuda", dtype=dt)
    fl = FusedLayerNorm(c).cuda()
    torch.nn.init.normal_(fl.weight); torch.nn.init.normal_(fl.bias)
    with torch.no_grad():
        ref = fl(x)
        t0 = bench(lambda: fl(x), 20)
        res = []
        for r in (4, 8, 16, 32, 64):
            for nw in (4, 8):
                t = bench(lambda: tln.layer_norm(x, fl.weight, fl.bias, 1e-5, r, nw), 20)
                res.append((t, r, nw))
        t, r, nw = min(res)
        tdef = bench(lambda: tln.layer_norm(x, fl.weight, fl.bias, 1e-5), 20)
        err = (tln.layer_norm(x, fl.weight, fl.bias, 1e-5).float() - ref.float()).abs().max().item()
    print(f"rows={rows} C={c} {dt}: fastLN {t0*1e3:.1f} us | triton default {tdef*1e3:.1f} us | best {t*1e3:.1f} us (rows={r}, warps={nw}) | maxdiff {err:.2g}")
