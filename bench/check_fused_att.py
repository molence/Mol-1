"""Compare TriangleAttention fused inference path vs the reference mha path."""
import os, sys, warnings
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import mol.model.triangular.triangular as tri
from micro_tri import bench

for N, C, dt, ac in ((648, 320, torch.bfloat16, True), (648, 448, torch.bfloat16, True), (200, 320, torch.float32, False), (648, 64, torch.bfloat16, True)):
    heads = max(1, C // 32)
    for starting in (True, False):
        mod = tri.TriangleAttention(c_in=C, c_hidden=32 if C >= 32 else C, no_heads=heads, starting=starting).cuda().eval()
        for p in mod.parameters():
            torch.nn.init.normal_(p, std=0.05)
        z = torch.randn(N, N, C, device="cuda", dtype=dt)
        mask = torch.ones(N, N, device="cuda", dtype=dt); mask[:, N - 5:] = 0; mask[N - 5:, :] = 0
        call = lambda: mod(z, mask=mask, chunk_size=1024, triangle_attention="cuequivariance", inplace_safe=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
            tri._FUSED_TRI_ATT = False; ref = call().float(); t0 = bench(call)
            tri._FUSED_TRI_ATT = True; out = call().float(); t1 = bench(call)
        m = mask.bool()
        err = (out - ref)[m].abs().max().item()
        print(f"N={N} C={C} {dt} starting={starting}: ref {t0:.3f} ms fused {t1:.3f} ms | max err {err:.3g} (ref max {ref[m].abs().max().item():.3g})")
