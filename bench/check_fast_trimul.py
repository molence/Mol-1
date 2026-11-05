"""fast_trimul vs cuEquivariance triangle_multiplicative_update: speed and agreement."""
import os, sys, warnings
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update as cueq
from mol.model.triangular import cueq_tuning
from mol.model.triangular.fast_trimul import triangle_multiplicative_update as fast
from micro_tri import bench

cueq_tuning.install(torch.device("cuda"))
for N, C in ((648, 320), (648, 448), (648, 64)):
    for direction in ("outgoing", "incoming"):
        dt = torch.bfloat16
        x = torch.randn(N, N, C, device="cuda", dtype=dt)
        mask = torch.ones(N, N, device="cuda", dtype=dt); mask[:, N - 4:] = 0; mask[N - 4:, :] = 0
        w = lambda *s: (torch.randn(*s, device="cuda") / s[-1] ** 0.5).to(dt)
        ln = lambda: (1 + 0.1 * torch.randn(C, device="cuda")).to(dt)
        nb = lambda: (0.1 * torch.randn(C, device="cuda")).to(dt)
        niw, nib, now, nob = ln(), nb(), ln(), nb()
        p_in, g_in, p_out, g_out = w(2 * C, C), w(2 * C, C), w(C, C), w(C, C)
        with torch.no_grad():
            f_ref = lambda: cueq(x[None], direction=direction, mask=mask[None], norm_in_weight=niw, norm_in_bias=nib,
                                 p_in_weight=p_in, g_in_weight=g_in, norm_out_weight=now, norm_out_bias=nob,
                                 p_out_weight=p_out, g_out_weight=g_out, eps=1e-5)[0]
            w_in = torch.cat([g_in, p_in], 0)
            f_new = lambda: fast(x, direction, mask, niw, nib, w_in, now, nob, p_out, g_out, 1e-5)
            ref, new = f_ref().float(), f_new().float()
            t0, t1 = bench(f_ref), bench(f_new)
        m = mask.bool()
        d = (new - ref)[m]
        print(f"N={N} C={C} {direction}: cueq {t0:.3f} ms  fast {t1:.3f} ms | max err {d.abs().max().item():.3g} mean {d.abs().mean().item():.3g} (ref absmax {ref[m].abs().max().item():.3g})")
