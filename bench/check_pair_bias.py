"""Fused pair-bias projection vs reference LayerNorm + linear + permute."""
import os, sys, warnings
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import mol.model.modules.transformer as tr
from mol.model.modules.transformer import AttentionPairBias
from micro_tri import bench

for N, C in ((648, 320), (648, 448), (644, 320)):
    apb = AttentionPairBias(has_s=False, n_heads=16, c_a=384, c_z=C).cuda().eval()
    for p in apb.parameters():
        torch.nn.init.normal_(p, std=0.1)
    zp = torch.randn(N + 4, N + 4, C, device="cuda", dtype=torch.bfloat16)
    z = zp[:N, :N]  # strided view, as the padded trunk passes it
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tr._FUSED_PAIR_BIAS = False; ref = apb.project_pair_bias(z).float(); t0 = bench(lambda: apb.project_pair_bias(z))
        tr._FUSED_PAIR_BIAS = True; new = apb.project_pair_bias(z).float(); t1 = bench(lambda: apb.project_pair_bias(z))
    print(f"N={N} C={C}: ref {t0:.3f} ms fused {t1:.3f} ms | max err {(new - ref).abs().max().item():.3g} (ref absmax {ref.abs().max().item():.3g}) shape {tuple(new.shape)}")
