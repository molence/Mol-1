import os, sys, warnings
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from mol.model.modules.pair_bias import layer_norm_head_projection as f
from micro_tri import bench
for C in (320, 448):
    zp = torch.randn(652, 652, C, device="cuda", dtype=torch.bfloat16); z = zp[:648, :648]
    w = torch.randn(16, C, device="cuda"); lw = torch.randn(C, device="cuda"); lb = torch.randn(C, device="cuda")
    res = sorted((bench(lambda: f(z, lw, lb, w, 1e-5, torch.bfloat16, r, nw)), r, nw) for r in (16, 32, 64, 128) for nw in (2, 4, 8))
    print(C, [(round(t, 3), r, nw) for t, r, nw in res[:4]])
