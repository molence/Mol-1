"""First-call vs steady-state time of each trunk component in a fresh process."""
import os, sys, time, warnings
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from mol.model.modules.primitives import Transition
from mol.model.modules.transformer import AttentionPairBias
from mol.model.triangular.triangular import TriangleAttention, TriangleMultiplicationOutgoing

def t(fn):
    torch.cuda.synchronize(); s = time.perf_counter(); fn(); torch.cuda.synchronize(); return time.perf_counter() - s

torch.backends.cuda.matmul.allow_tf32 = True
z = torch.randn(648, 648, 320, device="cuda", dtype=torch.bfloat16)
s = torch.randn(648, 384, device="cuda", dtype=torch.bfloat16)
mods = {
    "tri_att": (TriangleAttention(c_in=320, c_hidden=32, no_heads=10), lambda m: m(z, mask=None, chunk_size=1024, triangle_attention="cuequivariance")),
    "tri_mul": (TriangleMultiplicationOutgoing(c_z=320, c_hidden=320), lambda m: m(z, mask=None, inplace_safe=True, _add_with_inplace=True, triangle_multiplicative="cuequivariance")),
    "transition": (Transition(c_in=320, n=4), lambda m: m(z, chunk_size=128)),
    "apb": (AttentionPairBias(has_s=False, n_heads=16, c_a=384, c_z=320), lambda m: m(a=s, s=None, z=z)),
}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for name, (m, f) in mods.items():
        m = m.cuda().eval()
        print(f"{name}: first {t(lambda: f(m)) * 1e3:.1f} ms, second {t(lambda: f(m)) * 1e3:.1f} ms", flush=True)
    z2 = torch.randn(656, 656, 320, device="cuda", dtype=torch.bfloat16)
    m = mods["tri_att"][0]
    print(f"tri_att new shape: first {t(lambda: m(z2, mask=None, chunk_size=1024, triangle_attention='cuequivariance')) * 1e3:.1f} ms")
