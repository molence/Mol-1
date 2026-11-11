"""Speed-of-light model for Mol-1 inference, compared with an nsys trace.

For each module: FLOPs, the minimum DRAM traffic (read the input and write the
output once), and the time those take at the GB300 ceilings measured by
bench/peaks.py. SOL = max(compute, memory); for attention, also the exp
throughput of the special-function units. "Measured" is the kernel time per
call from an nsys trace taken with MOL_BENCH_NVTX=1 (bench/nsys.sh).

Usage: python bench/sol.py [trace.sqlite] [--tokens 644]
"""

import argparse
import os
import subprocess
import sys

# Measured on one GB300 (bench/peaks.py): cuBLAS 16k^3 GEMM and a 1 GiB copy.
BF16 = 2.05e15
TF32 = 1.05e15
FP32 = 73e12
DRAM = 6.8e12  # bytes/s, read + write
# exp: 16 MUFU ops/clk/SM x 152 SMs x ~1.9 GHz under load
EXP = 16 * 152 * 1.9e9


def tri_att(n, c, h, d=32, act=2):
    L = n * n
    proj = 4 * h * d + h
    flops = 2 * L * c * proj + 4 * n**3 * h * d + 2 * L * h * d * c
    return flops, 2 * L * c * act, n**3 * h


def tri_mul(n, c, act=2):
    L = n * n
    flops = 2 * L * c * 4 * c + 2 * n**3 * c + 2 * L * c * 2 * c
    return flops, 2 * L * c * act, 0


def transition(n_rows, c, f=4, act=2):
    flops = 2 * n_rows * c * 2 * f * c + 2 * n_rows * f * c * c
    return flops, 2 * n_rows * c * act, 0


def sol(flops, bytes_, exps, peak):
    t_c, t_m, t_e = flops / peak, bytes_ / DRAM, exps / EXP
    return max(t_c, t_m, t_e), t_c, t_m, t_e


def measured(db, match_exact):
    """Kernel ms per call for ranges whose path ends with the given suffix and contains a prefix."""
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "nsys_attr.py"), db, "--depth", "12", "--top", "0"],
        capture_output=True, text=True,
    ).stdout
    res = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        res[parts[-1]] = (float(parts[0].rstrip("s")), int(parts[4]), float(parts[7]))
    return {k: v for k, v in res.items() if match_exact(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", nargs="?")
    ap.add_argument("--tokens", type=int, default=644)
    args = ap.parse_args()
    n = (args.tokens + 7) // 8 * 8
    P = "pairformer_stack.blocks.*"
    rows = [
        # name, nsys range path, calls per prediction, (flops, bytes, exps), peak
        ("Pairformer tri_att (c=320, 10 heads)", f"trunk;pairformer_stack;{P};{P}.tri_att_start", 2 * 40 * 20,
         tri_att(n, 320, 10), BF16),
        ("Pairformer tri_mul (c=320)", f"trunk;pairformer_stack;{P};{P}.tri_mul_out", 2 * 40 * 20, tri_mul(n, 320), BF16),
        ("Pairformer transition (c=320, 4x)", f"trunk;pairformer_stack;{P};{P}.pair_transition", 40 * 20,
         transition(n * n, 320), BF16),
        ("PairMixer tri_mul (c=448)", f"trunk;pairformer_stack;{P}.tri_mul_out", 2 * 8 * 20, tri_mul(n, 448), BF16),
        ("PairMixer transition (c=448, 4x)", f"trunk;pairformer_stack;{P}.pair_transition", 8 * 20,
         transition(n * n, 448), BF16),
    ]
    meas = {}
    if args.db:
        meas = measured(args.db, lambda k: True)
    print(f"N = {n} (padded), ceilings: BF16 {BF16/1e15:.2f} PF, TF32 {TF32/1e15:.2f} PF, DRAM {DRAM/1e12:.1f} TB/s, exp {EXP/1e12:.1f} T/s")
    print(f"{'module':40s} {'GFLOP':>7s} {'MB':>6s} {'t_comp':>7s} {'t_mem':>7s} {'t_exp':>7s} {'SOL ms':>7s} {'meas ms':>8s} {'gap':>5s} {'calls':>6s} {'excess s':>8s}")
    total_meas = total_sol = 0
    for name, path, calls, (f, b, e), peak in rows:
        s, tc, tm, te = sol(f, b, e, peak)
        m = meas.get(path)
        if m is not None:
            # Range names are shared (e.g. PairMixer and Pairformer blocks), so
            # divide the range's total time by this module's own call count.
            # tri_att_start / tri_mul_out stand for both directions.
            ms = m[0] * 1e3 / (calls // 2 if calls % 2 == 0 and ("tri_" in path) else calls)
        mtxt = f"{ms:8.3f} {ms / (s * 1e3):5.1f} {calls:6d} {(ms / 1e3 - s) * calls:8.2f}" if m else ""
        if m:
            total_meas += ms / 1e3 * calls
            total_sol += s * calls
        print(f"{name:40s} {f/1e9:7.0f} {b/1e6:6.0f} {tc*1e3:7.3f} {tm*1e3:7.3f} {te*1e3:7.3f} {s*1e3:7.3f} {mtxt}")
    if total_meas:
        print(f"{'trunk pair ops total':40s} measured {total_meas:.2f} s, SOL {total_sol:.2f} s")

    # Diffusion: token transformer per step (5 samples, 24 blocks, c=768, 16 heads) in TF32.
    tok, c, steps, S = args.tokens, 768, 200, 5
    per_tok = 2 * (4 * c * c + c * c + 3 * c * 2 * c + 4 * c * c)  # q,k,v,g + o + transition (2x) + adaLN
    att = 4 * tok * tok * c
    f_step = 24 * S * (tok * per_tok + att)
    w_bytes = 24 * (15 * c * c) * 4 + 24 * 16 * tok * tok * 4  # FP32 weights + pair bias per step
    s_tok = max(f_step / TF32, w_bytes / DRAM)
    print(f"\nDiffusion token transformer per step: {f_step/1e9:.0f} GFLOP, {w_bytes/1e6:.0f} MB -> SOL {s_tok*1e3:.2f} ms "
          f"({steps} steps: {s_tok*steps:.2f} s)")


if __name__ == "__main__":
    main()
