"""Measured device ceilings used by the SOL model: dense GEMM throughput and DRAM copy bandwidth."""
import torch

def t(fn, n=20):
    for _ in range(3): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n / 1e3

print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).multi_processor_count, "SMs")
for dt, name, tf32 in ((torch.bfloat16, "bf16", False), (torch.float32, "tf32", True), (torch.float32, "fp32", False)):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    for m in (8192, 16384):
        a = torch.randn(m, m, device="cuda", dtype=dt); b = torch.randn(m, m, device="cuda", dtype=dt)
        s = t(lambda: a @ b, 5)
        print(f"gemm {name} {m}^3: {2*m**3/s/1e12:.0f} TFLOPS")
# Mol-1 shaped: 648^2 rows x 320 -> 1280 etc.
torch.backends.cuda.matmul.allow_tf32 = False
for k, n in ((320, 1280), (320, 1600), (1280, 320), (448, 1792)):
    a = torch.randn(648*648, k, device="cuda", dtype=torch.bfloat16); w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    s = t(lambda: a @ w.t())
    print(f"gemm bf16 [{648*648}x{k}]x[{k}x{n}]: {2*a.shape[0]*k*n/s/1e12:.0f} TFLOPS, {(a.numel()+a.shape[0]*n)*2/s/1e12:.2f} TB/s")
x = torch.empty(2**30, device="cuda", dtype=torch.uint8); y = torch.empty_like(x)
s = t(lambda: y.copy_(x))
print(f"copy 1 GiB: {2*x.numel()/s/1e12:.2f} TB/s (read+write)")
x = torch.empty(2**28, device="cuda", dtype=torch.float32)
s = t(lambda: x.sum())
print(f"read 1 GiB: {x.numel()*4/s/1e12:.2f} TB/s")
