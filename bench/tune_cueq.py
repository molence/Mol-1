"""Tune cuEquivariance's gated dual GEMM for Mol-1's pair widths on this GPU.

cuEquivariance ships tuned configs only for pair width 128/256; Mol-1 uses
64 (template), 320 and 448. This runs the real triangle multiplication at a
few token counts under CUEQ_TRITON_TUNING=ONDEMAND and writes the Mol-1
entries to mol/model/triangular/cueq_tuning/<kernel>.<major>.<minor>.json.

Usage: python bench/tune_cueq.py <shard> <n_shards> <cache_dir>   (tune)
       python bench/tune_cueq.py merge <out_dir> <cache_dir>...      (merge)
"""

import json
import os
import sys

FN = "fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper"
NS = (96, 160, 256, 384, 512)
CS = (64, 320, 448)
PRECISIONS = ("bf16", "tf32", "fp32")


def tune(shard, n_shards, cache_dir):
    os.environ["CUEQ_TRITON_TUNING"] = "ONDEMAND"
    os.environ["CUEQ_TRITON_CACHE_DIR"] = cache_dir
    os.environ["CUEQ_TRITON_IGNORE_EXISTING_CACHE"] = "1"
    import torch

    sys.path.insert(0, os.path.dirname(__file__))
    from micro_tri import tri_mul

    jobs = [(n, c, p) for n in NS for c in CS for p in PRECISIONS]
    for i, (n, c, p) in enumerate(jobs):
        if i % n_shards != shard:
            continue
        torch.backends.cuda.matmul.allow_tf32 = p == "tf32"
        t = tri_mul(n, c, torch.bfloat16 if p == "bf16" else torch.float32)
        print(f"tuned N={n} C={c} {p}: {t:.3f} ms", flush=True)


def merge(out_dir, cache_dirs):
    merged = {}
    for d in cache_dirs:
        for name in os.listdir(d):
            if not name.startswith(FN):
                continue
            arch = name[len(FN) + 1 : -len(".json")]
            with open(os.path.join(d, name)) as f:
                data = json.load(f)
            for gpu_cache in data.values():
                for k, v in gpu_cache.items():
                    if k == "gpu_information":
                        continue
                    _, n, kk = k.split("_")[:3]
                    if kk == "128" and n in ("128", "256"):
                        continue  # shipped by cuEquivariance
                    merged.setdefault(arch, {})[k] = v
    os.makedirs(out_dir, exist_ok=True)
    for arch, entries in merged.items():
        path = os.path.join(out_dir, f"{FN}.{arch}.json")
        with open(path, "w") as f:
            json.dump(dict(sorted(entries.items())), f, indent=1)
        print(f"wrote {len(entries)} entries to {path}")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3:])
    else:
        tune(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
