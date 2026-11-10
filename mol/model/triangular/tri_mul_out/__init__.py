# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused output side of triangle multiplication on Blackwell (CuTe DSL, BF16).

out = sigmoid(x_in W_g^T) * (LayerNorm(t)^T W_p^T) (+ residual), with t the [C, M] einsum
output read in place (no LayerNorm-transpose pass, no [M, C] projections in memory): one
persistent kernel with both GEMMs on tcgen05 and the LayerNorm folded into the epilogue
(see kernel_sm100.py). The per-position LayerNorm statistics come from a small Triton pass.

Needs the CuTe DSL (``pip install "nvidia-cutlass-dsl[cu13]==4.6.0"``, or a directory holding
it in ``MOL_CUTE_DSL_DIR``). Each channel count is compiled once (about 25 s with the DSL
import) into a shared library next to the other extension builds; later processes load it
through tvm_ffi without importing the DSL. ``MOL_TRIMUL_FUSED_OUT=0`` disables it.
"""

import fcntl
import hashlib
import importlib.metadata
import logging
import os
import site
import subprocess

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("MOL_TRIMUL_FUSED_OUT", "1") != "0"
_HERE = os.path.dirname(os.path.abspath(__file__))

_fns = {}
_failed = set()
_num_sms = {}


def _library_path(c: int, has_res: bool):
    from mol.model.layer_norm.torch_ext_compile import _resolve_build_directory

    h = hashlib.sha256()
    for dist in ("nvidia-cutlass-dsl", "apache-tvm-ffi"):
        try:
            h.update(f"{dist}={importlib.metadata.version(dist)}".encode())
        except importlib.metadata.PackageNotFoundError:
            pass
    for name in ("kernel_sm100.py", "__init__.py"):
        with open(os.path.join(_HERE, name), "rb") as f:
            h.update(f.read())
    major, minor = torch.cuda.get_device_capability()
    tag = f"c{c}_{'res' if has_res else 'nores'}_sm{major}{minor}_{h.hexdigest()[:16]}"
    return _resolve_build_directory("mol_tri_mul_out", None) / f"tri_mul_out_{tag}.so"


def _dyn_m(t: torch.Tensor):
    """(M, C, 1) view with M dynamic (multiple of 8), C static."""
    from cutlass.cute.runtime import from_dlpack

    order = (2, 1, 0) if t.stride(0) == 1 else (2, 0, 1)
    return from_dlpack(t, assumed_align=32 if t.stride(0) != 1 else 16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
        mode=0, stride_order=order, divisibility=8)


def _build(so, c: int, has_res: bool):
    """Compile the kernel for c channels with the CuTe DSL and link it into `so`."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.runtime import find_runtime_libraries

    from .kernel_sm100 import TriMulOutSm100

    m = 256
    bf = dict(device="cuda", dtype=torch.bfloat16)
    t = torch.empty(c, m, **bf)
    x, res, out = (torch.empty(m, c, **bf) for _ in range(3))
    w1, w2 = (torch.empty(c, c, **bf) for _ in range(2))
    mean, rstd = (torch.empty(m, device="cuda") for _ in range(2))
    u, v = (torch.empty(c, device="cuda") for _ in range(2))
    vec = lambda a: from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
    fn = cute.compile(
        TriMulOutSm100(c, has_res=has_res),
        _dyn_m(t.t().unsqueeze(-1)), _dyn_m(x.unsqueeze(-1)), vec(w1.unsqueeze(-1)), vec(w2.unsqueeze(-1)),
        _dyn_m(res.unsqueeze(-1)), _dyn_m(out.unsqueeze(-1)),
        vec(mean).mark_layout_dynamic(), vec(rstd).mark_layout_dynamic(), vec(u), vec(v), 1,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    obj = f"{so}.{os.getpid()}.o"
    fn.export_to_c(object_file_path=obj, function_name="tri_mul_out")
    runtime = [p for p in find_runtime_libraries(enable_tvm_ffi=False) if p.endswith("libcute_dsl_runtime.so")]
    lib_dir = os.path.dirname(runtime[0])
    tmp = f"{so}.{os.getpid()}.tmp"
    subprocess.run(
        ["g++", "-shared", "-o", tmp, obj, f"-L{lib_dir}", "-lcute_dsl_runtime", f"-Wl,-rpath,{lib_dir}"],
        check=True, capture_output=True,
    )
    os.replace(tmp, so)
    os.remove(obj)


def _load(c: int, has_res: bool):
    key = (c, has_res)
    if key in _fns or key in _failed:
        return _fns.get(key)
    _failed.add(key)
    dsl_dir = os.environ.get("MOL_CUTE_DSL_DIR")
    if dsl_dir:
        site.addsitedir(dsl_dir)
    try:
        so = _library_path(c, has_res)
        if not so.exists():
            with open(f"{so}.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)  # one build across processes
                if not so.exists():
                    logger.info("Compiling the fused triangle multiplication output kernel for C=%d (once)", c)
                    _build(so, c, has_res)
        import tvm_ffi

        _fns[key] = tvm_ffi.load_module(str(so))["tri_mul_out"]
        _failed.discard(key)
    except Exception as ex:  # no CuTe DSL, or a failed build: callers fall back
        msg = next((line for line in str(ex).splitlines() if line.strip()), type(ex).__name__)
        logger.info("Fused triangle multiplication output kernel unavailable: %s", msg)
    return _fns.get(key)


def supported(t: torch.Tensor, c: int) -> bool:
    """t: the [C, M] einsum output."""
    bn = c // 2
    return (
        _ENABLED
        and t.is_cuda
        and t.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and c % 64 == 0
        and bn % 32 == 0
        and t.shape[-1] % 8 == 0
        and torch.cuda.get_device_capability(t.device)[0] == 10
        and (c, True) not in _failed
    )


def gated_output(
    t: torch.Tensor,
    x_in: torch.Tensor,
    w1: torch.Tensor,
    w_g: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    residual: torch.Tensor = None,
):
    """t: [C, M] (contiguous); x_in: [M, C]; w1 = W_p * gamma and w_g: [C, C] BF16;
    mean, rstd: [M] FP32 statistics of t over C; u = rowsum(w1), v = W_p beta: [C] FP32;
    residual: [M, C] or None. Returns sigmoid(x_in w_g^T) * (rstd (t^T w1^T - mean u) + v)
    (+ residual) as [M, C], or None where the kernel does not apply."""
    c, m = t.shape
    if not supported(t, c):
        return None
    tensors = [t, x_in, w1, w_g] + ([residual] if residual is not None else [])
    if any(a.dtype != torch.bfloat16 or not a.is_contiguous() or a.data_ptr() % 32 for a in tensors):
        return None
    fn = _load(c, residual is not None)
    if fn is None:
        return None
    out = torch.empty(m, c, device=t.device, dtype=t.dtype)
    res = residual if residual is not None else out
    dev = t.device.index if t.device.index is not None else torch.cuda.current_device()
    if dev not in _num_sms:
        _num_sms[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    fn(t.t().unsqueeze(-1), x_in.unsqueeze(-1), w1.unsqueeze(-1), w_g.unsqueeze(-1), res.unsqueeze(-1),
       out.unsqueeze(-1), mean, rstd, u, v, min(_num_sms[dev], (m + 127) // 128))
    return out
