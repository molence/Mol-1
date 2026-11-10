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

"""Triangle multiplication GEMMs on Blackwell as CuTe DSL kernels (BF16, inference).

- gated_projection: ab = sigmoid(x_in W_g^T) * (x_in W_p^T) * mask as [2C, M]: CTA pairs
  (cta_group::2 MMAs) with the X tile resident in shared memory while the weight blocks
  stream past (projection_sm100.py).
  ``MOL_TRIMUL_CUTE_PROJECTION=0`` disables it.
- gated_output: out = sigmoid(x_in W_g^T) * (LayerNorm(t)^T W_p^T) (+ residual), with t the
  [C, M] einsum output read in place (no LayerNorm-transpose pass, no [M, C] projections in
  memory): both GEMMs on tcgen05 and the LayerNorm folded into the epilogue
  (output_sm100.py); the per-position statistics come from a small Triton pass.
  ``MOL_TRIMUL_FUSED_OUT=0`` disables it.

Needs the CuTe DSL (``pip install "nvidia-cutlass-dsl[cu13]==4.6.0"``, or a directory holding
it in ``MOL_CUTE_DSL_DIR``). Each kernel and channel count is compiled once (about 25 s with the
DSL import) into a shared library next to the other extension builds; later processes load it
through tvm_ffi without importing the DSL.
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
_PROJ_ENABLED = os.getenv("MOL_TRIMUL_CUTE_PROJECTION", "1") != "0"
BLOCK = 128  # rows per gate / projection block of the interleaved input weight
_HERE = os.path.dirname(os.path.abspath(__file__))

_fns = {}
_failed = set()
_num_sms = {}


def _library_path(key):
    from mol.model.layer_norm.torch_ext_compile import _resolve_build_directory

    h = hashlib.sha256()
    for dist in ("nvidia-cutlass-dsl", "apache-tvm-ffi"):
        try:
            h.update(f"{dist}={importlib.metadata.version(dist)}".encode())
        except importlib.metadata.PackageNotFoundError:
            pass
    for name in ("output_sm100.py", "projection_sm100.py", "__init__.py"):
        with open(os.path.join(_HERE, name), "rb") as f:
            h.update(f.read())
    major, minor = torch.cuda.get_device_capability()
    tag = "_".join(str(k) for k in key) + f"_sm{major}{minor}_{h.hexdigest()[:16]}"
    return _resolve_build_directory("mol_tri_mul_cute", None) / f"tri_mul_{tag}.so"


def _dyn_m(t: torch.Tensor):
    """(M, C, 1) view with M dynamic (multiple of 8), C static."""
    from cutlass.cute.runtime import from_dlpack

    order = (2, 1, 0) if t.stride(0) == 1 else (2, 0, 1)
    return from_dlpack(t, assumed_align=32 if t.stride(0) != 1 else 16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
        mode=0, stride_order=order, divisibility=8)


def _build(so, key):
    """Compile kernel `key` with the CuTe DSL and link it into `so`."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.runtime import find_runtime_libraries

    kind, c = key[0], key[1]
    m = 256
    bf = dict(device="cuda", dtype=torch.bfloat16)
    vec = lambda a: from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if kind == "out":
        from .output_sm100 import TriMulOutSm100

        t = torch.empty(c, m, **bf)
        x, res, out = (torch.empty(m, c, **bf) for _ in range(3))
        w1, w2 = (torch.empty(c, c, **bf) for _ in range(2))
        mean, rstd = (torch.empty(m, device="cuda") for _ in range(2))
        u, v = (torch.empty(c, device="cuda") for _ in range(2))
        fn = cute.compile(
            TriMulOutSm100(c, has_res=key[2]),
            _dyn_m(t.t().unsqueeze(-1)), _dyn_m(x.unsqueeze(-1)), vec(w1.unsqueeze(-1)), vec(w2.unsqueeze(-1)),
            _dyn_m(res.unsqueeze(-1)), _dyn_m(out.unsqueeze(-1)),
            vec(mean).mark_layout_dynamic(), vec(rstd).mark_layout_dynamic(), vec(u), vec(v), 1, stream,
            options="--enable-tvm-ffi",
        )
    else:
        from .projection_sm100 import TriMulProjectionSm100

        x = torch.empty(m, c, **bf)
        w = torch.empty(4 * c, c, **bf)
        mask = torch.empty(m, **bf)
        ab = torch.empty(2 * c, m, **bf)
        fn = cute.compile(
            TriMulProjectionSm100(c), _dyn_m(x.unsqueeze(-1)), vec(w.unsqueeze(-1)), vec(mask).mark_layout_dynamic(),
            from_dlpack(ab.t(), assumed_align=16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
                mode=0, stride_order=(1, 0), divisibility=8),
            1, stream, options="--enable-tvm-ffi",
        )
    obj = f"{so}.{os.getpid()}.o"
    fn.export_to_c(object_file_path=obj, function_name="tri_mul")
    runtime = [p for p in find_runtime_libraries(enable_tvm_ffi=False) if p.endswith("libcute_dsl_runtime.so")]
    lib_dir = os.path.dirname(runtime[0])
    tmp = f"{so}.{os.getpid()}.tmp"
    subprocess.run(
        ["g++", "-shared", "-o", tmp, obj, f"-L{lib_dir}", "-lcute_dsl_runtime", f"-Wl,-rpath,{lib_dir}"],
        check=True, capture_output=True,
    )
    os.replace(tmp, so)
    os.remove(obj)


def _load(key):
    if key in _fns or key in _failed:
        return _fns.get(key)
    _failed.add(key)
    dsl_dir = os.environ.get("MOL_CUTE_DSL_DIR")
    if dsl_dir:
        site.addsitedir(dsl_dir)
    try:
        so = _library_path(key)
        if not so.exists():
            with open(f"{so}.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)  # one build across processes
                if not so.exists():
                    logger.info("Compiling the triangle multiplication kernel %s (once)", key)
                    _build(so, key)
        import tvm_ffi

        _fns[key] = tvm_ffi.load_module(str(so))["tri_mul"]
        _failed.discard(key)
    except Exception as ex:  # no CuTe DSL, or a failed build: callers fall back
        msg = next((line for line in str(ex).splitlines() if line.strip()), type(ex).__name__)
        logger.info("Triangle multiplication kernel %s unavailable: %s", key, msg)
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
        and t.numel() < 2**31  # 32-bit offsets in the kernel
        and torch.cuda.get_device_capability(t.device)[0] == 10
        and ("out", c, True) not in _failed
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
    fn = _load(("out", c, residual is not None))
    if fn is None:
        return None
    out = torch.empty(m, c, device=t.device, dtype=t.dtype)
    res = residual if residual is not None else out
    if u.dtype != torch.float32 or v.dtype != torch.float32 or mean.dtype != torch.float32:
        return None
    pairs = min(_ctas(t.device, m) // 2, (m + 255) // 256)  # CTA pairs, 256 positions each
    if not _call(("out", c, residual is not None), fn, t.t().unsqueeze(-1), x_in.unsqueeze(-1), w1.unsqueeze(-1),
                 w_g.unsqueeze(-1), res.unsqueeze(-1), out.unsqueeze(-1), mean, rstd, u, v, 2 * pairs):
        return None
    return out


def _call(key, fn, *args) -> bool:
    """Run a loaded kernel; an argument mismatch (raised before launch) disables that kernel."""
    try:
        fn(*args)
    except (TypeError, ValueError) as ex:
        logger.warning("Triangle multiplication kernel %s rejected its arguments, disabled: %s", key, ex)
        _fns.pop(key, None)
        _failed.add(key)
        return False
    return True


def _ctas(device, m: int) -> int:
    dev = device.index if device.index is not None else torch.cuda.current_device()
    if dev not in _num_sms:
        _num_sms[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return min(_num_sms[dev], (m + 127) // 128)


def interleave_128(in_weight: torch.Tensor) -> torch.Tensor:
    """[g; p] as [2R, K] -> rows interleaved in blocks of 128: [g0; p0; g1; p1; ...]."""
    r = in_weight.shape[0] // 2
    g = in_weight[:r].view(r // BLOCK, BLOCK, -1)
    p = in_weight[r:].view(r // BLOCK, BLOCK, -1)
    return torch.stack([g, p], 1).reshape(2 * r, -1).contiguous()


def projection_supported(x: torch.Tensor, c: int) -> bool:
    return (
        _PROJ_ENABLED
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and c % 64 == 0
        and x.shape[0] % 8 == 0
        and 2 * x.numel() < 2**31  # 32-bit offsets in the kernel ([2C, M] output)
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and ("proj", c) not in _failed
    )


def gated_projection(x: torch.Tensor, w_il: torch.Tensor, mask: torch.Tensor):
    """x: [M, C] BF16 contiguous; w_il = interleave_128([g; p]) as [4C, C]; mask: [M] BF16.
    Returns sigmoid(x g^T) * (x p^T) * mask as [2C, M], or None where the kernel does not apply."""
    m, c = x.shape
    if not projection_supported(x, c) or w_il.shape != (4 * c, c):
        return None
    if any(a.dtype != torch.bfloat16 or not a.is_contiguous() or a.data_ptr() % 32 for a in (x, w_il, mask)):
        return None
    fn = _load(("proj", c))
    if fn is None:
        return None
    ab = torch.empty(2 * c, m, device=x.device, dtype=x.dtype)
    pairs = min(_ctas(x.device, m) // 2, (m + 255) // 256)  # CTA pairs, 256 positions each
    if not _call(("proj", c), fn, x.unsqueeze(-1), w_il.unsqueeze(-1), mask, ab.t(), 2 * pairs):
        return None
    return ab
