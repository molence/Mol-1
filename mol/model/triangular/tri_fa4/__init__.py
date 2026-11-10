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

"""Gated triangle attention on FlashAttention-4 (CuTe DSL), Blackwell, BF16, head dim 32.

out = softmax(scale * q k^T + bias) v * sigmoid(gate), keys past ``n_keys``
masked: the FA4 SM100 forward kernel with the pair bias added by the tensor
cores (bias tiles by TMA, S = bias @ I / scale before q k^T accumulates) and the
gate applied in the correction epilogue (see flash_fwd_sm100.py). On GB300 at
N=648, 10 heads this takes 0.91 ms against 1.13 ms for the CUTLASS kernel in
``tri_fmha``, which remains the fallback.

Needs flash-attn-4 (shipped in NVIDIA's PyTorch containers) and the CuTe DSL
(``pip install "nvidia-cutlass-dsl[cu13]==4.6.0"``, or a directory holding it in
``MOL_CUTE_DSL_DIR``). The kernel is compiled once (about 30 s) and linked into a
shared library next to the other extension builds; later processes load that
through tvm_ffi alone, without importing the DSL. ``MOL_TRI_FA4=0`` disables it.

``load_variant`` builds the same kernel for other dtypes, head dims and tile
configs (the diffusion token attention, ``modules/fa4_attention.py``).
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

_ENABLED = os.getenv("MOL_TRI_FA4", "1") != "0"
_HERE = os.path.dirname(os.path.abspath(__file__))

_fn = None
_failed = False
_seqused = {}
_variants = {}

# The triangle attention build: BF16, head dim 32.
_TRI = ("tri_fa4", "bfloat16", 32, (("m_block_size", 128), ("n_block_size", 128), ("q_stage", 2)))


def _library_path(name, dtype, head_dim, cfg):
    from mol.model.layer_norm.torch_ext_compile import _resolve_build_directory

    h = hashlib.sha256()
    for dist in ("nvidia-cutlass-dsl", "flash-attn-4", "apache-tvm-ffi"):
        try:
            h.update(f"{dist}={importlib.metadata.version(dist)}".encode())
        except importlib.metadata.PackageNotFoundError:
            pass
    with open(os.path.join(_HERE, "flash_fwd_sm100.py"), "rb") as f:
        h.update(f.read())
    if name != _TRI[0]:  # (keeps the triangle build's existing file name)
        h.update(repr((dtype, head_dim, cfg)).encode())
    major, minor = torch.cuda.get_device_capability()
    return _resolve_build_directory("mol_tri_fa4", None) / f"{name}_sm{major}{minor}_{h.hexdigest()[:16]}.so"


def _build(so, name, dtype, head_dim, cfg):
    """Compile the kernel with the CuTe DSL and link it into `so`, exported as `name`."""
    import cutlass.cute as cute
    from cutlass.runtime import find_runtime_libraries
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor
    from flash_attn.cute.utils import AuxData

    from .flash_fwd_sm100 import FlashAttentionForwardSm100

    b, n, h, d = 1, 256, 2, head_dim
    dt = getattr(torch, dtype)
    q, k, v, g, o = (torch.empty(b, n, h, d, device="cuda", dtype=dt) for _ in range(5))
    bias = torch.empty(h, n, n, device="cuda", dtype=dt)
    seqused_k = torch.full((b,), n, dtype=torch.int32, device="cuda")
    fa = FlashAttentionForwardSm100(
        d, d, qhead_per_kvhead=1, is_causal=False, is_local=False, is_split_kv=False,
        pack_gqa=False, is_persistent=True, has_bias=True, has_gate=True, **dict(cfg),
    )
    fn = cute.compile(
        fa, *(to_cute_tensor(t) for t in (q, k, v, o)), None, 1.0, None, None, None,
        to_cute_tensor(seqused_k, assumed_align=4, leading_dim=0), None, None, None, None, None, None,
        AuxData(None, None), to_cute_tensor(bias), to_cute_tensor(g),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    obj = f"{so}.{os.getpid()}.o"
    fn.export_to_c(object_file_path=obj, function_name=name)
    runtime = [p for p in find_runtime_libraries(enable_tvm_ffi=False) if p.endswith("libcute_dsl_runtime.so")]
    lib_dir = os.path.dirname(runtime[0])
    tmp = f"{so}.{os.getpid()}.tmp"
    subprocess.run(
        ["g++", "-shared", "-o", tmp, obj, f"-L{lib_dir}", "-lcute_dsl_runtime", f"-Wl,-rpath,{lib_dir}"],
        check=True, capture_output=True,
    )
    os.replace(tmp, so)
    os.remove(obj)


def load_variant(name, dtype, head_dim, cfg):
    """The FA4 forward with bias and gate for (dtype, head_dim, tile config
    `cfg`, a tuple of (key, value) pairs), compiled once and cached on disk;
    None (logged) without the CuTe DSL / FA4 or if the build fails."""
    key = (name, dtype, head_dim, cfg)
    if key in _variants:
        return _variants[key]
    _variants[key] = None
    dsl_dir = os.environ.get("MOL_CUTE_DSL_DIR")
    if dsl_dir:
        site.addsitedir(dsl_dir)
    try:
        so = _library_path(name, dtype, head_dim, cfg)
        if not so.exists():
            with open(f"{so}.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)  # one build across processes
                if not so.exists():
                    logger.info("Compiling the FA4 kernel %s (once, about 30 s)", name)
                    _build(so, name, dtype, head_dim, cfg)
        import tvm_ffi

        _variants[key] = tvm_ffi.load_module(str(so))[name]
    except Exception as ex:  # no CuTe DSL / FA4, or a failed build: callers fall back
        msg = next((line for line in str(ex).splitlines() if line.strip()), type(ex).__name__)
        logger.info("FA4 attention (%s) unavailable: %s", name, msg)
    return _variants[key]


def _load():
    global _fn, _failed
    if _fn is not None or _failed:
        return _fn
    _fn = load_variant(*_TRI)
    _failed = _fn is None
    return _fn


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and t.stride(-1) == 1 and all(s % 8 == 0 for s in t.stride()[:-1])


def supported(q: torch.Tensor, k: torch.Tensor, gate: torch.Tensor, out: torch.Tensor) -> bool:
    return (
        _ENABLED
        and not _failed
        and q.is_cuda
        and q.dtype == k.dtype == gate.dtype == out.dtype == torch.bfloat16
        and q.dim() == 4
        and q.shape[-1] == 32
        and not torch.is_grad_enabled()
        and torch.cuda.get_device_capability(q.device)[0] == 10
    )


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    n_keys: int,
    scale: float,
) -> bool:
    """out = softmax(scale * q k^T + bias) v * sigmoid(gate) over the first
    n_keys keys. q, gate, out: [B, H, Q, 32] views; k, v: [B, H, K, 32] views
    (unit stride last, 16-byte aligned strides); bias: [H, Q, K] (made
    contiguous here). Returns False, with nothing written, where the kernel
    does not apply."""
    if not supported(q, k, gate, out):
        return False
    bias = bias.contiguous()
    if not all(_aligned(t) for t in (q, k, v, bias, gate, out)):
        return False
    if bias.shape != (q.shape[1], q.shape[2], k.shape[2]) or bias.dtype != torch.bfloat16:
        return False
    fn = _load()
    if fn is None:
        return False
    b = q.shape[0]
    key = (b, int(n_keys), q.device)
    if key not in _seqused:
        _seqused[key] = torch.full((b,), int(n_keys), dtype=torch.int32, device=q.device)
    qt, kt, vt, gt, ot = (t.transpose(1, 2) for t in (q, k, v, gate, out))  # [B, S, H, D]
    fn(qt, kt, vt, ot, None, float(scale), None, None, None, _seqused[key], None, None, None, None, None, None,
       (None, None), bias, gt)
    return True
