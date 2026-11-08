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

"""Gated triangle attention for Blackwell, BF16, head dim 32.

out = softmax(scale * q k^T + bias) v * sigmoid(gate), keys past ``n_keys``
masked, in one kernel: the CUTLASS Blackwell FMHA forward kernel
(examples/77_blackwell_fmha, CUTLASS 4.6) with two changes. The [H, Q, K]
pair bias, which every row of the batch shares, is loaded by TMA and added by
the tensor cores (S = bias_tile @ I + q k^T, as 128x16x16 MMAs against a
16x16 identity), so the softmax warps never touch it; and the output gate is
applied in the epilogue, which writes the result in the caller's layout. On
GB300 at N=648, 10 heads this takes 1.17 ms against 1.44 ms for
cuEquivariance's kernel plus the gating kernel.

The extension is compiled at first use against CUTLASS headers found in
``MOL_CUTLASS_DIR`` or the PyTorch source tree of NVIDIA's containers; without
them, or on other GPUs, ``attention`` returns False and callers fall back.
``MOL_TRI_FMHA=0`` disables it.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("MOL_TRI_FMHA", "1") != "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_CUTLASS_CANDIDATES = (
    "/opt/pytorch/pytorch/third_party/cutlass",
    "/usr/local/cutlass",
)

_ext = None
_failed = False


def _cutlass_dir() -> Optional[str]:
    for d in (os.environ.get("MOL_CUTLASS_DIR"),) + _CUTLASS_CANDIDATES:
        if d and os.path.isfile(os.path.join(d, "include", "cutlass", "cutlass.h")):
            return d
    return None


def _load(device: torch.device):
    global _ext, _failed
    if _ext is not None or _failed:
        return _ext
    _failed = True
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        return None
    cutlass = _cutlass_dir()
    if cutlass is None:
        logger.info("Triangle attention kernel disabled: CUTLASS headers not found (set MOL_CUTLASS_DIR)")
        return None
    from torch.utils.cpp_extension import load

    from mol.model.layer_norm.torch_ext_compile import _clear_stale_lock, _resolve_build_directory

    arch = f"{major}{minor}a"
    name = f"mol_tri_fmha_sm{arch}"
    build_directory = _resolve_build_directory(name, None)
    _clear_stale_lock(build_directory)
    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}a"
    try:
        _ext = load(
            name=name,
            sources=[os.path.join(_HERE, "csrc", "tri_fmha.cu")],
            extra_include_paths=[
                os.path.join(cutlass, "include"),
                os.path.join(cutlass, "tools", "util", "include"),
                os.path.join(_HERE, "csrc"),
            ],
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--use_fast_math", "-DNDEBUG"],
            build_directory=str(build_directory),
            verbose=False,
        )
        _failed = False
    except Exception as ex:  # a failed build leaves the fallback in place
        logger.warning("Triangle attention kernel unavailable, falling back: %s", str(ex).splitlines()[0])
    finally:
        if old_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch_list
    return _ext


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
        and q.shape[-2] >= 128
        and k.shape[-2] >= 128
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
    ext = _load(q.device)
    if ext is None:
        return False
    ext.fwd(q, k, v, bias, gate, out, int(n_keys), float(scale))
    return True
