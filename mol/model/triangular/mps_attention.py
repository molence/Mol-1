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

"""Inference-only triangle attention for Apple MPS."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Final, Sequence

import torch
import torch.nn.functional as F

_METAL_SOURCE: Final[Path] = Path(__file__).parent / "_attention_shader.metal"
_HEAD_DIM: Final[int] = 32
_ROWS_PER_THREADGROUP: Final[int] = 4
_SDPA_MAX_SCORE_ELEMENTS: Final[int] = 1 << 28


@functools.lru_cache(maxsize=1)
def _shader_library():
    compile_shader = getattr(torch.mps, "compile_shader", None)
    if compile_shader is None:
        raise RuntimeError("triangle_attention='metal' requires torch.mps.compile_shader")
    return compile_shader(_METAL_SOURCE.read_text())


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"triangle_attention='metal' {message}")


def mps_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Attention over [I,H,N,32] inputs, returned as [I,Q,H,32].

    ``biases`` is ``[triangle_bias]`` for unmasked inputs, which use fused SDPA,
    or ``[mask_bias, triangle_bias]``, which uses the custom Metal kernel.
    """
    _check(q.device.type == "mps", "requires MPS tensors")
    _check(
        not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, *biases))),
        "supports inference only",
    )
    _check(
        all(t.dtype == torch.float32 for t in (q, k, v, *biases)),
        "currently requires FP32",
    )
    _check(len(biases) in (1, 2), "requires one or two biases")
    _check(
        q.ndim == 4 and q.shape[-1] == _HEAD_DIM and k.shape == v.shape,
        f"requires [I,H,N,{_HEAD_DIM}] tensors",
    )

    triangle_bias = biases[-1]
    outer, heads, query_length, _ = q.shape
    if len(biases) == 1:
        # SDPA materializes the [I,H,Q,K] scores, so bound them per call.
        rows = max(1, _SDPA_MAX_SCORE_ELEMENTS // (heads * query_length * k.shape[-2]))
        out = torch.empty_like(q)
        for start in range(0, outer, rows):
            window = slice(start, start + rows)
            out[window] = F.scaled_dot_product_attention(
                q[window], k[window], v[window], attn_mask=triangle_bias, scale=1.0
            )
        return out.transpose(1, 2)

    key_length = k.shape[-2]
    mask_bias = biases[0]
    _check(
        mask_bias.shape == (outer, 1, 1, key_length)
        and triangle_bias.shape == (1, heads, query_length, key_length),
        "received incompatible bias shapes",
    )
    # The shader reads the physical [I,N,H,D] layout of the projected views.
    _check(
        q.stride() == (query_length * heads * _HEAD_DIM, _HEAD_DIM, heads * _HEAD_DIM, 1)
        and k.stride() == v.stride() == (key_length * heads * _HEAD_DIM, _HEAD_DIM, heads * _HEAD_DIM, 1),
        "requires Q/K/V projected-view strides",
    )

    out = torch.empty((outer, query_length, heads, _HEAD_DIM), dtype=q.dtype, device=q.device)
    threadgroups = math.ceil(outer * heads * query_length / _ROWS_PER_THREADGROUP)
    _shader_library().triangle_attention_d32(
        q,
        k,
        v,
        mask_bias.contiguous(),
        triangle_bias.contiguous(),
        out,
        outer,
        heads,
        query_length,
        key_length,
        threads=(threadgroups * 32 * _ROWS_PER_THREADGROUP,),
        group_size=(32 * _ROWS_PER_THREADGROUP,),
    )
    return out
