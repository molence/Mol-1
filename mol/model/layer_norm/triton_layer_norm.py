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

"""Inference-only Triton LayerNorm for large row counts (pair tensors).

The fast LayerNorm extension reaches only a fraction of HBM bandwidth on
N_token^2-row pair tensors in BF16 at Mol-1's widths (320/448); this kernel
processes several rows per program and runs close to bandwidth. It computes
in FP32 and, like the extension, applies the affine parameters cast to the
input dtype. Used by FusedLayerNorm for CUDA inputs without autograd and at
least ``MIN_ROWS`` rows; ``MOL_TRITON_LN=0`` disables it.
"""

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton ships with CUDA torch
    triton = None

MIN_ROWS = 65536


def enabled(x: torch.Tensor) -> bool:
    return (
        triton is not None
        and os.getenv("MOL_TRITON_LN", "1") != "0"
        and x.is_cuda
        and not torch.is_grad_enabled()
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and x.numel() // x.shape[-1] >= MIN_ROWS
        and 128 <= x.shape[-1] <= 4096
    )


if triton is not None:

    @triton.jit
    def _layer_norm_fwd(
        X, W, B, Y, n_rows, n_cols, eps,
        HAS_W: tl.constexpr, HAS_B: tl.constexpr,
        BLOCK_C: tl.constexpr, ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_C)
        row_mask = rows < n_rows
        col_mask = cols < n_cols
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None].to(tl.int64) * n_cols + cols[None, :]
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / n_cols
        xc = tl.where(mask, x - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=1) / n_cols + eps)
        y = xc * rstd[:, None]
        if HAS_W:
            y = y * tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
        if HAS_B:
            y = y + tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    rows_per_program: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    n_cols = x.shape[-1]
    x2 = x.contiguous().view(-1, n_cols)
    y = torch.empty_like(x2)
    w = weight.to(x.dtype).contiguous() if weight is not None else None
    b = bias.to(x.dtype).contiguous() if bias is not None else None
    block_c = triton.next_power_of_2(n_cols)
    # Tuned on GB300 for 64k-1M rows: wide rows want few rows per program.
    if x.dtype is torch.float32:
        rows, warps = 4, 4
    elif block_c > 256:
        rows, warps = 8, 4
    else:
        rows, warps = max(1, min(64, 8192 // block_c)), 8
    rows = rows_per_program or rows
    num_warps = num_warps or warps
    grid = (triton.cdiv(x2.shape[0], rows),)
    _layer_norm_fwd[grid](
        x2, x2 if w is None else w, x2 if b is None else b, y, x2.shape[0], n_cols, eps,
        HAS_W=w is not None, HAS_B=b is not None,
        BLOCK_C=block_c, ROWS=rows, num_warps=num_warps,
    )
    return y.view(x.shape)
