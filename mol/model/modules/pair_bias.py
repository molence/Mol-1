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

"""Fused LayerNorm + per-head projection of a pair tensor into attention bias.

``linear(layer_norm(z))`` permuted to heads-first reads and writes the full
pair tensor once for the LayerNorm and again for the projection, then copies
the head-last result into a head-first layout. This computes the per-head
bias in one pass over ``z`` and writes ``[H, N_q, N_k]`` directly.
Inference only.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _ln_head_proj_kernel(
    Z, LNW, LNB, W, OUT, n_rows, n_k, stride_i, stride_j, eps,
    C: tl.constexpr, H: tl.constexpr, HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr,
    BLOCK_C: tl.constexpr, ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < n_rows
    col_mask = cols < C
    mask = row_mask[:, None] & col_mask[None, :]
    r64 = rows.to(tl.int64)
    z_off = (r64 // n_k) * stride_i + (r64 % n_k) * stride_j
    x = tl.load(Z + z_off[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    xc = tl.where(mask, x - mean[:, None], 0.0)
    xn = xc / tl.sqrt(tl.sum(xc * xc, axis=1) / C + eps)[:, None]
    if HAS_LNW:
        xn = xn * tl.load(LNW + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
    if HAS_LNB:
        xn = xn + tl.load(LNB + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
    heads = tl.arange(0, H)
    # W: [H, C] -> load as [BLOCK_C, H]
    w = tl.load(W + heads[None, :] * C + cols[:, None], mask=col_mask[:, None], other=0.0)
    # Like the unfused path, the normalized row enters the projection in the
    # weight dtype (BF16 under autocast), with FP32 accumulation.
    y = tl.dot(xn.to(w.dtype), w)  # [ROWS, H]
    tl.store(OUT + heads[None, :].to(tl.int64) * n_rows + rows[:, None], y.to(OUT.dtype.element_ty),
             mask=row_mask[:, None])


def layer_norm_head_projection(
    z: torch.Tensor,
    ln_weight: Optional[torch.Tensor],
    ln_bias: Optional[torch.Tensor],
    proj_weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
    rows: int = 16,
    num_warps: int = 4,
) -> torch.Tensor:
    """z: [..., N_q, N_k, C] -> bias [..., H, N_q, N_k] (leading dims must be empty)."""
    *lead, n_q, n_k, c = z.shape
    assert not lead, "only unbatched pair tensors"
    h = proj_weight.shape[0]
    if z.stride(-1) != 1:
        z = z.contiguous()
    n_rows = n_q * n_k
    out = torch.empty((h, n_q, n_k), device=z.device, dtype=out_dtype)
    # LayerNorm affine params are applied in the activation dtype, as in FusedLayerNorm.
    lnw = ln_weight.to(z.dtype) if ln_weight is not None else None
    lnb = ln_bias.to(z.dtype) if ln_bias is not None else None
    _ln_head_proj_kernel[(triton.cdiv(n_rows, rows),)](
        z, z if lnw is None else lnw, z if lnb is None else lnb,
        proj_weight.to(out_dtype).contiguous(), out, n_rows, n_k, z.stride(-3), z.stride(-2), eps,
        C=c, H=h, HAS_LNW=lnw is not None, HAS_LNB=lnb is not None,
        BLOCK_C=triton.next_power_of_2(c), ROWS=rows, num_warps=num_warps,
    )
    return out
