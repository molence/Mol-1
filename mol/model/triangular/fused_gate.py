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

"""Fused sigmoid gating + head transpose for triangle attention output.

The attention kernel returns ``o`` as [N, H, Q, D]; the gate ``g`` comes out
of the projection as [N, Q, H*D] (possibly a column slice of a wider
projection). This computes ``sigmoid(g) * o`` and writes it as a contiguous
[N, Q, H*D] tensor ready for the output projection, in one pass instead of a
sigmoid, a strided multiply and a reshape copy.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_gate_kernel(
    O, G, Y, Q, g_row_stride,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    h = c // D
    d = c % D
    mask = (q < Q)[:, None] & (c < H * D)[None, :]
    row = (n * Q + q).to(tl.int64)
    g = tl.load(G + row[:, None] * g_row_stride + c[None, :], mask=mask, other=0.0)
    o_off = ((n * H + h[None, :]).to(tl.int64) * Q + q[:, None]) * D + d[None, :]
    o = tl.load(O + o_off, mask=mask, other=0.0)
    y = o.to(tl.float32) * tl.sigmoid(g.to(tl.float32))
    tl.store(Y + row[:, None] * (H * D) + c[None, :], y.to(Y.dtype.element_ty), mask=mask)


def sigmoid_gate_transpose(o: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """o: [N, H, Q, D]; g: [N, Q, H*D] with unit column stride. Returns [N, Q, H*D]."""
    n, h, q, d = o.shape
    o = o.contiguous()
    assert g.shape == (n, q, h * d) and g.stride(-1) == 1 and g.stride(0) == q * g.stride(1)
    y = torch.empty((n, q, h * d), device=o.device, dtype=o.dtype)
    block_q, block_c = 32, 64
    grid = (n, triton.cdiv(q, block_q), triton.cdiv(h * d, block_c))
    _sigmoid_gate_kernel[grid](
        o, g, y, q, g.stride(1), H=h, D=d, BLOCK_Q=block_q, BLOCK_C=block_c, num_warps=4
    )
    return y
