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
sigmoid, a strided multiply and a reshape copy. With ``transposed=True`` the
gate is read as, and the result written as, the [Q, N, H*D] layout, which
lets ending-node attention run on the untransposed pair tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_gate_kernel(
    O, G, Y, Q, g_n_stride, g_q_stride, y_n_stride, y_q_stride,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    h = c // D
    d = c % D
    mask = (q < Q)[:, None] & (c < H * D)[None, :]
    q64 = q.to(tl.int64)
    g = tl.load(G + (n * g_n_stride + q64 * g_q_stride)[:, None] + c[None, :], mask=mask, other=0.0)
    o_off = ((n * H + h[None, :]).to(tl.int64) * Q + q[:, None]) * D + d[None, :]
    o = tl.load(O + o_off, mask=mask, other=0.0)
    y = o.to(tl.float32) * tl.sigmoid(g.to(tl.float32))
    tl.store(Y + (n * y_n_stride + q64 * y_q_stride)[:, None] + c[None, :], y.to(Y.dtype.element_ty), mask=mask)


def sigmoid_gate_transpose(o: torch.Tensor, g: torch.Tensor, transposed: bool = False) -> torch.Tensor:
    """o: [N, H, Q, D]; g: [N, Q, H*D] with unit column stride. Returns [N, Q, H*D].

    transposed: g is given as [Q, N, H*D] and the result is returned as [Q, N, H*D].
    """
    n, h, q, d = o.shape
    o = o.contiguous()
    if transposed:
        g = g.transpose(0, 1)
    assert g.shape == (n, q, h * d) and g.stride(-1) == 1
    y = torch.empty((q, n, h * d) if transposed else (n, q, h * d), device=o.device, dtype=o.dtype)
    y_view = y.transpose(0, 1) if transposed else y
    block_q, block_c = 32, 64
    grid = (n, triton.cdiv(q, block_q), triton.cdiv(h * d, block_c))
    _sigmoid_gate_kernel[grid](
        o, g, y, q, g.stride(0), g.stride(1), y_view.stride(0), y_view.stride(1),
        H=h, D=d, BLOCK_Q=block_q, BLOCK_C=block_c, num_warps=4,
    )
    return y


@triton.jit
def _bias_heads_kernel(
    P, B, N, p_row, p_col, H: tl.constexpr, HP: tl.constexpr, TRANSPOSED: tl.constexpr,
    BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK_A + tl.arange(0, BLOCK_A)
    b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
    h = tl.arange(0, HP)
    if TRANSPOSED:
        row = b[None, :, None] * N + a[:, None, None]
    else:
        row = a[:, None, None] * N + b[None, :, None]
    m = (a[:, None, None] < N) & (b[None, :, None] < N) & (h[None, None, :] < H)
    x = tl.load(P + row.to(tl.int64) * p_row + h[None, None, :] * p_col, mask=m)
    out = B + h[None, None, :].to(tl.int64) * N * N + a[:, None, None] * N + b[None, :, None]
    tl.store(out, x, mask=m)


def bias_heads(proj_bias: torch.Tensor, transposed: bool = False) -> torch.Tensor:
    """[N, N, H] column slice of the projection -> contiguous [H, N, N] bias,
    out[h, a, b] = p[a, b, h] (``transposed``: p[b, a, h]). One pass at memory
    speed; ``permute(...).contiguous()`` takes 2-3x longer, the transposed case
    most."""
    n, n2, h = proj_bias.shape
    assert n == n2 and proj_bias.stride(0) == n * proj_bias.stride(1)
    out = torch.empty((h, n, n), device=proj_bias.device, dtype=proj_bias.dtype)
    grid = (triton.cdiv(n, 32), triton.cdiv(n, 64))
    _bias_heads_kernel[grid](
        proj_bias, out, n, proj_bias.stride(1), proj_bias.stride(2), H=h, HP=triton.next_power_of_2(h),
        TRANSPOSED=transposed, BLOCK_A=32, BLOCK_B=64, num_warps=4,
    )
    return out
