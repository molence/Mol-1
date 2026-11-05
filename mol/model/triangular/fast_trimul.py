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

"""Inference triangle multiplication: cuEquivariance's pipeline with cuBLAS GEMMs.

Same computation as ``cuequivariance_torch.triangle_multiplicative_update``
(LayerNorm, sigmoid-gated projections with the pair mask, triangular einsum,
output LayerNorm, sigmoid-gated output projection), but the four projections
run as cuBLAS GEMMs, which reach about 2.5x the throughput of cuEquivariance's
Triton gated GEMM at Mol-1's pair widths on Blackwell, with the gating (and the
residual add of the caller) applied in small Triton passes. The projections
are rounded to the activation dtype before gating.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_in_kernel(G, MASK, OUT, M, R: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_M: tl.constexpr):
    # OUT[r, m] = sigmoid(G[r, m]) * G[R + r, m] * MASK[m], r < R
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = (r < R)[:, None] & (m < M)[None, :]
    off = r.to(tl.int64)[:, None] * M + m[None, :]
    g = tl.load(G + off, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(G + R * M + off, mask=mask, other=0.0).to(tl.float32)
    pm = tl.load(MASK + m, mask=m < M, other=0.0).to(tl.float32)
    y = tl.sigmoid(g) * p * pm[None, :]
    tl.store(OUT + off, y.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _gate_out_kernel(G, P, RES, OUT, M, C: tl.constexpr, HAS_RES: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    # OUT[m, c] = sigmoid(G[m, c]) * P[m, c] (+ RES[m, c])
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (m < M)[:, None] & (c < C)[None, :]
    off = m.to(tl.int64)[:, None] * C + c[None, :]
    y = tl.sigmoid(tl.load(G + off, mask=mask, other=0.0).to(tl.float32)) * tl.load(
        P + off, mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_RES:
        y = y + tl.load(RES + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + off, y.to(OUT.dtype.element_ty), mask=mask)


def triangle_multiplicative_update(
    x: torch.Tensor,
    direction: str,
    mask: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor = None,
) -> torch.Tensor:
    """x: [N, N, C]; mask: broadcastable to [N, N]; in_weight: [g_in; p_in] as [4C, C]
    (weights already in x.dtype). Returns the update, plus ``residual`` if given."""
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose

    n1, n2, c = x.shape
    m = n1 * n2
    x_in = layer_norm_transpose(x[None], norm_in_weight, norm_in_bias, eps=eps, layout="bijd->bijd")
    x_in = x_in.view(m, c)

    # [4C, M] = [g_in; p_in] @ x_in^T, then sigmoid(g) * p * mask -> [2C, M] = [a; b]
    g_p = torch.mm(in_weight, x_in.t())
    ab = torch.empty((2 * c, m), device=x.device, dtype=x.dtype)
    pair_mask = mask.reshape(m).to(x.dtype).contiguous()
    _gate_in_kernel[(triton.cdiv(2 * c, 16), triton.cdiv(m, 256))](
        g_p, pair_mask, ab, m, R=2 * c, BLOCK_R=16, BLOCK_M=256, num_warps=4
    )
    del g_p
    a, b = (t.view(c, 1, n1, n2) for t in ab.chunk(2, dim=0))
    if direction == "outgoing":
        t = torch.einsum("dbik,dbjk->dbij", a, b)
    else:
        t = torch.einsum("dbki,dbkj->dbij", a, b)
    del ab, a, b
    x_out = layer_norm_transpose(t, norm_out_weight, norm_out_bias, eps=eps, layout="dbij->bijd")
    del t

    g = torch.mm(x_in, g_out_weight.t())
    p = torch.mm(x_out.view(m, c), p_out_weight.t())
    out = torch.empty((n1, n2, c), device=x.device, dtype=x.dtype)
    res = residual.contiguous() if residual is not None else out
    _gate_out_kernel[(triton.cdiv(m, 64), triton.cdiv(c, 64))](
        g, p, res, out, m, C=c, HAS_RES=residual is not None, BLOCK_M=64, BLOCK_C=64, num_warps=4
    )
    return out
