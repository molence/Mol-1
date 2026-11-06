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
are rounded to the activation dtype before gating. On Blackwell, the input
projection and its gating run as one persistent Triton GEMM instead
(gated_projection), so the [4C, M] projection never reaches memory.
"""

import os

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


# MOL_GATED_PROJECTION=0 keeps the input projection on cuBLAS + _gate_in_kernel.
_GATED_PROJECTION = os.getenv("MOL_GATED_PROJECTION", "1") != "0"
GATE_BLOCK = 64
_GP_BLOCK_N = 256
_GP_BLOCK_K = 64


@triton.jit
def _gated_projection_kernel(
    w_desc, x_desc, y_desc, MASK, R, M,
    K: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr, GROUP: tl.constexpr,
):
    # Y[r, m] = sigmoid(G[r, m]) * P[r, m] * MASK[m] with [G; P] = W @ X^T, W's
    # rows interleaved in blocks of BLOCK_R: [G block; P block] per output block.
    num_r = R // BLOCK_R
    num_n = tl.cdiv(M, BLOCK_N)
    num_tiles = num_r * num_n
    for tile in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True, warp_specialize=True):
        # All channel blocks of a group of position blocks run together (X tiles reused from L2).
        first_n = tile // (GROUP * num_r) * GROUP
        gsize = tl.minimum(num_n - first_n, GROUP)
        pid_n = first_n + (tile % (GROUP * num_r)) % gsize
        pid_r = (tile % (GROUP * num_r)) // gsize
        acc = tl.zeros((2 * BLOCK_R, BLOCK_N), tl.float32)
        for k in range(0, K, BLOCK_K):
            acc = tl.dot(w_desc.load([pid_r * 2 * BLOCK_R, k]), x_desc.load([pid_n * BLOCK_N, k]).T, acc)
        g, p = tl.split(tl.permute(tl.reshape(acc, (2, BLOCK_R, BLOCK_N)), (1, 2, 0)))
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        pm = tl.load(MASK + offs_n, mask=offs_n < M, other=0.0).to(tl.float32)
        y_desc.store([pid_r * BLOCK_R, pid_n * BLOCK_N], (tl.sigmoid(g) * p * pm[None, :]).to(y_desc.dtype))


def interleave_gate_weight(in_weight: torch.Tensor) -> torch.Tensor:
    """[g; p] as [2R, K] -> rows interleaved in GATE_BLOCK blocks: [g0; p0; g1; p1; ...]."""
    r = in_weight.shape[0] // 2
    g, p = in_weight[:r].view(r // GATE_BLOCK, GATE_BLOCK, -1), in_weight[r:].view(r // GATE_BLOCK, GATE_BLOCK, -1)
    return torch.stack([g, p], 1).reshape(2 * r, -1).contiguous()


def gated_projection_ok(x: torch.Tensor, in_weight: torch.Tensor) -> bool:
    return (
        _GATED_PROJECTION
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.shape[-1] % _GP_BLOCK_K == 0
        and (in_weight.shape[0] // 2) % GATE_BLOCK == 0
        and torch.cuda.get_device_capability(x.device)[0] >= 10
    )


def gated_projection(x: torch.Tensor, w_interleaved: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [M, K] contiguous; w_interleaved from interleave_gate_weight ([2R, K]);
    mask: [M]. Returns sigmoid(g) * p * mask as [R, M], with [g; p] = W @ x^T."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, k = x.shape
    r = w_interleaved.shape[0] // 2
    y = torch.empty((r, m), device=x.device, dtype=x.dtype)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    _gated_projection_kernel[(num_sms,)](
        TensorDescriptor.from_tensor(w_interleaved, [2 * GATE_BLOCK, _GP_BLOCK_K]),
        TensorDescriptor.from_tensor(x, [_GP_BLOCK_N, _GP_BLOCK_K]),
        TensorDescriptor.from_tensor(y, [GATE_BLOCK, _GP_BLOCK_N]),
        mask, r, m, K=k, BLOCK_R=GATE_BLOCK, BLOCK_N=_GP_BLOCK_N, BLOCK_K=_GP_BLOCK_K,
        NUM_SMS=num_sms, GROUP=8, num_warps=8, num_stages=3,
    )
    return y


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
    in_weight_interleaved: torch.Tensor = None,
) -> torch.Tensor:
    """x: [N, N, C]; mask: broadcastable to [N, N]; in_weight: [g_in; p_in] as [4C, C]
    (weights already in x.dtype); in_weight_interleaved: interleave_gate_weight(in_weight),
    enables gated_projection. Returns the update, plus ``residual`` if given."""
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose

    n1, n2, c = x.shape
    m = n1 * n2
    x_in = layer_norm_transpose(x[None], norm_in_weight, norm_in_bias, eps=eps, layout="bijd->bijd")
    x_in = x_in.view(m, c)

    # [4C, M] = [g_in; p_in] @ x_in^T, then sigmoid(g) * p * mask -> [2C, M] = [a; b]
    pair_mask = mask.reshape(m).to(x.dtype).contiguous()
    if in_weight_interleaved is not None and gated_projection_ok(x_in, in_weight):
        ab = gated_projection(x_in, in_weight_interleaved, pair_mask)
    else:
        g_p = torch.mm(in_weight, x_in.t())
        ab = torch.empty((2 * c, m), device=x.device, dtype=x.dtype)
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
