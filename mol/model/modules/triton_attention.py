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

"""FP32 attention with an additive bias on TF32 tensor cores, for inference.

With FP32 operands (diffusion sampling and the confidence head run outside
autocast up to 2,560 tokens), ``F.scaled_dot_product_attention`` falls back to
a CUDA-core FP32 kernel. When TF32 matmuls are enabled (``--enable_tf32``),
this flash-attention kernel computes QK^T and PV in TF32 with FP32 softmax
and accumulation instead, the precision the surrounding matmuls already use.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

# MOL_TF32_ATTENTION=0 disables (back to scaled_dot_product_attention).
_ENABLED = os.getenv("MOL_TF32_ATTENTION", "1") != "0"


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, B, O,
    sq0, sq1, sqm, sk0, sk1, skn, sv0, sv1, svn, sb0, sb1, sbm, sbn, so0, so1, som,
    H, M, N,
    D: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    z0 = (pid_z // H).to(tl.int64)
    z1 = (pid_z % H).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_ok = offs_m < M
    d_ok = offs_d < D
    # Softmax in base 2: scale logits (and the bias) by log2(e).
    q = tl.load(
        Q + z0 * sq0 + z1 * sq1 + offs_m[:, None] * sqm + offs_d[None, :],
        mask=m_ok[:, None] & d_ok[None, :], other=0.0,
    ) * 1.4426950408889634
    k_base = K + z0 * sk0 + z1 * sk1
    v_base = V + z0 * sv0 + z1 * sv1
    b_base = B + z0 * sb0 + z1 * sb1 + offs_m[:, None] * sbm
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for start in range(0, N, BLOCK_N):
        n = start + offs_n
        n_ok = n < N
        # K loaded as [D, N]
        kt = tl.load(k_base + n[None, :] * skn + offs_d[:, None], mask=n_ok[None, :] & d_ok[:, None], other=0.0)
        s = tl.dot(q, kt, input_precision="tf32")
        if HAS_BIAS:
            s += tl.load(
                b_base + n[None, :] * sbn, mask=m_ok[:, None] & n_ok[None, :], other=0.0
            ) * 1.4426950408889634
        s = tl.where(n_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + n[:, None] * svn + offs_d[None, :], mask=n_ok[:, None] & d_ok[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="tf32")
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(
        O + z0 * so0 + z1 * so1 + offs_m[:, None] * som + offs_d[None, :],
        o, mask=m_ok[:, None] & d_ok[None, :],
    )


def _as_4d(x: torch.Tensor) -> torch.Tensor:
    while x.dim() < 4:
        x = x.unsqueeze(0)
    return x


def tf32_attention_ok(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: Optional[torch.Tensor]
) -> bool:
    return (
        _ENABLED
        and q.is_cuda
        and q.dtype == k.dtype == v.dtype == torch.float32
        and (bias is None or bias.dtype == torch.float32)
        and torch.backends.cuda.matmul.allow_tf32
        and not torch.is_grad_enabled()
        and 2 <= q.dim() <= 4
        and q.dim() == k.dim() == v.dim()
        and q.shape[-1] == k.shape[-1] == v.shape[-1] <= 128
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and (bias is None or bias.dim() <= q.dim())
        # Short query blocks (atom attention windows) are no faster than SDPA.
        and q.shape[-2] >= 128
    )


def tf32_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """softmax(q k^T + bias) v with scale 1. q: [..., M, D], k/v: [..., N, D],
    bias broadcastable to [..., M, N]; up to two leading dims."""
    out_shape = q.shape
    q4, k4, v4 = _as_4d(q), _as_4d(k), _as_4d(v)
    lead = torch.broadcast_shapes(q4.shape[:2], k4.shape[:2], v4.shape[:2])
    q4, k4, v4 = (t.expand(*lead, *t.shape[2:]) for t in (q4, k4, v4))
    m, d = q4.shape[-2:]
    n = k4.shape[-2]
    o = torch.empty((*lead, m, d), device=q.device, dtype=q.dtype)
    if bias is not None:
        b4 = _as_4d(bias).expand(*lead, m, n)
        b_strides = b4.stride()
    else:
        b4, b_strides = q4, (0, 0, 0, 0)
    block_d = max(16, triton.next_power_of_2(d))
    # Tuned on GB300 at [5, 16, 644, 48] (2.9x faster than SDPA's FP32 kernel).
    block_m, block_n = 128, 64
    grid = (triton.cdiv(m, block_m), lead[0] * lead[1])
    _attn_fwd_kernel[grid](
        q4, k4, v4, b4, o,
        q4.stride(0), q4.stride(1), q4.stride(2),
        k4.stride(0), k4.stride(1), k4.stride(2),
        v4.stride(0), v4.stride(1), v4.stride(2),
        b_strides[0], b_strides[1], b_strides[2], b_strides[3],
        o.stride(0), o.stride(1), o.stride(2),
        lead[1], m, n,
        D=d, HAS_BIAS=bias is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        num_warps=4, num_stages=1,
    )
    return o.view(out_shape)
