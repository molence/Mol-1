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

# MOL_TF32_ATTENTION=0 disables both kernels here (back to the PyTorch paths).
_ENABLED = os.getenv("MOL_TF32_ATTENTION", "1") != "0"
# MOL_TMA_ATTENTION=0 keeps the plain-load kernel on Blackwell.
_TMA = os.getenv("MOL_TMA_ATTENTION", "1") != "0"


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


@triton.jit
def _attn_fwd_tma_kernel(
    q_desc, k_desc, v_desc, b_desc, O, so0, so1, som, G, sg0, sg1, sgm,
    H, M, N,
    D: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr, HAS_GATE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # As _attn_fwd_kernel, with q/k/v/bias tiles fetched by TMA and the key
    # loop software-pipelined: the TMA box zero-fills past the tensor edges,
    # so the D padding and the last key block need no masked loads. The
    # kernel is latency-bound (one row of the softmax per thread); the
    # pipelined loads take about 30% off the plain kernel on GB300. With
    # HAS_GATE the output is multiplied by sigmoid(G) (the attention module's
    # output gate) before it is stored.
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    z0 = pid_z // H
    z1 = pid_z % H
    m0 = pid_m * BLOCK_M
    q = q_desc.load([z0, m0, z1, 0]).reshape(BLOCK_M, BLOCK_D) * 1.4426950408889634
    bz0 = z0 if B0 else 0
    bz1 = z1 if B1 else 0
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in tl.range(0, N, BLOCK_N, num_stages=3):
        k = k_desc.load([z0, start, z1, 0]).reshape(BLOCK_N, BLOCK_D)
        s = tl.dot(q, k.T, input_precision="tf32")
        s += b_desc.load([bz0, bz1, m0, start]).reshape(BLOCK_M, BLOCK_N) * 1.4426950408889634
        s = tl.where((start + offs_n < N)[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = v_desc.load([z0, start, z1, 0]).reshape(BLOCK_N, BLOCK_D)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="tf32")
        m_i = m_new
    o = acc / l_i[:, None]
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m < M)[:, None] & (offs_d < D)[None, :]
    if HAS_GATE:
        g = tl.load(G + z0.to(tl.int64) * sg0 + z1 * sg1 + offs_m[:, None] * sgm + offs_d[None, :], mask=mask, other=0.0)
        o = o * tl.sigmoid(g)
    tl.store(O + z0.to(tl.int64) * so0 + z1 * so1 + offs_m[:, None] * som + offs_d[None, :], o, mask=mask)


_TMA_BLOCK_M, _TMA_BLOCK_N = 128, 32


def _tma_ok(t: torch.Tensor) -> bool:
    # TMA: 16-byte aligned base and strides, unit stride last, no broadcast dims
    return (
        t.data_ptr() % 16 == 0
        and t.stride(-1) == 1
        and all(s % 4 == 0 and s > 0 for s in t.stride()[:-1])
    )


def _tma_attention(q4, k4, v4, b, lead, m, n, d, gate=None, out=None) -> Optional[torch.Tensor]:
    """The TMA kernel when the layout allows it (Blackwell, d <= 64, a bias
    whose leading dims are 1 or match), else None. gate/out: optional
    [*lead, m, d] views (any strides, unit last) for the output gate and the
    output."""
    if torch.cuda.get_device_capability(q4.device)[0] < 10 or d > 64 or b is None:
        return None
    b4 = _as_4d(b)
    if b4.shape[-2:] != (m, n) or any(bs not in (1, ls) for bs, ls in zip(b4.shape[:2], lead)):
        return None
    if q4.shape[:2] != lead or k4.shape[:2] != lead or v4.shape[:2] != lead:
        return None
    if not all(_tma_ok(t) for t in (q4, k4, v4, b4)):
        return None
    from triton.tools.tensor_descriptor import TensorDescriptor

    bm, bn, bd = _TMA_BLOCK_M, _TMA_BLOCK_N, 64
    o = torch.empty((*lead, m, d), device=q4.device, dtype=q4.dtype) if out is None else out
    g = o if gate is None else gate
    _attn_fwd_tma_kernel[(triton.cdiv(m, bm), lead[0] * lead[1])](
        # [S, M, H, D] order (tokens outside heads), as the projections lay
        # them out; with heads outside tokens the kernel is 1.6x slower.
        TensorDescriptor.from_tensor(q4.transpose(1, 2), [1, bm, 1, bd]),
        TensorDescriptor.from_tensor(k4.transpose(1, 2), [1, bn, 1, bd]),
        TensorDescriptor.from_tensor(v4.transpose(1, 2), [1, bn, 1, bd]),
        TensorDescriptor.from_tensor(b4, [1, 1, bm, bn]),
        o, o.stride(0), o.stride(1), o.stride(2),
        g, g.stride(0), g.stride(1), g.stride(2),
        lead[1], m, n,
        D=d, B0=b4.shape[0] != 1, B1=b4.shape[1] != 1, HAS_GATE=gate is not None,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=bd,
        num_warps=4,
    )
    return o


def gated_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, gate: torch.Tensor, out: torch.Tensor
) -> bool:
    """out = softmax(q k^T + bias) v * sigmoid(gate), all [S, H, M, D] views
    (bias [1 or S, 1 or H, M, N]), in one TMA kernel. Returns False (nothing
    written) where that kernel does not apply."""
    if not (_ENABLED and _TMA and tf32_attention_ok(q, k, v, bias) and q.dim() == 4):
        return False
    s_, h, m, d = q.shape
    if k.shape[:2] != (s_, h) or gate.shape != q.shape or out.shape != q.shape:
        return False
    if gate.stride(-1) != 1 or out.stride(-1) != 1:
        return False
    return _tma_attention(q, k, v, bias, (s_, h), m, k.shape[-2], d, gate=gate, out=out) is not None


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
    if _TMA:
        o = _tma_attention(q4, k4, v4, bias, lead, m, n, d)
        if o is not None:
            return o.view(out_shape)
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


@triton.jit
def _local_attn_kernel(
    Q, K, V, B, O,
    sq0, sq1, sqn, sk0, sk1, skn, sv0, sv1, svn, sb0, sb1, sbt, sbm, sbn, so0, so1, son,
    H, N,
    D: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr, PAD_LEFT: tl.constexpr,
    HAS_BIAS: tl.constexpr, PREC: tl.constexpr, BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    pid_z = tl.program_id(1)
    z0 = (pid_z // H).to(tl.int64)
    z1 = (pid_z % H).to(tl.int64)
    rows = t * NQ + tl.arange(0, NQ)
    keys = t * NQ - PAD_LEFT + tl.arange(0, NK)
    offs_d = tl.arange(0, BLOCK_D)
    r_ok = rows < N
    k_ok = (keys >= 0) & (keys < N)
    d_ok = offs_d < D
    q = tl.load(Q + z0 * sq0 + z1 * sq1 + rows[:, None] * sqn + offs_d[None, :],
                mask=r_ok[:, None] & d_ok[None, :], other=0.0)
    kt = tl.load(K + z0 * sk0 + z1 * sk1 + keys[None, :] * skn + offs_d[:, None],
                 mask=k_ok[None, :] & d_ok[:, None], other=0.0)
    s = tl.dot(q, kt, input_precision=PREC)
    if HAS_BIAS:
        m_idx = tl.arange(0, NQ)
        n_idx = tl.arange(0, NK)
        s += tl.load(B + z0 * sb0 + z1 * sb1 + t * sbt + m_idx[:, None] * sbm + n_idx[None, :] * sbn)
    s = tl.where(k_ok[None, :], s, float("-inf"))
    p = tl.exp(s - tl.max(s, 1)[:, None])
    p = p / tl.sum(p, 1)[:, None]
    v = tl.load(V + z0 * sv0 + z1 * sv1 + keys[:, None] * svn + offs_d[None, :],
                mask=k_ok[:, None] & d_ok[None, :], other=0.0)
    o = tl.dot(p, v, input_precision=PREC)
    tl.store(O + z0 * so0 + z1 * so1 + rows[:, None] * son + offs_d[None, :], o,
             mask=r_ok[:, None] & d_ok[None, :])


def local_attention_ok(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    trunked_attn_bias: Optional[torch.Tensor],
    n_queries: int,
    n_keys: int,
) -> bool:
    return (
        _ENABLED
        and q.is_cuda
        and attn_bias is None
        and q.dtype == k.dtype == v.dtype == torch.float32
        and (trunked_attn_bias is None or trunked_attn_bias.dtype == torch.float32)
        # Triton's full-FP32 dot is slower than SDPA plus the window copies.
        and torch.backends.cuda.matmul.allow_tf32
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled("cuda")
        and q.shape == k.shape == v.shape
        and 2 <= q.dim() <= 4
        and q.shape[-1] <= 128
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and n_queries in (16, 32, 64)
        and n_keys in (64, 128)
        and (n_keys - n_queries) % 2 == 0
        and (trunked_attn_bias is None or trunked_attn_bias.dim() <= q.dim() + 1)
    )


def local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    trunked_attn_bias: Optional[torch.Tensor],
    n_queries: int,
    n_keys: int,
) -> torch.Tensor:
    """Sequence-local attention, as ``primitives._local_attention`` with no
    dense bias: query block t (rows t*n_queries...) attends to keys
    t*n_queries - (n_keys - n_queries)/2 ... (+n_keys), out-of-range keys
    masked. Reads q/k/v in place instead of materializing the key windows.

    q/k/v: [..., N, D] (up to two leading dims); trunked_attn_bias
    broadcastable to [..., n_trunks, n_queries, n_keys]. Returns [..., N, D].
    Dots in TF32 (local_attention_ok requires TF32 matmuls to be enabled;
    without it the dots run in full FP32).
    """
    out_shape = q.shape
    q4, k4, v4 = _as_4d(q), _as_4d(k), _as_4d(v)
    lead = q4.shape[:2]
    n, d = q4.shape[-2:]
    n_trunks = triton.cdiv(n, n_queries)
    o = torch.empty((*lead, n, d), device=q.device, dtype=q.dtype)
    if trunked_attn_bias is not None:
        b = trunked_attn_bias
        while b.dim() < 5:
            b = b.unsqueeze(0)
        b = b.expand(*lead, n_trunks, n_queries, n_keys)
        b_strides = b.stride()
    else:
        b, b_strides = q4, (0, 0, 0, 0, 0)
    prec = "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"
    _local_attn_kernel[(n_trunks, lead[0] * lead[1])](
        q4, k4, v4, b, o,
        q4.stride(0), q4.stride(1), q4.stride(2),
        k4.stride(0), k4.stride(1), k4.stride(2),
        v4.stride(0), v4.stride(1), v4.stride(2),
        *b_strides,
        o.stride(0), o.stride(1), o.stride(2),
        lead[1], n,
        D=d, NQ=n_queries, NK=n_keys, PAD_LEFT=(n_keys - n_queries) // 2,
        HAS_BIAS=trunked_attn_bias is not None, PREC=prec,
        BLOCK_D=max(16, triton.next_power_of_2(d)), num_warps=4,
    )
    return o.view(out_shape)
