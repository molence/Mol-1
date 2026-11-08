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

"""Inference fast path for DiffusionTransformer (Algorithm 23).

Every block conditions on the same single embedding s: two adaptive
LayerNorms (Algorithm 26) and two output gates project s, which does not
change from block to block. Per block that is six small GEMMs, two
LayerNorms of s and a dozen elementwise kernels; here the s projections of
a group of blocks are one LayerNorm of s and two wide GEMMs (the LayerNorm
weights folded into the GEMM weights), and per block what is left is

    x  = LN(a) * gate + shift                         (one kernel)
    qkvg = x W_qkvg + b                                (one GEMM, 1/sqrt(d) folded into W_q)
    o  = softmax(q k^T + bias) v * sigmoid(g)          (one attention kernel)
    a  = a + gate_out * (o W_o)                        (GEMM, addcmul)
    x  = LN(a) * gate + shift
    a  = a + gate_t * (silu(x W_1) * (x W_2)) W_b      (GEMM, silu_mul, GEMM, addcmul)

Used in evaluation without autograd or autocast on CUDA (the diffusion
sampler runs the transformer in FP32 up to 2,560 tokens).
MOL_FUSED_DIFFUSION_BLOCKS=0 disables.
"""

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_ENABLED = os.getenv("MOL_FUSED_DIFFUSION_BLOCKS", "1") != "0"
# Bound on the conditioning computed at once (bytes): blocks are grouped so a
# group's projections of s stay below it.
_COND_BYTES = 256 << 20


@triton.jit
def _adaln_kernel(X, G, SH, Y, sx, sg, ssh, sy, C, eps, BLOCK_C: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_C)
    m = offs < C
    x = tl.load(X + row * sx + offs, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / C
    xc = tl.where(m, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, 0) / C + eps)
    g = tl.load(G + row * sg + offs, mask=m, other=0.0)
    sh = tl.load(SH + row * ssh + offs, mask=m, other=0.0)
    tl.store(Y + row * sy + offs, xc * rstd * g + sh, mask=m)


def adaln(x: torch.Tensor, gate: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm(x) (no affine) * gate + shift for [rows, C] inputs (row strides free)."""
    rows, c = x.shape
    y = torch.empty((rows, c), device=x.device, dtype=x.dtype)
    block_c = triton.next_power_of_2(c)
    _adaln_kernel[(rows,)](
        x, gate, shift, y, x.stride(0), gate.stride(0), shift.stride(0), y.stride(0), c, eps,
        BLOCK_C=block_c, num_warps=4 if block_c >= 512 else 1,
    )
    return y


def fused_ok(dt, a: torch.Tensor, s: torch.Tensor) -> bool:
    if not (_ENABLED and a.is_cuda and not dt.training and not torch.is_grad_enabled()):
        return False
    if torch.is_autocast_enabled("cuda"):
        return False
    if a.dtype != torch.float32 or s.dtype != torch.float32 or a.dim() != 3:
        return False
    if s.shape[:-1] != a.shape[:-1]:
        return False
    block = dt.blocks[0]
    apb = block.attention_pair_bias
    return (
        apb.has_s
        and not apb.cross_attention_mode
        and apb.attention.gating
        and isinstance(block.drop_path, torch.nn.Identity)
        and all(isinstance(b.drop_path, torch.nn.Identity) for b in dt.blocks)
    )


class _BlockWeights:
    __slots__ = ("w_qkvg", "b_qkvg", "w_o", "w_12", "w_b")


def _fused_weights(dt):
    """Per-block attention/transition weights and, per block, the s-side
    weights in the layout _conditioning expects. Cached on the module and
    rebuilt when a parameter changes."""
    params = [p for p in dt.parameters()]
    key = tuple((p.data_ptr(), p._version) for p in params)
    cached = getattr(dt, "_fused_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    blocks, w_ln, b_ln, w_raw, b_raw = [], [], [], [], []
    for blk in dt.blocks:
        apb = blk.attention_pair_bias
        att = apb.attention
        ctb = blk.conditioned_transition_block
        scale = 1.0 / math.sqrt(att.c_hidden)
        bw = _BlockWeights()
        q_bias = att.linear_q.bias if att.linear_q.bias is not None else torch.zeros_like(att.linear_q.weight[:, 0])
        hd = att.linear_q.weight.shape[0]
        bw.w_qkvg = torch.cat(
            [att.linear_q.weight * scale, att.linear_k.weight, att.linear_v.weight, att.linear_g.weight], 0
        ).contiguous()
        bw.b_qkvg = torch.cat([q_bias * scale, q_bias.new_zeros(3 * hd)])
        bw.w_o = att.linear_o.weight
        bw.w_12 = torch.cat([ctb.linear_nobias_a1.weight, ctb.linear_nobias_a2.weight], 0).contiguous()
        bw.w_b = ctb.linear_nobias_b.weight
        blocks.append(bw)
        # AdaLN(a, s) = sigmoid(linear_s(LN_w(s))) * LN(a) + linear_nobias_s(LN_w(s));
        # with LN_w(s) = LN(s) * w the weights absorb w.
        ln1, ln2 = apb.layernorm_a, ctb.adaln
        rows, biases = [], []
        for ln in (ln1, ln2):  # gates
            w = ln.layernorm_s.weight
            rows.append(ln.linear_s.weight * w[None, :])
            biases.append(ln.linear_s.bias)
        for ln in (ln1, ln2):  # shifts
            w = ln.layernorm_s.weight
            rows.append(ln.linear_nobias_s.weight * w[None, :])
            biases.append(torch.zeros_like(ln.linear_s.bias))
        w_ln.append(torch.cat(rows, 0))
        b_ln.append(torch.cat(biases))
        w_raw.append(torch.cat([apb.linear_a_last.weight, ctb.linear_s.weight], 0))
        b_raw.append(torch.cat([apb.linear_a_last.bias, ctb.linear_s.bias]))
    # Block-major, so a group of consecutive blocks is a contiguous slice.
    out = (blocks, torch.cat(w_ln, 0), torch.cat(b_ln), torch.cat(w_raw, 0), torch.cat(b_raw))
    dt._fused_cache = (key, out)
    return out


def _attention(blk, x, rows_shape, z, n_queries, n_keys, pair_bias, chunk_size, enable_efficient_fusion, bw):
    """sigmoid(g) * attention(x) projected by W_o, as [rows, c_a]."""
    from mol.model.modules.primitives import _attention as sdpa, _local_attention
    from mol.model.modules.triton_attention import gated_attention
    from mol.model.utils import permute_final_dims

    apb = blk.attention_pair_bias
    att = apb.attention
    s_, n = rows_shape
    h, d = att.num_heads, att.c_hidden
    qkvg = torch.addmm(bw.b_qkvg, x, bw.w_qkvg.t()).view(s_, n, 4, h, d)
    q, k, v, g = (qkvg[:, :, i].transpose(1, 2) for i in range(4))  # [S, H, N, D] views
    out = torch.empty((s_, n, h, d), device=x.device, dtype=x.dtype)
    done = False
    if n_queries and n_keys:
        bias = apb.linear_nobias_z(apb.layernorm_z(z))
        bias = permute_final_dims(bias, [3, 0, 1, 2])
        if bias.dim() != q.dim() + 1:
            bias = bias.unsqueeze(-4)
        o = _local_attention(
            q=q, k=k, v=v, n_queries=n_queries, n_keys=n_keys, trunked_attn_bias=bias,
            use_efficient_implementation=att.use_efficient_implementation, chunk_size=chunk_size,
        )
    else:
        if pair_bias is not None:
            bias = pair_bias
        elif enable_efficient_fusion:
            bias = apb._project_pair_bias_from_normalized(z)
        else:
            bias = apb.project_pair_bias(z)
        if bias.dim() != q.dim():
            bias = bias.unsqueeze(-3)
        done = gated_attention(q, k, v, bias, g, out.transpose(1, 2))
        if not done:
            o = sdpa(q=q, k=k, v=v, attn_bias=bias, use_efficient_implementation=att.use_efficient_implementation)
    if not done:
        torch.mul(o.transpose(1, 2), torch.sigmoid(qkvg[:, :, 3]), out=out)
    return out.view(s_ * n, h * d) @ bw.w_o.t()


def fused_forward(
    dt,
    a: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
    n_queries: Optional[int],
    n_keys: Optional[int],
    pair_biases,
    chunk_size: Optional[int],
    enable_efficient_fusion: bool = False,
) -> torch.Tensor:
    from mol.model.modules.triton_swiglu import silu_mul

    blocks, w_ln, b_ln, w_raw, b_raw = _fused_weights(dt)
    s_, n, c_a = a.shape
    c_s = s.shape[-1]
    rows = s_ * n
    x_eps = dt.blocks[0].attention_pair_bias.layernorm_a.layernorm_a.eps
    s_eps = dt.blocks[0].attention_pair_bias.layernorm_a.layernorm_s.eps
    s2 = s.reshape(rows, c_s)
    s_ln = F.layer_norm(s2, (c_s,), None, None, s_eps)
    a2 = a.reshape(rows, c_a).contiguous()
    per_block = rows * 6 * c_a * a.element_size()
    group = max(1, min(len(blocks), _COND_BYTES // per_block))
    for g0 in range(0, len(blocks), group):
        g1 = min(g0 + group, len(blocks))
        idx = range(g0, g1)
        nb = len(idx)
        # [rows, nb, 4, c_a]: gate_attn, gate_trans (sigmoid), shift_attn, shift_trans
        sl = slice(g0 * 4 * c_a, g1 * 4 * c_a)
        cond = torch.addmm(b_ln[sl], s_ln, w_ln[sl].t()).view(rows, nb, 4, c_a)
        cond[:, :, :2].sigmoid_()
        # [rows, nb, 2, c_a]: sigmoid(linear_a_last(s)), sigmoid(linear_s(s))
        sl = slice(g0 * 2 * c_a, g1 * 2 * c_a)
        out_gates = torch.addmm(b_raw[sl], s2, w_raw[sl].t()).sigmoid_().view(rows, nb, 2, c_a)
        for j, i in enumerate(idx):
            blk, bw = dt.blocks[i], blocks[i]
            x = adaln(a2, cond[:, j, 0], cond[:, j, 2], x_eps)
            y = _attention(
                blk, x, (s_, n), z, n_queries, n_keys,
                None if pair_biases is None else pair_biases.blocks[i], chunk_size,
                enable_efficient_fusion, bw,
            )
            a2 = torch.addcmul(a2, out_gates[:, j, 0], y)
            x = adaln(a2, cond[:, j, 1], cond[:, j, 3], x_eps)
            y = silu_mul(x @ bw.w_12.t()) @ bw.w_b.t()
            a2.addcmul_(out_gates[:, j, 1], y)
    return a2.view(s_, n, c_a)
