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
    y  = o W_o                                         (GEMM)
    a  = a + gate_out * y; x = LN(a) * gate + shift    (one kernel)
    y  = (silu(x W_1) * (x W_2)) W_b                   (GEMM, silu_mul, GEMM)

and the next block's first kernel adds gate_t * y. The gates are sigmoids
of s projections, taken in those kernels.

In cross-attention mode (the atom transformers) keys and values project a
second AdaLN of x, and s may be shared by the samples ([N, c_s] for [S, N,
c_a]), in which case the conditioning is computed once per token.

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
def _adaln_kernel(
    A, OG, R, G, SH, AO, Y, sa, sog, sr, sg, ssh, sao, sy, C, COND_ROWS, eps,
    HAS_RES: tl.constexpr, HAS_LN: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # One row: with HAS_RES, a = a + sigmoid(og) * r (the previous sublayer's
    # gated residual) is stored to AO; with HAS_LN, LN(a) * sigmoid(g) + sh
    # (the adaptive LayerNorm) is stored to Y. og/g/sh have COND_ROWS rows,
    # repeated over the rows of a (conditioning shared by the samples).
    row = tl.program_id(0).to(tl.int64)
    crow = row % COND_ROWS
    offs = tl.arange(0, BLOCK_C)
    m = offs < C
    a = tl.load(A + row * sa + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        og = tl.load(OG + crow * sog + offs, mask=m, other=0.0)
        r = tl.load(R + row * sr + offs, mask=m, other=0.0)
        a = a + tl.sigmoid(og) * r
        tl.store(AO + row * sao + offs, a, mask=m)
    if HAS_LN:
        mean = tl.sum(a, 0) / C
        ac = tl.where(m, a - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(ac * ac, 0) / C + eps)
        g = tl.load(G + crow * sg + offs, mask=m, other=0.0)
        sh = tl.load(SH + crow * ssh + offs, mask=m, other=0.0)
        tl.store(Y + row * sy + offs, ac * rstd * tl.sigmoid(g) + sh, mask=m)


def adaln(
    a: torch.Tensor,
    gate: Optional[torch.Tensor],
    shift: Optional[torch.Tensor],
    eps: float,
    res: Optional[tuple] = None,
    a_out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """LayerNorm(a) (no affine) * sigmoid(gate) + shift for [rows, C] inputs
    (row strides free). res = (og, r): first a_out = a + sigmoid(og) * r
    (a_out may be a itself), and the LayerNorm is of a_out. gate None: only
    the residual update, returns None. gate/shift/og may have fewer rows than
    a (a divisor of them): they repeat over a's rows."""
    rows, c = a.shape
    has_ln = gate is not None
    y = torch.empty((rows, c), device=a.device, dtype=a.dtype) if has_ln else a
    og, r = res if res is not None else (a, a)
    ao = a if a_out is None else a_out
    g = gate if has_ln else a
    sh = shift if has_ln else a
    cond_rows = (gate if has_ln else og).shape[0]
    assert rows % cond_rows == 0
    block_c = triton.next_power_of_2(c)
    _adaln_kernel[(rows,)](
        a, og, r, g, sh, ao, y,
        a.stride(0), og.stride(0), r.stride(0), g.stride(0), sh.stride(0), ao.stride(0), y.stride(0), c, cond_rows, eps,
        HAS_RES=res is not None, HAS_LN=has_ln, BLOCK_C=block_c, num_warps=4 if block_c >= 512 else 1,
    )
    return y if has_ln else None


def _shared_s(a: torch.Tensor, s: torch.Tensor) -> bool:
    # s [(1,) N, c_s] shared by the samples of a [S, N, c_a] (the atom transformers)
    return s.shape[-2] == a.shape[-2] and s.numel() == a.shape[-2] * s.shape[-1]


def fused_ok(dt, a: torch.Tensor, s: torch.Tensor) -> bool:
    if not (_ENABLED and a.is_cuda and not dt.training and not torch.is_grad_enabled()):
        return False
    if torch.is_autocast_enabled("cuda"):
        return False
    if a.dtype != torch.float32 or s.dtype != torch.float32 or a.dim() != 3:
        return False
    if s.shape[:-1] != a.shape[:-1] and not _shared_s(a, s):
        return False
    block = dt.blocks[0]
    apb = block.attention_pair_bias
    return (
        apb.has_s
        and apb.attention.gating
        and isinstance(block.drop_path, torch.nn.Identity)
        and all(isinstance(b.drop_path, torch.nn.Identity) for b in dt.blocks)
    )


class _BlockWeights:
    __slots__ = ("w_qkvg", "b_qkvg", "w_o", "w_12", "w_b", "cross")


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
        # Rows [q; g; k; v]: with cross attention, [q; g] project x and [k; v]
        # the separately normalized x_kv.
        bw.w_qkvg = torch.cat(
            [att.linear_q.weight * scale, att.linear_g.weight, att.linear_k.weight, att.linear_v.weight], 0
        ).contiguous()
        bw.b_qkvg = torch.cat([q_bias * scale, q_bias.new_zeros(3 * hd)])
        bw.cross = apb.cross_attention_mode
        bw.w_o = att.linear_o.weight
        bw.w_12 = torch.cat([ctb.linear_nobias_a1.weight, ctb.linear_nobias_a2.weight], 0).contiguous()
        bw.w_b = ctb.linear_nobias_b.weight
        blocks.append(bw)
        # AdaLN(a, s) = sigmoid(linear_s(LN_w(s))) * LN(a) + linear_nobias_s(LN_w(s));
        # with LN_w(s) = LN(s) * w the weights absorb w.
        lns = (apb.layernorm_a, ctb.adaln) + ((apb.layernorm_kv,) if apb.cross_attention_mode else ())
        rows, biases = [], []
        for ln in lns:  # gates
            w = ln.layernorm_s.weight
            rows.append(ln.linear_s.weight * w[None, :])
            biases.append(ln.linear_s.bias)
        for ln in lns:  # shifts
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


def _attention(blk, x, x_kv, rows_shape, z, n_queries, n_keys, pair_bias, chunk_size, enable_efficient_fusion, bw):
    """sigmoid(g) * attention(x) projected by W_o, as [rows, c_a]; with x_kv
    (cross attention), keys and values project x_kv."""
    from mol.model.modules.primitives import _attention as sdpa, _local_attention
    from mol.model.modules.triton_attention import gated_attention
    from mol.model.utils import permute_final_dims

    apb = blk.attention_pair_bias
    att = apb.attention
    s_, n = rows_shape
    h, d = att.num_heads, att.c_hidden
    if x_kv is None:
        qgkv = torch.addmm(bw.b_qkvg, x, bw.w_qkvg.t()).view(s_, n, 4, h, d)
        q, g, k, v = (qgkv[:, :, i].transpose(1, 2) for i in range(4))  # [S, H, N, D] views
    else:
        hd2 = 2 * h * d
        qg = torch.addmm(bw.b_qkvg[:hd2], x, bw.w_qkvg[:hd2].t()).view(s_, n, 2, h, d)
        kv = (x_kv @ bw.w_qkvg[hd2:].t()).view(s_, n, 2, h, d)
        q, g, k, v = (t[:, :, i].transpose(1, 2) for t in (qg, kv) for i in range(2))
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
        torch.mul(o.transpose(1, 2), torch.sigmoid(g.transpose(1, 2)), out=out)
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
    # the conditioning has one row per row of a, or per token when s is shared
    s_rows = n if s.shape[:-1] != a.shape[:-1] else rows
    s2 = s.reshape(s_rows, c_s)
    s_ln = F.layer_norm(s2, (c_s,), None, None, s_eps)
    a2 = a.reshape(rows, c_a)
    res = None  # the pending gated residual (gate logits, update), applied by the next adaln
    owned = False  # a2 is ours to update in place (a is the caller's)

    def norm(gate, shift):
        nonlocal a2, owned
        a_out = None
        if res is not None and not owned:
            a_out = torch.empty((rows, c_a), device=a.device, dtype=a.dtype)
        y = adaln(a2, gate, shift, x_eps, res, a_out)
        if a_out is not None:
            a2, owned = a_out, True
        return y
    cross = blocks[0].cross
    nc = 6 if cross else 4  # conditioning rows per block (a gate and a shift per AdaLN)
    per_block = s_rows * (nc + 2) * c_a * a.element_size()
    group = max(1, min(len(blocks), _COND_BYTES // per_block))
    for g0 in range(0, len(blocks), group):
        g1 = min(g0 + group, len(blocks))
        idx = range(g0, g1)
        nb = len(idx)
        # [s_rows, nb, nc, c_a]: gate_attn, gate_trans, (gate_kv,) (logits), then the shifts
        sl = slice(g0 * nc * c_a, g1 * nc * c_a)
        cond = torch.addmm(b_ln[sl], s_ln, w_ln[sl].t()).view(s_rows, nb, nc, c_a)
        sh = nc // 2
        # [s_rows, nb, 2, c_a]: linear_a_last(s), linear_s(s) (output gate logits)
        sl = slice(g0 * 2 * c_a, g1 * 2 * c_a)
        out_gates = torch.addmm(b_raw[sl], s2, w_raw[sl].t()).view(s_rows, nb, 2, c_a)
        for j, i in enumerate(idx):
            blk, bw = dt.blocks[i], blocks[i]
            x = norm(cond[:, j, 0], cond[:, j, sh])
            # as AttentionPairBias: x_kv is the kv AdaLN of x (not of a)
            x_kv = adaln(x, cond[:, j, 2], cond[:, j, sh + 2], x_eps) if cross else None
            y = _attention(
                blk, x, x_kv, (s_, n), z, n_queries, n_keys,
                None if pair_biases is None else pair_biases.blocks[i], chunk_size,
                enable_efficient_fusion, bw,
            )
            res = (out_gates[:, j, 0], y)
            x = norm(cond[:, j, 1], cond[:, j, sh + 1])
            y = silu_mul(x @ bw.w_12.t()) @ bw.w_b.t()
            res = (out_gates[:, j, 1], y)
    if res is not None:
        norm(None, None)
    return a2.view(s_, n, c_a)
