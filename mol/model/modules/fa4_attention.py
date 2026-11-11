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

"""Diffusion token attention on FlashAttention-4, Blackwell, FP16 operands.

out = softmax(q k^T + bias) v * sigmoid(gate) for the diffusion token
transformer (pair bias shared by the samples, head dim 48 at Mol-1's size),
on the FA4 SM100 forward kernel of ``tri_fa4`` (bias tiles by TMA added by the
tensor cores, gate in the correction epilogue), compiled for FP16 operands.

The sampler runs this transformer in FP32 with TF32 matmuls. FP16 has the
same 10-bit mantissa as TF32, so with q/k/v/gate produced by an FP16
projection GEMM (FP32 accumulation) the operands carry the precision the TF32
path already gives them; FP16's range (6e-5 .. 65504) is ample for these
activations. Softmax, the gate's sigmoid and accumulation stay FP32; P is
rounded to FP16 for the P V product (as the TF32 kernel rounds it to TF32),
and so is the pair bias (converted once per prediction), which the Triton
kernel adds in FP32.

At [5 samples, 16 heads, 644 tokens, 48] on GB300 this takes 21 us per call
against 74 us for the Triton TMA kernel (``triton_attention.py``).

Used only with TF32 matmuls enabled, at inference. ``MOL_FA4_ATTENTION=0``
disables it (back to the Triton TF32 kernel).
"""

import os

import torch

_ENABLED = os.getenv("MOL_FA4_ATTENTION", "1") != "0"
_DTYPE = torch.float16
# Tile config: 2 x 128 query rows per CTA (two softmax warp groups), 128-key blocks;
# three bias stages (with four, shared memory has room for a single K/V stage at
# head dim 48), and 80 registers for the correction warps instead of 64 (the gate
# epilogue runs 0.7 us faster per call at the diffusion shape).
_CFG = (
    ("m_block_size", 128), ("n_block_size", 128), ("q_stage", 2),
    ("bias_stage", 3), ("num_regs", (192, 80)),
)

_seqused = {}


def _kernel(head_dim: int):
    from mol.model.triangular import tri_fa4

    return tri_fa4.load_variant(f"token_fa4_d{head_dim}", "float16", head_dim, _CFG)


def available(x: torch.Tensor, n_heads: int, head_dim: int) -> bool:
    """Whether the FP16 FA4 path applies to the FP32 attention of `x`'s
    transformer (compiles / loads the kernel on first use)."""
    return (
        _ENABLED
        and x.is_cuda
        and x.dtype == torch.float32
        and torch.backends.cuda.matmul.allow_tf32
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled("cuda")
        and head_dim % 16 == 0
        and head_dim <= 128
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and _kernel(head_dim) is not None
    )


def fp16_bias(bias: torch.Tensor) -> torch.Tensor:
    """[(1,) H, M, N] FP32 bias -> [H, M, N8] FP16, N padded to a multiple of 8
    (16-byte rows for TMA). Cached on the tensor while it is unchanged (the
    request-scoped pair biases are converted once per prediction)."""
    key = (bias.data_ptr(), bias._version, bias.shape)
    cached = getattr(bias, "_mol_fp16_bias", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    b = bias.reshape(bias.shape[-3:]) if bias.dim() > 3 else bias
    h, m, n = b.shape
    out = torch.empty((h, m, (n + 7) // 8 * 8), device=b.device, dtype=_DTYPE)
    out[..., :n].copy_(b)
    if out.shape[-1] != n:
        out[..., n:].zero_()
    bias._mol_fp16_bias = (key, out)
    return out


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and t.stride(-1) == 1 and all(s % 8 == 0 for s in t.stride()[:-1])


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias16: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """out = softmax(q k^T + bias) v * sigmoid(gate) (scale folded into q).
    q, k, v, gate, out: FP16 [B, S, H, D] views (tokens outside heads, unit
    stride last, 16-byte aligned strides); bias16: [H, S, S8] from
    fp16_bias. Returns False, with nothing written, where the kernel does not
    apply."""
    if not all(t.dtype == _DTYPE and t.dim() == 4 for t in (q, k, v, gate, out)):
        return False
    b, m, h, d = q.shape
    n = k.shape[1]
    if k.shape != (b, n, h, d) or v.shape != k.shape or gate.shape != q.shape or out.shape != q.shape:
        return False
    if bias16.dtype != _DTYPE or bias16.shape[:2] != (h, m) or bias16.shape[2] < n:
        return False
    if not all(_aligned(t) for t in (q, k, v, gate, out, bias16)):
        return False
    fn = _kernel(d)
    if fn is None:
        return False
    key = (b, n, q.device)
    if key not in _seqused:
        _seqused[key] = torch.full((b,), n, dtype=torch.int32, device=q.device)
    fn(q, k, v, out, None, 1.0, None, None, None, _seqused[key], None, None, None, None, None, None,
       (None, None), bias16, gate)
    return True
