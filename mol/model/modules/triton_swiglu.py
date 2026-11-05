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

"""``silu(a) * b`` over a fused ``[a | b]`` projection, in one pass."""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(AB, Y, M, F: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = (m < M)[:, None] & (f < F)[None, :]
    row = m.to(tl.int64)[:, None]
    a = tl.load(AB + row * (2 * F) + f[None, :], mask=mask, other=0.0).to(tl.float32)
    b = tl.load(AB + row * (2 * F) + F + f[None, :], mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(a) * b
    tl.store(Y + row * F + f[None, :], y.to(Y.dtype.element_ty), mask=mask)


def silu_mul(ab: torch.Tensor) -> torch.Tensor:
    """ab: contiguous [M, 2F] holding [a | b]; returns silu(a) * b as [M, F]."""
    m, f2 = ab.shape
    f = f2 // 2
    y = torch.empty((m, f), device=ab.device, dtype=ab.dtype)
    grid = (triton.cdiv(m, 32), triton.cdiv(f, 128))
    _silu_mul_kernel[grid](ab, y, m, f, BLOCK_M=32, BLOCK_F=128, num_warps=4)
    return y
