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

"""[B, C, D, E] -> [B, D, C, E] copy at memory bandwidth.

PyTorch's generic strided copy runs this at about a quarter of DRAM
bandwidth on GB300 (the outer product mean's [N, C, N, C] -> [N, N, C, C]
reshape: 0.99 ms vs 0.25 ms at 644 tokens).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swap12_kernel(X, Y, C, D, E: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    e = tl.arange(0, E)
    mask = (c < C)[:, None, None] & (d < D)[None, :, None]
    base = b * C * D * E
    x = tl.load(X + base + c[:, None, None] * (D * E) + d[None, :, None] * E + e[None, None, :], mask=mask)
    tl.store(Y + base + d[None, :, None] * (C * E) + c[:, None, None] * E + e[None, None, :], x, mask=mask)


def swap12_ok(x: torch.Tensor) -> bool:
    e = x.shape[-1]
    return x.is_cuda and x.dim() == 4 and x.is_contiguous() and e >= 8 and e & (e - 1) == 0 and e <= 128


def swap12(x: torch.Tensor) -> torch.Tensor:
    """Contiguous x: [B, C, D, E] (E a power of two) -> contiguous [B, D, C, E]."""
    b, c, d, e = x.shape
    y = torch.empty((b, d, c, e), device=x.device, dtype=x.dtype)
    _swap12_kernel[(b, triton.cdiv(c, 16), triton.cdiv(d, 32))](
        x, y, c, d, E=e, BLOCK_C=16, BLOCK_D=32, num_warps=8
    )
    return y
