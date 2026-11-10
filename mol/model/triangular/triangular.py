# Copyright 2024 ByteDance and/or its affiliates.
# Modifications Copyright 2026 Molence Bio
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
#
# This file has been modified by Molence Bio
# Based on Protenix: https://github.com/bytedance/Protenix
# Upstream revision: bd54a05d047b8925a241056f36d619700604068a
# Upstream file: protenix/model/triangular/triangular.py
# Changes: FlashPairformer backend for triangle multiplication.
# Copyright 2021 AlQuraishi Laboratory


from abc import ABC, abstractmethod
from functools import partial, partialmethod
from typing import List, Optional

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mol.model.triangular.layers import Attention, LayerNorm, OpenfoldLinear
from mol.model.utils import chunk_layer, is_fp16_enabled, permute_final_dims


# Inference-only padding of the pair representation for the cuEquivariance
# triangle kernels. On Blackwell, triangle attention only takes the SM100f path
# when the token count is a multiple of 8, and both kernels tile better on
# aligned sizes. Padded rows/columns are masked out and sliced off, so the
# result for the real tokens is unchanged. MOL_TRI_PAD=0 disables it.
_TRI_PAD_MULTIPLE = int(os.getenv("MOL_TRI_PAD", "8"))
# Inference fast path for TriangleAttention with cuEquivariance
# (see TriangleAttention._fused_inference_forward). MOL_FUSED_TRI_ATT=0 disables.
_FUSED_TRI_ATT = os.getenv("MOL_FUSED_TRI_ATT", "1") != "0"
# cuBLAS-based triangle multiplication at inference (fast_trimul.py).
# MOL_FAST_TRIMUL=0 disables.
_FAST_TRIMUL = os.getenv("MOL_FAST_TRIMUL", "1") != "0"


def _key_prefix(mask: torch.Tensor, ending: bool) -> Optional[int]:
    """For a pair mask [I, J] of the form row_valid[i] * col_valid[j] whose
    valid keys are a prefix (padding), the number of valid keys; else None.
    Keys are j for starting-node attention, i for ending-node attention.

    Computed once per mask tensor (one host sync) and kept on the tensor, so
    the blocks of a stack, which share the mask, reuse it. Rows that are
    entirely masked (padding) then attend over the valid keys instead of
    uniformly over all keys; they are padding either way.
    """
    cached = getattr(mask, "_mol_key_prefix", None)
    if cached is None or cached[0] != mask._version:
        m = mask != 0
        rows, cols = m.any(1), m.any(0)
        separable = (m == (rows[:, None] & cols[None, :])).all()
        stats = []
        for keys in (cols, rows):
            n = keys.sum()
            prefix = (keys == (torch.arange(keys.numel(), device=m.device) < n)).all()
            stats += [(separable & prefix).int(), n.int()]
        ok_c, n_c, ok_r, n_r = torch.stack(stats).tolist()
        cached = (mask._version, (n_c if ok_c and n_c > 0 else None, n_r if ok_r and n_r > 0 else None))
        mask._mol_key_prefix = cached
    return cached[1][1 if ending else 0]


def _tri_pad_amount(x: torch.Tensor) -> int:
    """Rows/columns to append to a square [*, N, N, C] pair tensor, or 0."""
    if (
        _TRI_PAD_MULTIPLE <= 0
        or not x.is_cuda
        or torch.is_grad_enabled()
        or x.shape[-2] != x.shape[-3]
    ):
        return 0
    return (-x.shape[-2]) % _TRI_PAD_MULTIPLE


def kernel_triangular_mult(
    x: torch.Tensor,
    direction: str,
    mask: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update

    from mol.model.triangular import cueq_tuning

    cueq_tuning.install(x.device)

    """
        This function performs a triangle multiplicative update operation, which is a key component
    in the AlphaFold2 architecture. The operation consists of:

    1. Input normalization and gating
    2. Triangular projection (either outgoing or incoming)
    3. Output normalization and gating

    The function supports both ahead-of-time (AOT) tuning and just-in-time (JIT) tuning.
    Auto-tuning behavior can be controlled through environment variables:

    - Quick testing: Default configuration where tuning configs, if existent, are looked-up.
        If not, then falls back to default kernel parameters. No tuning is performed.
    - On-Demand tuning: Set `CUEQ_TRITON_TUNING= "ONDEMAND"` to auto-tune for new shapes encountered on first run (may take several minutes)
    - AOT tuning: Set `CUEQ_TRITON_TUNING= "AOT"` to perform full ahead-of-time tuning for optimal performance **(may take several hours)**
    - Ignore user cache: Set CUEQ_TRITON_IGNORE_EXISTING_CACHE to ignore both the default settings that come with the package
        and any user-local settings previously saved with AOT/ONDEMAND tuning. May be used to regenerate optimal settings for a particular setup.
    - Cache directory: Set `CUEQ_TRITON_CACHE_DIR` to specify where tuning configurations are stored
    - Note: When using Docker with default or on-demand tuning enabled, commit the container to persist tuning changes

    Notes:
        (1) Context is saved for backward pass. You don't need to save it manually.
        (2) Kernel precision (fp32, bf16, fp16) is based on input dtypes. For tf32, set it from torch global
            scope using torch.backends.cuda.matmul.allow_tf32
        (3) **Limitation**: Currently only supports hidden_dim values that are multiples of 32.
        (4) We have moved away from the default round-towards-zero (RZ) implementation to round-nearest (RN)
            for better tf32 accuracy in cuex.triangle_multiplicative_update. In rare circumstances,
            this may cause minor differences in results observed.
        (5) When using torch compile, use `cueuivariance_ops_torch.init_triton_cache()` to initialize
            triton cache before calling torch compiled triangular multiplicative update.
        (6) Although the example demonstrates the most common case of one batch dimension,
            the API supports variable number of leading batch dimensions.
    """
    return triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Implements Algorithms 11 and 12.

    Args:
        c_z:
            Input channel dimension
        c_hidden:
            Hidden channel dimension
        _outgoing:
            Whether this is an outgoing update
    """

    @abstractmethod
    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool) -> None:
        super(BaseTriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = OpenfoldLinear(self.c_z, self.c_z, bias=False, init="gating")
        self.linear_z = OpenfoldLinear(
            self.c_hidden, self.c_z, bias=False, init="final"
        )

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _load_from_state_dict(self, *args, **kwargs):
        from .flashpairformer import clear_flashpairformer_cache

        clear_flashpairformer_cache(self)
        return super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, *args, **kwargs):
        from .flashpairformer import clear_flashpairformer_cache

        clear_flashpairformer_cache(self)
        return super()._apply(*args, **kwargs)

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        if _inplace_chunk_size is not None:
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.matmul(
                    a_chunk,
                    b_chunk,
                )

            p = a
        else:
            p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        pass


class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements Algorithms 11 and 12.

    Args:
        c_z:
            Input channel dimension
        c_hidden:
            Hidden channel dimension
        _outgoing:
            Whether this is an outgoing update. Defaults to True.
    """

    def _fast_trimul_ok(self, z: torch.Tensor) -> bool:
        return (
            _FAST_TRIMUL
            and z.is_cuda
            and z.dim() == 3
            # FP32 (the confidence head) with TF32: 1.8x faster than
            # cuEquivariance's FP32 gated GEMMs; in strict FP32 slightly slower.
            and (
                z.dtype in (torch.bfloat16, torch.float16)
                or (z.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32)
            )
            and not torch.is_grad_enabled()
            and z.shape[-2] > 16
        )

    def _fast_trimul_weights(self, dtype: torch.dtype) -> tuple:
        params = (
            self.layer_norm_in.weight,
            self.layer_norm_in.bias,
            self.linear_a_g.weight,
            self.linear_b_g.weight,
            self.linear_a_p.weight,
            self.linear_b_p.weight,
            self.layer_norm_out.weight,
            self.layer_norm_out.bias,
            self.linear_z.weight,
            self.linear_g.weight,
        )
        key = (dtype, tuple((p.data_ptr(), p._version) for p in params))
        cached = getattr(self, "_fast_w_cache", None)
        if cached is None or cached[0] != key:
            from mol.model.triangular.fast_trimul import GATE_BLOCK, fused_output_weights, interleave_gate_weight

            p = [t.to(dtype) for t in params]
            w_in = torch.cat(p[2:6], 0)
            w_il = interleave_gate_weight(w_in) if (w_in.shape[0] // 2) % GATE_BLOCK == 0 else None
            weights = (p[0], p[1], w_in, p[6], p[7], p[8], p[9])
            fused = fused_output_weights(params[6], params[7], params[8]) if dtype == torch.bfloat16 else None
            self._fast_w_cache = (key, (weights, w_il, fused))
        return self._fast_w_cache[1]

    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool = True) -> None:
        super(TriangleMultiplicativeUpdate, self).__init__(
            c_z=c_z, c_hidden=c_hidden, _outgoing=_outgoing
        )

        self.linear_a_p = OpenfoldLinear(self.c_z, self.c_hidden, bias=False)
        self.linear_a_g = OpenfoldLinear(
            self.c_z, self.c_hidden, bias=False, init="gating"
        )
        self.linear_b_p = OpenfoldLinear(self.c_z, self.c_hidden, bias=False)
        self.linear_b_g = OpenfoldLinear(
            self.c_z, self.c_hidden, bias=False, init="gating"
        )

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade
                memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z

        More memory-efficient, inference-only version of the forward function.
        Uses in-place operations, fusion of the addition that happens after
        this module in the Evoformer, a smidge of recomputation, and
        a cache of overwritten values to lower peak memory consumption of this
        module from 5x the size of the input tensor z to 2.5x its size. Useful
        for inference on extremely long sequences.
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask, a=True):
            if a:
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p

            pair = self.layer_norm_in(pair)
            p = linear_g(pair)
            p.sigmoid_()
            p *= linear_p(pair)
            p *= mask
            p = permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True):
            need_transpose = self._outgoing ^ a
            if not chunked:
                p = compute_projection_helper(pair, mask, a)
                if need_transpose:
                    p = p.transpose(-1, -2)
            else:
                # This computation is chunked so as not to exceed our 2.5x
                # budget with a large intermediate tensor
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.weight.shape[0]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = pair[..., i : i + inplace_chunk_size, :, :]
                    mask_chunk = mask[..., i : i + inplace_chunk_size, :, :]
                    pair_chunk = compute_projection_helper(
                        pair[..., i : i + inplace_chunk_size, :, :],
                        mask[..., i : i + inplace_chunk_size, :, :],
                        a,
                    )
                    if need_transpose:
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i : i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i : i + inplace_chunk_size, :] = pair_chunk

                    del pair_chunk

            return p

        # We start by fully manifesting a. In addition to the input, this
        # brings total memory consumption to 2x z (disregarding size of chunks)
        # [*, N, N, c]
        a = compute_projection(z, mask, True, chunked=True)

        if inplace_chunk_size is not None:
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim

            def empty_slicer(t):
                return [slice(None) for _ in t.shape]

            def slice_tensor(t, start, end, dim):
                # Slices start:end from the dim dimension of t
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[tuple(s)]

            def flip_z_cache_(z_cache, z):
                # "Reorient" the z_cache (see below), filling it with quadrants
                # 3---recovered from the z_cache---and 4---recovered from z---
                # of the input tensor z.
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.transpose(row_dim, col_dim)

                # If n is odd, we need to shrink the z_cache by one row
                z_cache = z_cache[..., : (n // 2), :, :]

                # Move the 3rd quadrant of z into the
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[tuple(first_half_slicer)] = quadrant_3

                # Get the fourth quadrant of z
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)

                # Insert said quadrant into the rotated z-cache
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)

                z_cache[tuple(quadrant_3_slicer)] = quadrant_4

                return z_cache

            # Initialize the z cache to the left half of z.
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[tuple(z_cache_slicer)])
            z_cache_rotated = False

            # We need to reorient the z-cache at the halfway point, and we
            # don't want a single chunk to straddle that point. We contract one
            # of the chunks in the middle to address that problem.
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [
                i_2 - i_1 for i_1, i_2 in zip(i_range, i_range[1:] + [half_n])
            ]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(
                i_range + after_half, initial_offsets + after_half_offsets
            )
            for i, offset in combined_range_with_offsets:
                if not z_cache_rotated and i >= half_n:
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(
                    z,
                    i,
                    i + offset,
                    b_chunk_dim,
                )
                mask_chunk = slice_tensor(
                    mask,
                    i,
                    i + offset,
                    b_chunk_dim,
                )

                z_chunk_b = z_chunk_b.clone()
                if b_chunk_dim == col_dim:
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                else:  # b_chunk_dim == row_dim
                    # In this case, the b-dimension (b_chunk_dim) is partially
                    # overwritten at the end of each iteration. We need to
                    # restore the missing component from the z-cache.
                    if not z_cache_rotated:
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[tuple(z_chunk_slicer)] = slice_tensor(
                            z_cache,
                            i,
                            i + offset,
                            row_dim,
                        )
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(
                            z_cache, z_cache_offset, z_cache_offset + offset, row_dim
                        )

                b_chunk = compute_projection(
                    z_chunk_b, mask_chunk, a=False, chunked=False
                )
                del z_chunk_b

                x_chunk = torch.matmul(
                    a,
                    b_chunk,
                )
                x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)

                # The g dimension (col_dim) is parallel to and ahead of the
                # overwrites in z. We can extract the g chunk normally.
                z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
                g_chunk.sigmoid_()
                del z_chunk_g

                x_chunk *= g_chunk

                # Write the columns into z in-place
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if with_add:
                    z[tuple(z_slicer)] += x_chunk
                else:
                    z[tuple(z_slicer)] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = torch.matmul(a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.linear_g(z)
            g.sigmoid_()
            x *= g
            if with_add:
                z += x
            else:
                z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
        triangle_multiplicative: str = "torch",
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        _input_inplace_safe = inplace_safe is True

        if triangle_multiplicative == "flashpairformer":
            from .flashpairformer import triangle_multiplication

            update = triangle_multiplication(self, z, mask)
            if update is not None:
                return update + z if _input_inplace_safe and _add_with_inplace else update
            triangle_multiplicative = "cuequivariance" if z.is_cuda else "torch"

        # Note: cuequivariance requires that the hidden dimension c must equal c_z.
        # If this condition is not met, an AssertionError will be raised.
        # Therefore, we include a check here: if c != c_z, we fall back to using plain PyTorch.
        # This situation may occur in our template module.
        if triangle_multiplicative == "cuequivariance" and (self.c_z == self.c_hidden):
            if _input_inplace_safe and _add_with_inplace:
                # The kernel does not write to its input, so no clone needed.
                z_in = z
            if mask is None:
                mask = z.new_ones(z.shape[:-1])
            if mask.dim() == z.dim() - 1:
                mask = mask[None]
            n_tok = z.shape[-2]
            pad = _tri_pad_amount(z)
            if pad:
                z = F.pad(z, (0, 0, 0, pad, 0, pad))
                mask = F.pad(mask, (0, pad, 0, pad))
            if self._fast_trimul_ok(z):
                from mol.model.triangular import fast_trimul

                add_residual = _input_inplace_safe and _add_with_inplace
                fuse_residual = add_residual and not pad
                weights, w_in_interleaved, out_fused = self._fast_trimul_weights(z.dtype)
                out = fast_trimul.triangle_multiplicative_update(
                    z,
                    "outgoing" if self._outgoing else "incoming",
                    mask.reshape(z.shape[:-1]),
                    *weights,
                    eps=1e-5,
                    residual=z_in if fuse_residual else None,
                    in_weight_interleaved=w_in_interleaved,
                    out_fused_weights=out_fused,
                )
                if pad:
                    out = out[..., :n_tok, :n_tok, :]
                if add_residual and not fuse_residual:
                    out = out + z_in
                return out
            z = kernel_triangular_mult(
                z[None],
                direction="outgoing" if self._outgoing else "incoming",
                mask=mask,
                norm_in_weight=self.layer_norm_in.weight,
                norm_in_bias=self.layer_norm_in.bias,
                p_in_weight=torch.cat(
                    [self.linear_a_p.weight, self.linear_b_p.weight], 0
                ),
                g_in_weight=torch.cat(
                    [self.linear_a_g.weight, self.linear_b_g.weight], 0
                ),
                norm_out_weight=self.layer_norm_out.weight,
                norm_out_bias=self.layer_norm_out.bias,
                p_out_weight=self.linear_z.weight,
                g_out_weight=self.linear_g.weight,
                eps=1e-5,  # In BF16, we use the default eps of 1e-5.
            )[0]
            if pad:
                z = z[..., :n_tok, :n_tok, :]
            if _input_inplace_safe and _add_with_inplace:
                return z + z_in
            else:
                return z
        elif (triangle_multiplicative == "torch") or (self.c_z != self.c_hidden):
            if inplace_safe:
                x = self._inference_forward(
                    z,
                    mask,
                    inplace_chunk_size=_inplace_chunk_size,
                    with_add=_add_with_inplace,
                )
                return x

            if mask is None:
                mask = z.new_ones(z.shape[:-1])

            mask = mask.unsqueeze(-1)

            if _input_inplace_safe and _add_with_inplace:
                z_in = z.clone()

            z = self.layer_norm_in(z)
            a = mask
            a = a * self.sigmoid(self.linear_a_g(z))
            a = a * self.linear_a_p(z)
            b = mask
            b = b * self.sigmoid(self.linear_b_g(z))
            b = b * self.linear_b_p(z)

            # Prevents overflow of torch.matmul in combine projections in
            # reduced-precision modes
            a_std = a.std()
            b_std = b.std()
            if is_fp16_enabled() and a_std != 0.0 and b_std != 0.0:
                a = a / a.std()
                b = b / b.std()

            if is_fp16_enabled():
                with torch.amp.autocast("cuda", enabled=False):
                    x = self._combine_projections(a.float(), b.float())
            else:
                x = self._combine_projections(a, b)

            del a, b
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.sigmoid(self.linear_g(z))
            x = x * g
            if _input_inplace_safe and _add_with_inplace:
                x = x + z_in
            return x
        else:
            raise ValueError(
                f"triangle_multiplicative must be 'cuequivariance', 'flashpairformer' or 'torch', but got {triangle_multiplicative}"
            )


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class TriangleAttention(nn.Module):
    """
    Triangle attention.

    Args:
        c_in:
            Input channel dimension
        c_hidden:
            Overall hidden channel dimension (not per-head)
        no_heads:
            Number of attention heads
        starting:
            Whether this is a starting node attention. Defaults to True.
        inf:
            Value for attention masking. Defaults to 1e9.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ) -> None:
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = OpenfoldLinear(c_in, self.no_heads, bias=False)

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    def _fused_inference_ok(
        self, x: torch.Tensor, triangle_attention: str, chunk_size: Optional[int]
    ) -> bool:
        return (
            _FUSED_TRI_ATT
            and triangle_attention == "cuequivariance"
            and x.is_cuda
            and x.dim() == 3
            and not torch.is_grad_enabled()
            and self.mha.gating
            and (chunk_size is None or chunk_size >= x.shape[-3])
            and x.shape[-2] > 16
        )

    def _fused_projection_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """[Wq; Wk; Wv; Wg; W_bias] in one matrix, rows padded to a multiple of 8.

        Padding keeps every column slice of the projection 16-byte aligned, as
        the Blackwell attention kernel's TMA layout requires.
        """
        weights = [
            self.mha.linear_q.weight,
            self.mha.linear_k.weight,
            self.mha.linear_v.weight,
            self.mha.linear_g.weight,
            self.linear.weight,
        ]
        key = (dtype, tuple((w.data_ptr(), w._version) for w in weights))
        cached = getattr(self, "_fused_w_cache", None)
        if cached is None or cached[0] != key:
            w = torch.cat(weights, 0)
            w = F.pad(w, (0, 0, 0, (-w.shape[0]) % 8)).to(dtype)
            self._fused_w_cache = (key, w)
        return self._fused_w_cache[1]

    def _fused_inference_forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ending: bool = False,
        residual: Optional[torch.Tensor] = None,
        inplace: bool = False,
    ) -> torch.Tensor:
        """Inference fast path: one projection GEMM, cuEquivariance attention,
        fused sigmoid gating. Same math as the mha path.

        ending: attend along the first axis of ``x`` (ending-node attention on
        the untransposed pair tensor) instead of transposing ``x`` and the result.
        residual: [I, J, C] tensor added to the output in the output GEMM.
        inplace: accumulate into ``residual`` itself instead of a new tensor.
        """
        from mol.model.triangular.fused_gate import sigmoid_gate_transpose
        from mol.model.triangular.layers import cuequivariance_triangular_attn

        n_i, n_j, _ = x.shape
        h, d = self.no_heads, self.c_hidden
        hd = h * d
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            proj = F.linear(x.to(dtype), self._fused_projection_weight(dtype))
            # [I, H, J, D] views (ending: [J, H, I, D]); the kernel takes these strides as they are.
            perm = (1, 2, 0, 3) if ending else (0, 2, 1, 3)
            q, k, v = (
                proj[..., i * hd : (i + 1) * hd].view(n_i, n_j, h, d).permute(perm)
                for i in range(3)
            )
            g = proj[..., 3 * hd : 4 * hd]
            # [1, H, I, J] (ending: [1, H, J, I])
            bias = proj[..., 4 * hd : 4 * hd + h].permute((2, 1, 0) if ending else (2, 0, 1)).unsqueeze(0)
            if dtype == torch.float32:
                bias = bias.contiguous()
            key_mask = (mask.t() if ending else mask) != 0
            o = None
            if dtype == torch.bfloat16 and mask.dim() == 2:
                # Blackwell, padding masks: FlashAttention-4 (tri_fa4) or else
                # CUTLASS FMHA (tri_fmha), both with the bias on the tensor
                # cores and the gate in the epilogue.
                from mol.model.triangular import tri_fa4, tri_fmha

                gate = g.view(n_i, n_j, h, d).permute(perm)
                o = torch.empty((n_i, n_j, h, d), device=x.device, dtype=dtype)
                kernels = [m for m in (tri_fa4, tri_fmha) if m.supported(q, k, gate, o)]
                n_keys = _key_prefix(mask, ending) if kernels else None
                if n_keys is not None:
                    from mol.model.triangular.fused_gate import bias_heads

                    # [H, Q, K] contiguous, as the kernels load it by TMA
                    bias_hqk = bias_heads(proj[..., 4 * hd : 4 * hd + h], transposed=ending)
                if n_keys is not None and any(
                    m.attention(q, k, v, bias_hqk, gate, o.permute(perm), n_keys, 1.0 / math.sqrt(d)) for m in kernels
                ):
                    o = o.view(n_i, n_j, hd)
                else:
                    o = None
            if dtype == torch.float32:
                # FP32 (the confidence head) with TF32 enabled: the TF32 TMA
                # attention kernel with the gate fused; cuEquivariance's FP32
                # path is an sm80 kernel about 4x slower.
                from mol.model.modules.triton_attention import gated_attention

                o = torch.empty((n_i, n_j, h, d), device=x.device, dtype=dtype)
                gate = g.view(n_i, n_j, h, d).permute(perm)
                if gated_attention(
                    q, k, v, bias, gate, o.permute(perm), key_mask=key_mask, scale=1.0 / math.sqrt(d)
                ):
                    o = o.view(n_i, n_j, hd)
                else:
                    o = None
            if o is None:
                o = cuequivariance_triangular_attn(
                    q, k, v, bias, key_mask.contiguous()[:, None, None, :], 1.0 / math.sqrt(d)
                )
                # [I, J, H*D] in both cases
                o = sigmoid_gate_transpose(o, g, transposed=ending)
            w_o = self.mha.linear_o.weight.to(dtype)
            if residual is not None and residual.dtype == dtype and residual.is_contiguous():
                c = w_o.shape[0]
                if inplace:
                    residual.view(-1, c).addmm_(o.view(-1, hd), w_o.t())
                    return residual
                return torch.addmm(residual.view(-1, c), o.view(-1, hd), w_o.t()).view(residual.shape)
            out = F.linear(o, w_o)
            return out if residual is None else residual + out

    def fused_residual_update(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor],
        triangle_attention: str,
        chunk_size: Optional[int],
        ending: bool,
        inplace_safe: bool = False,
    ) -> Optional[torch.Tensor]:
        """z + attention(z) at inference, for starting- or ending-node attention
        on the untransposed pair tensor, without transposing it. Returns None
        when the fast path does not apply (the caller then uses forward())."""
        if not (
            self.starting
            and self._fused_inference_ok(z, triangle_attention, chunk_size)
            and z.shape[0] == z.shape[1]
            and _tri_pad_amount(z) == 0
        ):
            return None
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        return self._fused_inference_forward(
            self.layer_norm(z), mask, ending=ending, residual=z, inplace=inplace_safe
        )

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                triangle_attention=triangle_attention,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        n_tok = x.shape[-2]
        pad = _tri_pad_amount(x) if triangle_attention == "cuequivariance" else 0
        if pad:
            x = F.pad(x, (0, 0, 0, pad, 0, pad))
            mask = F.pad(mask, (0, pad, 0, pad))

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        if self._fused_inference_ok(x, triangle_attention, chunk_size):
            x = self._fused_inference_forward(x, mask)
            if not self.starting:
                x = x.transpose(-2, -3)
            if pad:
                x = x[..., :n_tok, :n_tok, :]
            return x

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                triangle_attention=triangle_attention,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        if pad:
            x = x[..., :n_tok, :n_tok, :]

        return x


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
