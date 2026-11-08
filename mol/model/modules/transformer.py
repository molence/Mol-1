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
# Upstream file: protenix/model/modules/transformer.py
# Changes: Projected pair-bias cache shared across denoising steps and sample chunks.

import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mol.model.modules.primitives import (
    AdaptiveLayerNorm,
    Attention,
    BiasInitLinear,
    broadcast_token_to_local_atom_pair,
    DropPath,
    LinearNoBias,
    rearrange_qk_to_dense_trunk,
)
from mol.model.triangular.layers import LayerNorm
from mol.model.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    checkpoint_blocks,
    permute_final_dims,
)


@dataclass(frozen=True)
class ProjectedPairBiases:
    """Detached per-block pair biases prepared for a DiffusionTransformer."""

    blocks: tuple[torch.Tensor, ...]


# Fused LayerNorm + head projection for large pair biases at inference
# (pair_bias.py). MOL_FUSED_PAIR_BIAS=0 disables.
_FUSED_PAIR_BIAS = os.getenv("MOL_FUSED_PAIR_BIAS", "1") != "0"


class AttentionPairBias(nn.Module):
    """
    Implements Algorithm 24 in AF3

    Args:
        has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
        create_offset_ln_z (bool, optional): the value of create_offset for the LayerNorm applied to z. Defaults to False.
        n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
        c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
        cross_attention_mode (bool, optional): If cross_attention_model = True, the adaptive layernorm will be applied
            to query and key/value seperately. Defaults to False.
    """

    def __init__(
        self,
        has_s: bool = True,
        create_offset_ln_z: bool = False,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super(AttentionPairBias, self).__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        self.create_offset_ln_z = create_offset_ln_z
        self.cross_attention_mode = cross_attention_mode
        if has_s:
            # Line2
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            if self.cross_attention_mode:
                self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            self.layernorm_a = LayerNorm(c_a)
            if self.cross_attention_mode:
                self.layernorm_kv = LayerNorm(c_a)

        # Line 6-11
        self.local_attention_method = "local_cross_attention"
        self.attention = Attention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            local_attention_method=self.local_attention_method,
            zero_init=not self.has_s,  # Adaptive zero init
        )
        self.layernorm_z = LayerNorm(c_z, create_offset=self.create_offset_ln_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.linear_nobias_z = LinearNoBias(in_features=c_z, out_features=n_heads)

        # Line 13
        if self.has_s:
            self.linear_a_last = BiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )

    def project_pair_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Project raw channel-last pair features to attention-head biases.

        Args:
            z: Pair features with shape ``[..., N_query, N_key, c_z]``.

        Returns:
            Pair bias with shape ``[..., n_heads, N_query, N_key]``.
        """
        if z.ndim < 3 or z.shape[-1] != self.linear_nobias_z.in_features:
            raise ValueError(
                "z must be a raw channel-last pair tensor with shape "
                f"[..., N_query, N_key, {self.linear_nobias_z.in_features}]"
            )
        if (
            _FUSED_PAIR_BIAS
            and z.is_cuda
            and z.dim() == 3
            and not torch.is_grad_enabled()
            and z.shape[0] * z.shape[1] >= 65536
            and self.n_heads >= 16
            and self.n_heads & (self.n_heads - 1) == 0
        ):
            from mol.model.modules.pair_bias import layer_norm_head_projection

            return layer_norm_head_projection(
                z,
                self.layernorm_z.weight,
                self.layernorm_z.bias,
                self.linear_nobias_z.weight,
                self.layernorm_z.eps,
                out_dtype=(
                    torch.get_autocast_dtype("cuda")
                    if torch.is_autocast_enabled("cuda")
                    else z.dtype
                ),
            )
        bias = self.linear_nobias_z(self.layernorm_z(z))
        return permute_final_dims(bias, [2, 0, 1])

    def _project_pair_bias_from_normalized(self, z: torch.Tensor) -> torch.Tensor:
        """Project shared-normalized channel-first pair features.

        ``z`` is the output of an affine-free LayerNorm shared by all blocks,
        laid out as ``[..., c_z, N_query, N_key]``. This helper folds each
        block's LayerNorm affine parameters into a 1x1 convolution.
        """
        c_z = self.linear_nobias_z.in_features
        if z.ndim < 3 or z.shape[-3] != c_z:
            raise ValueError(
                "z must be a shared-normalized channel-first pair tensor with "
                f"shape [..., {c_z}, N_query, N_key]"
            )

        projection_weight = self.linear_nobias_z.weight
        layernorm_weight = self.layernorm_z.weight
        if layernorm_weight is not None:
            projection_weight = projection_weight * layernorm_weight[None, :]

        projection_bias = None
        layernorm_bias = self.layernorm_z.bias
        if layernorm_bias is not None:
            projection_bias = F.linear(
                layernorm_bias,
                self.linear_nobias_z.weight,
            )

        leading_shape = z.shape[:-3]
        n_queries, n_keys = z.shape[-2:]
        z_4d = z.reshape(-1, c_z, n_queries, n_keys)
        bias = F.conv2d(
            z_4d,
            projection_weight[:, :, None, None],
            bias=projection_bias,
        )
        return bias.reshape(leading_shape + (self.n_heads, n_queries, n_keys))

    def local_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            q (torch.Tensor): query embedding
                [..., N_atom, c_a]
            kv (torch.Tensor): key/value embedding
                [..., N_atom, c_a]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """

        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(q.shape) + 2

        # Multi-head attention bias
        bias = self.linear_nobias_z(
            self.layernorm_z(z)
        )  # [..., n_blocks, n_queries, n_keys, n_heads]
        bias = permute_final_dims(
            bias, [3, 0, 1, 2]
        )  # [..., n_heads, n_blocks, n_queries, n_keys]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        q = self.attention(
            q_x=q,
            kv_x=kv,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return q

    def standard_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
        enable_efficient_fusion: bool = False,
        pair_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Used by Algorithm 7/20

        Args:
            q (torch.Tensor): the query embedding
                [..., N_token, c_a]
            kv (torch.Tensor): the key/value embedding
                [..., N_token, c_a]
            z (torch.Tensor): pair embedding, used for computing pair bias.
                [..., N_token, N_token, c_z], or normalized channel-first
                [..., c_z, N_token, N_token] for the torch efficient path.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            enable_efficient_fusion (bool): Whether to enable efficient fusion of bias calculation in attention to speed up. Defaults to False. (Alg 24)
            pair_bias (torch.Tensor, optional): Precomputed pair bias with shape
                ``[..., n_heads, N_query, N_key]``. When supplied, projection
                from ``z`` is skipped.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """
        if pair_bias is not None:
            expected_pair_bias_shape = z.shape[:-3] + (
                self.n_heads,
                q.shape[-2],
                kv.shape[-2],
            )
            if pair_bias.shape != expected_pair_bias_shape:
                raise ValueError(
                    "pair_bias has shape "
                    f"{tuple(pair_bias.shape)}, expected "
                    f"{tuple(expected_pair_bias_shape)}"
                )
            if (
                pair_bias.device != q.device
                or pair_bias.device != kv.device
                or pair_bias.device != z.device
            ):
                raise ValueError("pair_bias must be on the same device as q, kv, and z")

        return self._torch_standard_multihead_attention(
            q=q,
            kv=kv,
            z=z,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=enable_efficient_fusion,
            pair_bias=pair_bias,
        )

    def _torch_standard_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool,
        enable_efficient_fusion: bool,
        pair_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Reference PyTorch SDPA implementation."""
        if pair_bias is None:
            if enable_efficient_fusion:
                pair_bias = self._project_pair_bias_from_normalized(z)
            else:
                pair_bias = self.project_pair_bias(z)

        return self.attention(
            q_x=q,
            kv_x=kv,
            attn_bias=pair_bias,
            inplace_safe=inplace_safe,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
        pair_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Details are given in local_forward and standard_forward"""
        # Input projections
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        if self.cross_attention_mode:
            if self.has_s:
                kv = self.layernorm_kv(a=a, s=s)
            else:
                kv = self.layernorm_kv(a)
        else:
            kv = None

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
                pair_bias=pair_bias,
            )

        # Output projection (from adaLN-Zero [27])
        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a

        return a


class DiffusionTransformerBlock(nn.Module):
    """
    Implements Algorithm 23[Line2-Line3] in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_heads (int): number of heads for DiffusionTransformerBlock.
        biasinit (float, optional): bias initialization value. Defaults to -2.0.
        drop_path_rate (float, optional): drop path rate. Defaults to 0.0.
        cross_attention_mode (bool, optional): whether to use cross attention. Defaults to False.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_heads: int,  # could be 16 or 4 or ... in AF3
        biasinit: float = -2.0,
        drop_path_rate: float = 0.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super(DiffusionTransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.attention_pair_bias = AttentionPairBias(
            has_s=True,
            create_offset_ln_z=False,
            n_heads=n_heads,
            c_a=c_a,
            c_s=c_s,
            c_z=c_z,
            biasinit=biasinit,
            cross_attention_mode=cross_attention_mode,
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
        pair_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]
            z (torch.Tensor): pair embedding
                [..., N, N, c_z] or [..., n_block, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            enable_efficient_fusion (bool): Whether to enable efficient fusion of bias calculation in attention to speed up. Defaults to False. (Alg 24)
            pair_bias (torch.Tensor, optional): Precomputed pair bias for this
                block. Global attention only.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - out_a: the output of DiffusionTransformerBlock [..., N, c_a]
                - s: the single embedding [..., N, c_s]
                - z: the pair embedding
        """
        attn_out = self.drop_path(
            self.attention_pair_bias(
                a=a,
                s=s,
                z=z,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
                pair_bias=pair_bias,
            )
        )
        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a
        ff_out = self.drop_path(self.conditioned_transition_block(a=attn_out, s=s))
        out_a = ff_out + attn_out
        # Avoid s/z to be deleted by torch.utils.checkpoint
        return out_a, s, z


class DiffusionTransformer(nn.Module):
    """
    Implements Algorithm 23 in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_blocks (int): number of blocks in DiffusionTransformer.
        n_heads (int): number of heads in attention.
        cross_attention_mode (bool, optional): whether to use cross attention. Defaults to False.
        drop_path_rate (float, optional): drop skip connection path rate. Defaults to 0.0.
        blocks_per_ckpt (int, optional): number of DiffusionTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_blocks: int,  # could be 3 or 24 in AF3
        n_heads: int,  # could be 16 or 4 or ... in AF3
        cross_attention_mode: bool = False,
        drop_path_rate: float = 0.0,  # drop skip connection path
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(DiffusionTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        drop_path_rates = [
            drop_path_value.item()
            for drop_path_value in torch.linspace(0, drop_path_rate, n_blocks)
        ]
        for i in range(n_blocks):
            block = DiffusionTransformerBlock(
                n_heads=n_heads,
                c_a=c_a,
                c_s=c_s,
                c_z=c_z,
                cross_attention_mode=cross_attention_mode,
                drop_path_rate=drop_path_rates[i],
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
        projected_pair_biases: Optional[ProjectedPairBiases] = None,
    ) -> list[Callable]:
        blocks = [
            partial(
                b,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
                pair_bias=(
                    None
                    if projected_pair_biases is None
                    else projected_pair_biases.blocks[i]
                ),
            )
            for i, b in enumerate(self.blocks)
        ]
        return blocks

    @torch.no_grad()
    def prepare_pair_biases(
        self,
        z: torch.Tensor,
    ) -> ProjectedPairBiases:
        """Prepare one detached pair-bias tensor per block from raw ``z``.

        Args:
            z: Raw channel-last pair features
                ``[..., N_query, N_key, c_z]``.

        Returns:
            An immutable container whose tensors have canonical
            ``[..., n_heads, N_query, N_key]`` layout.
        """
        return self._prepare_pair_biases(z, shared_normalized=False)

    @torch.no_grad()
    def _prepare_pair_biases_from_normalized(
        self,
        z: torch.Tensor,
    ) -> ProjectedPairBiases:
        """Prepare biases from shared-normalized channel-first pair features."""
        return self._prepare_pair_biases(z, shared_normalized=True)

    def _prepare_pair_biases(
        self,
        z: torch.Tensor,
        shared_normalized: bool,
    ) -> ProjectedPairBiases:
        pair_biases = []
        for block in self.blocks:
            attention_pair_bias = block.attention_pair_bias
            if shared_normalized:
                pair_bias = attention_pair_bias._project_pair_bias_from_normalized(z)
            else:
                pair_bias = attention_pair_bias.project_pair_bias(z)
            pair_biases.append(pair_bias.detach().contiguous())
        return ProjectedPairBiases(blocks=tuple(pair_biases))

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
        projected_pair_biases: Optional[ProjectedPairBiases] = None,
    ) -> torch.Tensor:
        """Apply all Diffusion Transformer blocks.

        ``projected_pair_biases`` is an immutable, request-scoped projection of
        ``z``. It is consumed only in evaluation mode; training always computes
        the projection from ``z`` to preserve parameter and input gradients.

        Args:
            a (torch.Tensor): Per-atom representation ``[..., N, c_a]``.
            s (torch.Tensor): Single embedding ``[..., N, c_s]``.
            z (torch.Tensor): Raw channel-last pair embedding
                ``[..., N, N, c_z]``. The explicit torch efficient path instead
                receives shared-normalized channel-first ``[..., c_z, N, N]``.
            n_queries (int, optional): Local query-window size.
            n_keys (int, optional): Local key-window size.
            enable_efficient_fusion (bool): Use the channel-first conv2d pair
                projection in the PyTorch attention path.
            projected_pair_biases (ProjectedPairBiases, optional):
                Request-scoped projected pair biases, exactly one for each
                Diffusion Transformer block.

        Returns:
            torch.Tensor: Updated representation ``[..., N, c_a]``.
        """
        if projected_pair_biases is not None:
            if not isinstance(projected_pair_biases, ProjectedPairBiases):
                raise TypeError("projected_pair_biases must be a ProjectedPairBiases")
            if len(projected_pair_biases.blocks) != self.n_blocks:
                raise ValueError(
                    "projected_pair_biases must contain exactly one tensor per "
                    f"DiffusionTransformer block ({self.n_blocks})"
                )
            if not all(
                isinstance(pair_bias, torch.Tensor)
                for pair_bias in projected_pair_biases.blocks
            ):
                raise TypeError(
                    "every ProjectedPairBiases.blocks entry must be a tensor"
                )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled() or (
            projected_pair_biases is not None and not self.training
        ):
            blocks_per_ckpt = None

        # Never reuse detached pair features in training. Explicit eval-time
        # projections disable activation checkpointing above, so a caller can
        # retain gradients to q/k/v while treating the pair projection as a
        # request-constant inference input.
        pair_biases_for_blocks = None if self.training else projected_pair_biases

        from mol.model.modules import diffusion_fast

        if diffusion_fast.fused_ok(self, a, s):
            return diffusion_fast.fused_forward(
                self, a, s, z, n_queries, n_keys, pair_biases_for_blocks, chunk_size,
                enable_efficient_fusion,
            )

        blocks = self._prep_blocks(
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            enable_efficient_fusion=enable_efficient_fusion,
            projected_pair_biases=pair_biases_for_blocks,
        )
        a, s, z = checkpoint_blocks(
            blocks, args=(a, s, z), blocks_per_ckpt=blocks_per_ckpt
        )
        del s, z
        return a


class AtomTransformer(nn.Module):
    """
    Implements Algorithm 7 in AF3

    Performs local transformer among atom embeddings, with bias predicted from atom pair embeddings

    Args:
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        n_blocks (int, optional): number of block in AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads in attention. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomTransformer/DiffusionTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atompair,
            cross_attention_mode=True,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p (torch.Tensor): atompair embedding in dense block shape.
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the output of AtomTransformer
                [..., N_atom, c_atom]
        """
        n_blocks, n_queries, n_keys = p.shape[-4:-1]

        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )


class ConditionedTransitionBlock(nn.Module):
    """
    Implements Algorithm 25 in AF3

    Args:
        c_a (int): single embedding dim (single feature aggregated atom info).
        c_s (int):  single embedding dim.
        n (int, optional): channel scale factor. Defaults to 2.
        biasinit (float, optional): bias initialization value. Defaults to -2.0.
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        super(ConditionedTransitionBlock, self).__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.n = n
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = LinearNoBias(
            in_features=c_a, out_features=n * c_a, initializer="relu"
        )
        self.linear_nobias_a2 = LinearNoBias(
            in_features=c_a, out_features=n * c_a, initializer="relu"
        )
        self.linear_nobias_b = LinearNoBias(in_features=n * c_a, out_features=c_a)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]

        Returns:
            torch.Tensor: the updated a from ConditionedTransitionBlock
                [..., N, c_a]
        """
        a = self.adaln(a, s)
        b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
        # Output projection (from adaLN-Zero [27])
        a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
        return a


class AtomAttentionEncoder(nn.Module):
    """
    Implements Algorithm 5 in AF3

    Args:
        has_coords (bool): whether the module input will contains coordinates (r_l).
        c_token (int): token embedding dim.
        c_atom (int, optional): atom embedding dim. Defaults to 128.
        c_atompair (int, optional): atompair embedding dim. Defaults to 16.
        c_s (int, optional):  single embedding dim. Defaults to 384.
        c_z (int, optional): pair embedding dim. Defaults to 128.
        n_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads in AtomTransformer. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomAttentionEncoder/AtomTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        has_coords: bool,
        c_token: int,  # 384 or 768
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomAttentionEncoder, self).__init__()
        self.has_coords = has_coords
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.local_attention_method = "local_cross_attention"

        self.input_feature = {
            # "ref_pos": 3,
            # "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.linear_no_bias_ref_pos = LinearNoBias(
            in_features=3, out_features=self.c_atom, precision=torch.float32
        )  # use high precision for ref_pos
        self.linear_no_bias_ref_charge = LinearNoBias(
            in_features=1, out_features=self.c_atom
        )
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair, precision=torch.float32
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

        if self.has_coords:
            # Line9
            self.layernorm_s = LayerNorm(self.c_s, create_offset=False)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s,
                out_features=self.c_atom,
                initializer="zeros",
                precision=torch.float32,
            )
            # Line10
            self.layernorm_z = LayerNorm(
                self.c_z, create_offset=False
            )  # memory bottleneck
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z,
                out_features=self.c_atompair,
                initializer="zeros",
                precision=torch.float32,
            )
            # Line11
            self.linear_no_bias_r = LinearNoBias(
                in_features=3, out_features=self.c_atom, precision=torch.float32
            )
        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="relu",
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="relu",
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="zeros",
            ),
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def prepare_cache(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: torch.Tensor,
        r_l: Union[torch.Tensor, bool, None] = None,
        z: torch.Tensor = None,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = ref_pos.shape[:-2]
        N_atom = ref_pos.shape[-2]
        c_l = self.linear_no_bias_ref_pos(ref_pos) + self.linear_no_bias_ref_charge(
            # use arcsinh for ref_charge
            torch.arcsinh(ref_charge).reshape(*batch_shape, N_atom, 1)
        )
        if inplace_safe:
            c_l += self.linear_no_bias_f(
                torch.cat(
                    [
                        ref_mask.reshape(*batch_shape, N_atom, 1),
                        ref_element.reshape(*batch_shape, N_atom, 128),
                        ref_atom_name_chars.reshape(*batch_shape, N_atom, 4 * 64),
                    ],
                    dim=-1,
                ).to(dtype=c_l.dtype)
            )
            c_l *= ref_mask.reshape(*batch_shape, N_atom, 1)
        else:
            c_l = c_l + self.linear_no_bias_f(
                torch.cat(
                    [
                        ref_mask.reshape(*batch_shape, N_atom, 1),
                        ref_element.reshape(*batch_shape, N_atom, 128),
                        ref_atom_name_chars.reshape(*batch_shape, N_atom, 4 * 64),
                    ],
                    dim=-1,
                ).to(dtype=c_l.dtype)
            )
            c_l = c_l * ref_mask.reshape(*batch_shape, N_atom, 1)

        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info[
            "mask_trunked"
        ].unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, C_atompair]

        # Line5-Line6: Embed pairwise inverse squared distances, and the valid mask
        if inplace_safe:
            p_lm += (
                self.linear_no_bias_invd(1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True)))
                * v_lm
            )
            p_lm += self.linear_no_bias_v(
                v_lm.to(dtype=p_lm.dtype)
            )  # not multipling v_lm
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_invd(
                    1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
                )
                * v_lm
            )
            p_lm = p_lm + self.linear_no_bias_v(
                v_lm.to(dtype=p_lm.dtype)
            )  # not multipling v_lm

        # Line7: Initialise the atom single representation as the single conditioning
        # q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        if r_l is not None:
            p_lm = (
                p_lm.unsqueeze(dim=-5)
                + broadcast_token_to_local_atom_pair(
                    z_token=self.linear_no_bias_z(self.layernorm_z(z)),
                    atom_to_token_idx=atom_to_token_idx,
                    n_queries=self.n_queries,
                    n_keys=self.n_keys,
                    compute_mask=False,
                )[0]
            )  # [..., N_sample, n_blocks, n_queries, n_keys, c_atompair]
        return p_lm, c_l

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_element: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: torch.Tensor,
        r_l: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor = None,
        p_lm: torch.Tensor = None,
        c_l: torch.Tensor = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_to_token_idx (torch.Tensor): atom_to_token_idx
            ref_pos (torch.Tensor): ref_pos
            ref_charge (torch.Tensor): ref_charge
            ref_mask (torch.Tensor): ref_mask
            ref_atom_name_chars (torch.Tensor): ref_atom_name_chars
            ref_element (torch.Tensor): ref_element
            r_l (torch.Tensor, optional): noisy position.
                [..., N_sample, N_atom, 3] if has_coords else None.
            s (torch.Tensor, optional): single embedding.
                [..., N_sample, N_token, c_s] if has_coords else None.
            z (torch.Tensor, optional): pair embedding
                [..., N_sample, N_token, N_token, c_z] if has_coords else None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: the output of AtomAttentionEncoder
            a:
                [..., (N_sample), N_token, c_token]
            q_l:
                [..., (N_sample), N_atom, c_atom]
            c_l:
                [..., (N_sample), N_atom, c_atom]
            p_lm:
                [..., (N_sample), N_atom, N_atom, c_atompair]

        """

        if self.has_coords:
            assert r_l is not None
            assert s is not None
            assert z is not None

        if p_lm is None or c_l is None:
            p_lm, c_l = self.prepare_cache(
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=ref_mask,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_element=ref_element,
                atom_to_token_idx=atom_to_token_idx,
                d_lm=d_lm,
                v_lm=v_lm,
                pad_info=pad_info,
                r_l=r_l,
                z=z,
                inplace_safe=inplace_safe,
            )
        else:
            if inplace_safe:
                p_lm_clone = p_lm.clone()
                c_l_clone = c_l.clone()
                p_lm = p_lm_clone
                c_l = c_l_clone

        # Line7: Initialise the atom single representation as the single conditioning
        # q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        n_token = None
        if r_l is not None:
            # Broadcast the single and pair embedding from the trunk
            n_token = s.size(-2)
            c_l = c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=atom_to_token_idx,
            )  # [..., N_sample, N_atom, c_atom]

            # Add the noisy positions
            # Different from paper!!
            q_l = c_l + self.linear_no_bias_r(r_l)  # [..., N_sample, N_atom, c_atom]
        else:
            q_l = c_l.clone()

        # Add the combined single conditioning to the pair representation
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
            dim_q=-2,
            dim_k=-2,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        if inplace_safe:
            p_lm = p_lm + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            p_lm += self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            p_lm += self.small_mlp(p_lm)
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
                + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            )  # [..., (N_sample), n_blocks, n_queries, n_keys, c_atompair]

            # Run a small MLP on the pair activations
            p_lm = p_lm + self.small_mlp(p_lm)

        # Cross attention transformer
        q_l = self.atom_transformer(
            q_l, c_l, p_lm, chunk_size=chunk_size
        )  # [..., (N_sample), N_atom, c_atom]

        # Aggregate per-atom representation to per-token representation
        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )  # [..., (N_sample), N_token, c_token]
        return a, q_l, c_l, p_lm


class AtomAttentionDecoder(nn.Module):
    """
    Implements Algorithm 6 in AF3

    Args:
        n_blocks (int, optional): number of blocks for AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads for AtomTransformer. Defaults to 4.
        c_token (int, optional): feature channel of token (single a). Defaults to 384.
        c_atom (int, optional): embedding dim for atom embedding. Defaults to 128.
        c_atompair (int, optional): embedding dim for atom pair embedding. Defaults to 16.
        n_queries (int, optional): local window size of query tensor. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomAttentionDecoder/AtomTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomAttentionDecoder, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.linear_no_bias_a = LinearNoBias(in_features=c_token, out_features=c_atom)
        self.layernorm_q = LayerNorm(c_atom, create_offset=False)
        self.linear_no_bias_out = LinearNoBias(
            in_features=c_atom, out_features=3, precision=torch.float32
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            atom_to_token_idx (torch.Tensor): the atom to token index
                [..., N_atom]
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_token]
            q_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p_skip (torch.Tensor): atompair single embedding
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the updated noisy coordinates
                [..., N_atom, 3]
        """
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        q = (
            broadcast_token_to_atom(
                x_token=self.linear_no_bias_a(a),  # [..., N_token, c_atom]
                atom_to_token_idx=atom_to_token_idx,
            )  # [..., N_atom, c_atom]
            + q_skip
        )

        # Cross attention transformer
        q = self.atom_transformer(
            q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size
        )

        # Map to positions update
        r = self.linear_no_bias_out(self.layernorm_q(q))

        return r
