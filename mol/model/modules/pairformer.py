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
# Upstream file: protenix/model/modules/pairformer.py
# Changes: Added the PairMixer block and the width adapter into the Pairformer stack; MSA row chunking.

# pylint: disable=C0114
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mol.data.constants import STD_RESIDUES_WITH_GAP
from mol.model.modules.primitives import LinearNoBias, Transition
from mol.model.modules.transformer import AttentionPairBias
from mol.model.modules.fused_ops import dropout_add_rowwise
from mol.model.triangular.layers import DropoutRowwise, LayerNorm, OuterProductMean
from mol.model.triangular.triangular import _tri_pad_amount
from mol.model.triangular.triangular import (
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from mol.model.utils import (
    checkpoint_blocks,
    expand_at_dim,
    get_checkpoint_fn,
    sample_msa_feature_dict_random_without_replacement,
)


def _unpad_pair(z: torch.Tensor, n_token: int) -> torch.Tensor:
    """View of the real tokens of a pair tensor padded by PairformerStack."""
    if z.shape[-2] == n_token:
        return z
    return z[..., :n_token, :n_token, :]


class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3

    c_hidden_mul is set as openfold
    Ref to:
    https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/evoformer.py#L123

    Args:
        n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
            Defaults to 128.
        c_hidden_pair_att (int, optional): hidden dim [for TriangleAttention]. Defaults to 32.
        no_heads_pair (int, optional): number of head [for TriangleAttention]. Defaults to 4.
        num_intermediate_factor (int, optional): number of intermediate factor for pair_transition. Defaults to 4.
        dropout (float, optional): dropout ratio [for TriangleUpdate]. Defaults to 0.25.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        dropout: float = 0.25,
        hidden_scale_up: bool = False,
    ) -> None:
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        if hidden_scale_up:
            no_heads_pair = c_z // c_hidden_pair_att
            c_hidden_mul = c_z
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_att_start = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.tri_att_end = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.p_drop = dropout
        self.pair_transition = Transition(c_in=c_z, n=num_intermediate_factor)
        self.c_s = c_s
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False,
                create_offset_ln_z=True,
                n_heads=n_heads,
                c_a=c_s,
                c_z=c_z,
            )
            self.single_transition = Transition(c_in=c_s, n=4)

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        pair_transition_chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": Cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: the update of s[Optional] and z
                [..., N_token, c_s] | None
                [..., N_token, N_token, c_z]
        """
        if inplace_safe:
            z = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
                triangle_multiplicative=triangle_multiplicative,
            )
            z += self.tri_att_start(
                z,
                mask=pair_mask,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.pair_transition(z, chunk_size=pair_transition_chunk_size)
        else:
            tmu_update = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=False,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = dropout_add_rowwise(z, tmu_update, self.p_drop, self.training)
            del tmu_update
            tmu_update = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=False,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = dropout_add_rowwise(z, tmu_update, self.p_drop, self.training)
            del tmu_update
            z = dropout_add_rowwise(
                z,
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    triangle_attention=triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                ),
                self.p_drop,
                self.training,
            )
            z = z.transpose(-2, -3).contiguous()
            z = dropout_add_rowwise(
                z,
                self.tri_att_end(
                    z,
                    mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                    triangle_attention=triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                ),
                self.p_drop,
                self.training,
            )
            z = z.transpose(-2, -3).contiguous()

            z = z + self.pair_transition(z, chunk_size=pair_transition_chunk_size)
        if self.c_s > 0:
            s = s + self.attention_pair_bias(
                a=s,
                s=None,
                z=_unpad_pair(z, s.shape[-2]),
            )
            s = s + self.single_transition(s)
        return s, z


class PairMixerBlock(nn.Module):
    """Pairformer-style block without triangle attention.

    PairMixer keeps the triangle multiplication, pair transition, and single
    representation updates from :class:`PairformerBlock`.  Its forward
    signature intentionally matches ``PairformerBlock`` so the two block types
    can share one checkpointed stack.
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        num_intermediate_factor: int = 4,
        dropout: float = 0.25,
        hidden_scale_up: bool = False,
    ) -> None:
        super(PairMixerBlock, self).__init__()
        self.n_heads = n_heads
        if hidden_scale_up:
            c_hidden_mul = c_z
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_mul)
        self.dropout_row = DropoutRowwise(dropout)
        self.p_drop = dropout
        self.pair_transition = Transition(c_in=c_z, n=num_intermediate_factor)
        self.c_s = c_s
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False,
                create_offset_ln_z=True,
                n_heads=n_heads,
                c_a=c_s,
                c_z=c_z,
            )
            self.single_transition = Transition(c_in=c_s, n=4)

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        pair_transition_chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Update pair and single features without triangle attention.

        ``triangle_attention`` and ``chunk_size`` are accepted only to keep the
        same callable interface as :class:`PairformerBlock`.
        """
        del triangle_attention, chunk_size

        if inplace_safe:
            z = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
                triangle_multiplicative=triangle_multiplicative,
            )
            z += self.pair_transition(z, chunk_size=pair_transition_chunk_size)
        else:
            tmu_update = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=False,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = dropout_add_rowwise(z, tmu_update, self.p_drop, self.training)
            del tmu_update
            tmu_update = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=False,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = dropout_add_rowwise(z, tmu_update, self.p_drop, self.training)
            del tmu_update
            z = z + self.pair_transition(z, chunk_size=pair_transition_chunk_size)

        if self.c_s > 0:
            s = s + self.attention_pair_bias(
                a=s,
                s=None,
                z=_unpad_pair(z, s.shape[-2]),
            )
            s = s + self.single_transition(s)
        return s, z


# Concise public name for callers that treat PairMixer as a layer rather than a
# Pairformer block variant.
PairMixer = PairMixerBlock


class PairformerWidthAdapter(nn.Module):
    """Project the pair representation between trunk widths."""

    def __init__(
        self,
        c_z_in: int,
        c_z_out: int,
    ) -> None:
        super().__init__()
        self.pair_adapter = (
            LinearNoBias(c_z_in, c_z_out) if c_z_in != c_z_out else nn.Identity()
        )

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        **_: Any,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        return s, self.pair_adapter(z)


class PairformerStack(nn.Module):
    """
    Implements Algorithm 17 [PairformerStack] in AF3

    Args:
        n_blocks (int, optional): total number of blocks in the Pairformer trunk.
            Defaults to 48.
        n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
        c_z (int, optional): resident pair width at the stack input/output. It
            is also the width used by recycling and downstream heads. Defaults
            to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        num_intermediate_factor (int, optional): number of intermediate factor for transition. Defaults to 4.
        dropout (float, optional): dropout ratio. Defaults to 0.25.
        blocks_per_ckpt (int, optional): number of Pairformer blocks in each activation checkpoint. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
        n_pairmixer_blocks (int, optional): number of leading PairMixer blocks.
            All remaining blocks are complete Pairformer blocks. Defaults to 0.
        c_z_pairmixer (int, optional): pair width used by PairMixer/U blocks.
            ``None`` reuses ``c_z``. Defaults to None.
        c_z_pairformer (int, optional): pair width used by complete
            Pairformer/F blocks. ``None`` reuses ``c_z``. Defaults to None.
    """

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        num_intermediate_factor: int = 4,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
        hidden_scale_up: bool = False,
        n_pairmixer_blocks: int = 0,
        c_z_pairmixer: Optional[int] = None,
        c_z_pairformer: Optional[int] = None,
    ) -> None:
        super(PairformerStack, self).__init__()
        if isinstance(n_pairmixer_blocks, bool) or not isinstance(
            n_pairmixer_blocks, int
        ):
            raise TypeError(
                "n_pairmixer_blocks must be an integer, "
                f"got {type(n_pairmixer_blocks).__name__}"
            )
        if not 0 <= n_pairmixer_blocks <= n_blocks:
            raise ValueError(
                "n_pairmixer_blocks must satisfy "
                f"0 <= n_pairmixer_blocks <= n_blocks, got "
                f"n_pairmixer_blocks={n_pairmixer_blocks}, n_blocks={n_blocks}"
            )
        resident_c_z = c_z
        pairmixer_c_z = resident_c_z if c_z_pairmixer is None else c_z_pairmixer
        pairformer_c_z = resident_c_z if c_z_pairformer is None else c_z_pairformer
        if isinstance(pairmixer_c_z, bool) or not isinstance(pairmixer_c_z, int):
            raise TypeError(
                "c_z_pairmixer must be an integer, "
                f"got {type(pairmixer_c_z).__name__}"
            )
        if isinstance(pairformer_c_z, bool) or not isinstance(pairformer_c_z, int):
            raise TypeError(
                "c_z_pairformer must be an integer, "
                f"got {type(pairformer_c_z).__name__}"
            )
        if pairmixer_c_z <= 0:
            raise ValueError(f"c_z_pairmixer must be positive, got {pairmixer_c_z}")
        if pairformer_c_z <= 0:
            raise ValueError(f"c_z_pairformer must be positive, got {pairformer_c_z}")
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_s = c_s
        self.c_z = c_z
        self.c_z_pairmixer = pairmixer_c_z
        self.c_z_pairformer = pairformer_c_z
        self.blocks_per_ckpt = blocks_per_ckpt
        self.n_pairmixer_blocks = n_pairmixer_blocks
        self.n_pairformer_blocks = n_blocks - n_pairmixer_blocks
        self.blocks = nn.ModuleList()
        self.adapters = nn.ModuleDict()
        self._adapter_before_block: dict[int, str] = {}

        # The stack always starts and ends at the resident pair width. Adapters
        # are materialized only at boundaries whose pair widths differ.
        current_c_z = resident_c_z

        for block_idx in range(n_blocks):
            block_cls = (
                PairMixerBlock if block_idx < n_pairmixer_blocks else PairformerBlock
            )
            block_c_z = pairmixer_c_z if block_cls is PairMixerBlock else pairformer_c_z
            if current_c_z != block_c_z:
                adapter_name = f"before_block_{block_idx}"
                self.adapters[adapter_name] = PairformerWidthAdapter(
                    current_c_z, block_c_z
                )
                self._adapter_before_block[block_idx] = adapter_name
                current_c_z = block_c_z
            block = block_cls(
                n_heads=n_heads,
                c_z=block_c_z,
                c_s=c_s,
                num_intermediate_factor=num_intermediate_factor,
                dropout=dropout,
                hidden_scale_up=hidden_scale_up,
            )
            self.blocks.append(block)

        self._output_adapter_name: Optional[str] = None
        if current_c_z != resident_c_z:
            self._output_adapter_name = "after_blocks"
            self.adapters[self._output_adapter_name] = PairformerWidthAdapter(
                current_c_z, resident_c_z
            )

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        pairmixer_transition_chunk_size: Optional[int] = None,
        pairformer_transition_chunk_size: Optional[int] = None,
    ):
        block_kwargs = {
            "pair_mask": pair_mask,
            "triangle_multiplicative": triangle_multiplicative,
            "triangle_attention": triangle_attention,
            "inplace_safe": inplace_safe,
            "chunk_size": chunk_size,
        }
        blocks = []
        for block_idx, block in enumerate(self.blocks):
            adapter_name = self._adapter_before_block.get(block_idx)
            if adapter_name is not None:
                blocks.append(partial(self.adapters[adapter_name], **block_kwargs))
            transition_chunk_size = (
                pairmixer_transition_chunk_size
                if isinstance(block, PairMixerBlock)
                else pairformer_transition_chunk_size
            )
            blocks.append(
                partial(
                    block,
                    pair_transition_chunk_size=transition_chunk_size,
                    **block_kwargs,
                )
            )
        if self._output_adapter_name is not None:
            blocks.append(
                partial(self.adapters[self._output_adapter_name], **block_kwargs)
            )
        return blocks

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        pairmixer_transition_chunk_size: Optional[int] = None,
        pairformer_transition_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the update of s and z
                [..., N_token, c_s]
                [..., N_token, N_token, c_z]
        """
        # Pad the pair tensor once for the whole stack instead of in every
        # triangle op (see _tri_pad_amount); padded pairs are masked out.
        n_token = z.shape[-2]
        pad = (
            _tri_pad_amount(z)
            if z.dim() == 3
            and "cuequivariance" in (triangle_multiplicative, triangle_attention)
            else 0
        )
        if pad:
            z = F.pad(z, (0, 0, 0, pad, 0, pad))
            if pair_mask is None:
                pair_mask = z.new_ones((n_token, n_token))
            pair_mask = F.pad(pair_mask, (0, pad, 0, pad))

        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            pairmixer_transition_chunk_size=pairmixer_transition_chunk_size,
            pairformer_transition_chunk_size=pairformer_transition_chunk_size,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        if pad:
            z = z[..., :n_token, :n_token, :].contiguous()
        return s, z


class MSAPairWeightedAveraging(nn.Module):
    """
    Implements Algorithm 10 [MSAPairWeightedAveraging] in AF3

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c (int, optional): hidden dim [for MSAPairWeightedAveraging]. Defaults to 32.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        n_heads (int, optional): number of heads [for MSAPairWeightedAveraging]. Defaults to 8.
    """

    def __init__(
        self, c_m: int = 64, c: int = 32, c_z: int = 128, n_heads: int = 8
    ) -> None:
        super(MSAPairWeightedAveraging, self).__init__()
        self.c_m = c_m
        self.c = c
        self.n_heads = n_heads
        self.c_z = c_z
        # Input projections
        self.layernorm_m = LayerNorm(self.c_m)
        self.linear_no_bias_mv = LinearNoBias(
            in_features=self.c_m, out_features=self.c * self.n_heads
        )
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=self.c_z, out_features=self.n_heads
        )
        self.linear_no_bias_mg = LinearNoBias(
            in_features=self.c_m,
            out_features=self.c * self.n_heads,
            initializer="zeros",
        )
        # Weighted average with gating
        self.softmax_w = nn.Softmax(dim=-2)
        # Output projection
        self.linear_no_bias_out = LinearNoBias(
            in_features=self.c * self.n_heads,
            out_features=self.c_m,
            initializer="zeros",
        )

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]
        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        # Input projections
        m = self.layernorm_m(m)  # [...,n_msa_sampled, n_token, c_m]
        v = self.linear_no_bias_mv(m)  # [...,n_msa_sampled, n_token, n_heads * c]
        v = v.reshape(
            *v.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        b = self.linear_no_bias_z(
            self.layernorm_z(z)
        )  # [...,n_token, n_token, n_heads]
        g = torch.sigmoid(
            self.linear_no_bias_mg(m)
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        g = g.reshape(
            *g.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        w = self.softmax_w(b)  # [...,n_token, n_token, n_heads]
        wv = torch.einsum(
            "...ijh,...mjhc->...mihc", w, v
        )  # [...,n_msa_sampled,n_token,n_heads,c]
        o = g * wv
        o = o.reshape(
            *o.shape[:-2], self.n_heads * self.c
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        m = self.linear_no_bias_out(o)  # [...,n_msa_sampled, n_token, c_m]
        if (not self.training) and m.shape[-3] > 5120:
            del v, b, g, w, wv, o
        return m


class MSAStack(nn.Module):
    """
    Implements MSAStack Line7-Line8 in Algorithm 8

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c (int, optional): hidden [for MSAStack] dim. Defaults to 8.
        dropout (float, optional): dropout ratio. Defaults to 0.15.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): deprecated compatibility argument. The stack
            no longer pads MSA activations to this size.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c: int = 8,
        dropout: float = 0.15,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
    ) -> None:
        super(MSAStack, self).__init__()
        self.c = c
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            c_m=c_m, c=self.c, c_z=c_z
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.p_drop = dropout
        self.transition_m = Transition(c_in=c_m, n=4)
        self.msa_chunk_size = msa_chunk_size
        self.msa_max_size = msa_max_size

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]

        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        chunk_size = self.msa_chunk_size if msa_chunk_size is None else msa_chunk_size
        if self.training:
            msa_pair_weighted = self.chunk_forward(
                self.msa_pair_weighted_averaging, m, z, chunk_size
            )
            m = dropout_add_rowwise(m, msa_pair_weighted, self.p_drop, self.training)
            m_transition = self.chunk_forward(self.transition_m, m, None, chunk_size)
            m = m + m_transition
        else:
            m = self.inference_forward(m, z, chunk_size)
        return m

    def chunk_forward(
        self,
        module: nn.Module,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        chunk_size: Optional[int] = 2048,
    ) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [..., n_msa_sampled, n_token, c_m]
            z (Optional[torch.Tensor]): pair embedding, if required by the module
                [..., n_token, n_token, c_z]
            chunk_size (Optional[int]): size of each chunk; None disables chunking

        Returns:
            torch.Tensor: updated msa embedding
                [..., n_msa_sampled, n_token, c_m]
        """

        checkpoint_fn = get_checkpoint_fn()
        # torch.chunk's second argument is the number of chunks, whereas
        # torch.split accepts the desired size of each chunk.
        m_chunks = (m,) if chunk_size is None else torch.split(m, chunk_size, dim=-3)

        # Process each chunk with gradient checkpointing
        if z is not None:
            processed_chunks = [checkpoint_fn(module, chunk, z) for chunk in m_chunks]
        else:
            processed_chunks = [checkpoint_fn(module, chunk) for chunk in m_chunks]
        # Concatenate the processed chunks back together
        m = torch.cat(processed_chunks, dim=-3)
        return m

    def inference_forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: Optional[int] = 2048,
    ) -> torch.Tensor:
        """Inplace slice forward for saving memory
        Args:
            m (torch.Tensor): msa embedding
                [..., n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [..., n_token, n_token, c_z]
            chunk_size (Optional[int]): size of each chunk; None disables chunking

        Returns:
            torch.Tensor: updated msa embedding
                [..., n_msa_sampled, n_token, c_m]
        """
        num_msa = m.shape[-3]
        if chunk_size is None:
            m += self.msa_pair_weighted_averaging(m, z)
            m += self.transition_m(m)
            return m
        no_chunks = num_msa // chunk_size + (num_msa % chunk_size != 0)
        for i in range(no_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_msa)
            # Use inplace to save memory
            m[..., start:end, :, :] += self.msa_pair_weighted_averaging(
                m[..., start:end, :, :], z
            )
            m[..., start:end, :, :] += self.transition_m(m[..., start:end, :, :])
        return m


class MSABlock(nn.Module):
    """
    Base MSA Block, Line6-Line13 in Algorithm 8

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_hidden (int, optional): hidden dim [for MSABlock]. Defaults to 32.
        is_last_block (bool, optional): if this is the last block of MSAModule. Defaults to False.
        msa_dropout (float, optional): dropout ratio for msa block. Defaults to 0.15.
        pair_dropout (float, optional): dropout ratio for pair stack. Defaults to 0.25.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): deprecated compatibility argument; no longer
            used to pad MSA activations. Defaults to 16384.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c_hidden: int = 32,
        is_last_block: bool = False,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
        hidden_scale_up: bool = False,
    ) -> None:
        super(MSABlock, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.is_last_block = is_last_block
        # Communication
        self.outer_product_mean_msa = OuterProductMean(
            c_m=self.c_m, c_z=self.c_z, c_hidden=self.c_hidden
        )
        if not self.is_last_block:
            # MSA stack
            self.msa_stack = MSAStack(
                c_m=self.c_m,
                c_z=self.c_z,
                dropout=msa_dropout,
                msa_chunk_size=msa_chunk_size,
                msa_max_size=msa_max_size,
            )
        # Pair stack
        self.pair_stack = PairformerBlock(
            c_z=c_z, c_s=0, dropout=pair_dropout, hidden_scale_up=hidden_scale_up
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        pair_mask,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        msa_chunk_size: Optional[int] = None,
        pair_transition_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: updated m z of MSABlock
                [...,n_msa_sampled, n_token, c_m]
                [...,n_token, n_token, c_z]
        """
        # Communication
        z = z + self.outer_product_mean_msa(
            m, inplace_safe=inplace_safe, chunk_size=chunk_size
        )
        if not self.is_last_block:
            # MSA stack
            m = self.msa_stack(m, z, msa_chunk_size=msa_chunk_size)
        # Pair stack
        _, z = self.pair_stack(
            s=None,
            z=z,
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            pair_transition_chunk_size=pair_transition_chunk_size,
        )
        if not self.is_last_block:
            return m, z
        else:
            return None, z  # to ensure that `m` will not be used.


class MSAModule(nn.Module):
    """
    Implements Algorithm 8 [MSAModule] in AF3

    Args:
        n_blocks (int, optional): number of blocks [for MSAModule]. Defaults to 4.
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional):
            hidden dim for single embedding from InputFeatureEmbedder. Defaults to 449.
        msa_dropout (float, optional): dropout ratio for msa block. Defaults to 0.15.
        pair_dropout (float, optional): dropout ratio for pair stack. Defaults to 0.25.
        blocks_per_ckpt: number of MSAModule blocks in each activation checkpoint. Defaults to 1.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): max size for msa. Defaults to 16384.
        msa_configs (dict, optional): a dictionary containing keys: "enable", "strategy", etc. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_m: int = 64,
        c_z: int = 128,
        c_s_inputs: int = 449,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = 1,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
        msa_configs: Optional[dict[str, Any]] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(MSAModule, self).__init__()
        self.n_blocks = n_blocks
        self.c_m = c_m
        self.c_s_inputs = c_s_inputs
        self.blocks_per_ckpt = blocks_per_ckpt
        self.msa_chunk_size = msa_chunk_size
        self.msa_max_size = msa_max_size
        self.input_feature = {
            "msa": 32,
            "has_deletion": 1,
            "deletion_value": 1,
        }

        self.msa_configs = {
            "enable": msa_configs.get("enable", False),
            "strategy": msa_configs.get("strategy", "random"),
        }
        if "sample_cutoff" in msa_configs:
            self.msa_configs["train_cutoff"] = msa_configs["sample_cutoff"].get(
                "train", 512
            )
            self.msa_configs["test_cutoff"] = msa_configs["sample_cutoff"].get(
                "test", 16384
            )
            # the default msa_max_size is 16384 if not specified
            self.msa_max_size = self.msa_configs["train_cutoff"]
        if "min_size" in msa_configs:
            self.msa_configs["train_lowerb"] = msa_configs["min_size"].get("train", 1)
            self.msa_configs["test_lowerb"] = msa_configs["min_size"].get("test", 1)

        self.linear_no_bias_m = LinearNoBias(
            in_features=32 + 1 + 1, out_features=self.c_m
        )

        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_m
        )
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            block = MSABlock(
                c_m=self.c_m,
                c_z=c_z,
                is_last_block=(i + 1 == n_blocks),
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                msa_chunk_size=self.msa_chunk_size,
                msa_max_size=self.msa_max_size,
                hidden_scale_up=hidden_scale_up,
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        msa_chunk_size: Optional[int] = None,
        pair_transition_chunk_size: Optional[int] = None,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                msa_chunk_size=msa_chunk_size,
                pair_transition_chunk_size=pair_transition_chunk_size,
            )
            for b in self.blocks
        ]
        return blocks

    def one_hot_fp32(
        self, tensor: torch.Tensor, num_classes: int, dtype=torch.float32
    ) -> torch.Tensor:
        """like F.one_hot, but output dtype is float32.

        Args:
            tensor (torch.Tensor): the input tensor
            num_classes (int): num_classes
            dtype (torch.float32, optional): the output dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: the one-hot encoded tensor with shape
                [..., n_msa_sampled, N_token, num_classes]
        """
        shape = tensor.shape
        one_hot_tensor = torch.zeros(
            *shape, num_classes, dtype=dtype, device=tensor.device
        )
        one_hot_tensor.scatter_(len(shape), tensor.unsqueeze(-1), 1)
        return one_hot_tensor

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        msa_chunk_size: Optional[int] = None,
        pair_transition_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Any]):
                input meta feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_token, c_s_inputs]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated z
                [..., N_token, N_token, c_z]
        """
        # If n_blocks < 1, return z
        if self.n_blocks < 1:
            return z

        if "msa" not in input_feature_dict:
            return z
        # Check msa shape!
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
        if input_feature_dict["msa"].dim() < 2:
            return z
        msa_feat = sample_msa_feature_dict_random_without_replacement(
            feat_dict=input_feature_dict,
            dim_dict={feat_name: -2 for feat_name in self.input_feature},
            cutoff=(
                self.msa_configs["train_cutoff"]
                if self.training
                else self.msa_configs["test_cutoff"]
            ),
            lower_bound=(
                self.msa_configs["train_lowerb"]
                if self.training
                else self.msa_configs["test_lowerb"]
            ),
            strategy=self.msa_configs["strategy"],
        )
        # pylint: disable=E1102
        if not self.training and z.shape[-2] > 2000:
            # msa_feat["msa"] is torch.int64, we convert it
            # to torch.float32 for saving half of the CUDA memory
            msa_feat["msa"] = self.one_hot_fp32(
                msa_feat["msa"],
                num_classes=self.input_feature["msa"],
            )
        else:
            msa_feat["msa"] = torch.nn.functional.one_hot(
                msa_feat["msa"],
                num_classes=self.input_feature["msa"],
            )

        target_shape = msa_feat["msa"].shape[:-1]
        msa_sample = torch.cat(
            [
                msa_feat[name].reshape(*target_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )  # [..., N_msa_sample, N_token, 32 + 1 + 1]
        # Msa_feat is very large, if N_MSA=16384 and N_token=4000,
        # msa_feat["msa"] consumes about 16G CUDA memory, so we
        # need to clear cache to avoid OOM
        if not self.training:
            del msa_feat
        # Line2
        msa_sample = self.linear_no_bias_m(msa_sample)

        # Auto broadcast [...,n_msa_sampled, n_token, c_m]
        msa_sample = msa_sample + self.linear_no_bias_s(s_inputs)
        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            msa_chunk_size=msa_chunk_size,
            pair_transition_chunk_size=pair_transition_chunk_size,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        msa_sample, z = checkpoint_blocks(
            blocks,
            args=(msa_sample, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return z


class TemplateEmbedder(nn.Module):
    """
    Implements Algorithm 16 in AF3

    Args:
        n_blocks (int, optional): number of blocks for TemplateEmbedder. Defaults to 2.
        c (int, optional): hidden dim of TemplateEmbedder. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        num_intermediate_factor (int, optional): number of intermediate factor for transition. Defaults to 2.
        dropout (float, optional): dropout ratio for PairformerStack. Defaults to 0.25.
            Note this value is missed in Algorithm 16, so we use default ratio for Pairformer
        blocks_per_ckpt (int, optional): number of TemplateEmbedder/Pairformer blocks in each activation
            checkpoint. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        num_intermediate_factor: int = 2,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(TemplateEmbedder, self).__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.input_feature1 = {
            "template_distogram": 39,
            "template_backbone_frame_mask": 1,
            "template_unit_vector": 3,
            "template_pseudo_beta_mask": 1,
        }
        self.input_feature2 = {
            "template_restype_i": 32,
            "template_restype_j": 32,
        }
        self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
        self.inf = 100000.0

        self.linear_no_bias_z = LinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_a = LinearNoBias(
            in_features=sum(self.input_feature1.values())
            + sum(self.input_feature2.values()),
            out_features=self.c,
        )
        self.pairformer_stack = PairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=self.n_blocks,
            num_intermediate_factor=num_intermediate_factor,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
            hidden_scale_up=hidden_scale_up,
        )
        self.layernorm_v = LayerNorm(self.c)
        self.relu = nn.ReLU()
        self.linear_no_bias_u = LinearNoBias(in_features=self.c, out_features=self.c_z)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: torch.Tensor = None,
        triangle_attention: str = "torch",
        triangle_multiplicative: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Any]): input feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor, optional): pair masking. Default to None.
                [..., N_token, N_token]
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module

        Returns:
            torch.Tensor: the template feature
                [..., N_token, N_token, c_z]
        """
        # Do not use TemplateEmbedder by setting n_blocks=0
        if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
            # Compatible with the Mol-1 0.5.0 model series
            return 0
        asym_id = input_feature_dict["asym_id"]
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(z.dtype)

        num_residues = z.shape[0]
        # determine whether the number of templates is the configured maximum value, otherwise error out
        num_templates = input_feature_dict["template_aatype"].shape[0]
        query_num_channels = z.shape[-1]

        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        z = self.layernorm_z(z)
        u = 0
        for template_id in range(num_templates):
            u = u + self.single_template_forward(
                template_id=template_id,
                input_feature_dict=input_feature_dict,
                z=z,
                pair_mask=pair_mask,
                multichain_mask=multichain_mask,
                triangle_attention=triangle_attention,
                triangle_multiplicative=triangle_multiplicative,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        u = u / (1e-7 + num_templates)
        u = self.linear_no_bias_u(self.relu(u))
        assert u.shape == (num_residues, num_residues, query_num_channels)
        return u

    def single_template_forward(
        self,
        template_id: int,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        multichain_mask: Optional[torch.Tensor] = None,
        triangle_attention: str = "torch",
        triangle_multiplicative: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        to_concat = []

        dgram = input_feature_dict["template_distogram"][
            template_id
        ]  # [N_token, N_token, 39]
        pseudo_beta_mask_2d = input_feature_dict["template_pseudo_beta_mask"][
            template_id
        ]
        dgram = dgram * multichain_mask[..., None] * pair_mask[..., None]
        pseudo_beta_mask_2d = (
            pseudo_beta_mask_2d * multichain_mask * pair_mask
        )  # [N_token, N_token]
        to_concat.append(dgram)
        to_concat.append(pseudo_beta_mask_2d.unsqueeze(-1))

        aatype = input_feature_dict["template_aatype"][template_id]  # [N_token]
        aatype = F.one_hot(aatype, num_classes=len(STD_RESIDUES_WITH_GAP))
        to_concat.append(expand_at_dim(aatype, dim=-3, n=z.shape[0]))
        to_concat.append(expand_at_dim(aatype, dim=-2, n=z.shape[0]))

        unit_vector = input_feature_dict["template_unit_vector"][template_id]
        unit_vector = (
            unit_vector * multichain_mask[..., None] * pair_mask[..., None]
        )  # [N_token, N_token, 3]
        to_concat.append(unit_vector)

        backbone_mask_2d = input_feature_dict["template_backbone_frame_mask"][
            template_id
        ]
        backbone_mask_2d = backbone_mask_2d * multichain_mask * pair_mask
        to_concat.append(backbone_mask_2d.unsqueeze(-1))

        at = torch.concat(to_concat, dim=-1)
        v = self.linear_no_bias_z(z) + self.linear_no_bias_a(at)
        _, v = self.pairformer_stack(
            s=None,
            z=v,
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        v = self.layernorm_v(v)
        return v
