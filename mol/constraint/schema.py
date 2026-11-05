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
"""Tensor contracts and constants for SITE, CONTACT and EXCLUSION."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch


SITE_FEATURE_KEYS = {
    "constraint_mask": "site_mask",
    "protein_token_mask": "protein_token_mask",
    "partner_token_mask": "site_partner_token_mask",
    "conditioned_residue_mask": "site_residue_mask",
}

CONTACT_DISTANCE_ANGSTROM = 4.5


CONTACT_PREFIX = "contact_"


CONTACT_PARAM_PREFIX = "constraint_conditioner.contact."


@dataclass(frozen=True)
class ContactMask:
    pair_indices: torch.Tensor
    pair_mask: torch.Tensor
    num_tokens: int

    @classmethod
    def empty(cls, num_tokens, capacity=0, *, device=None):
        return cls(torch.zeros((capacity, 2), dtype=torch.long, device=device),
                   torch.zeros(capacity, dtype=torch.bool, device=device), num_tokens)

    @classmethod
    def from_pairs(cls, pairs, num_tokens, *, capacity=None, device=None):
        pairs = sorted({tuple(sorted((int(i), int(j)))) for i, j in pairs})
        capacity = len(pairs) if capacity is None else capacity
        if capacity < len(pairs):
            raise ValueError('CONTACT capacity smaller than requested pair count')
        result = cls.empty(num_tokens, capacity, device=device)
        if pairs:
            result.pair_indices[:len(pairs)] = torch.tensor(pairs, device=device)
            result.pair_mask[:len(pairs)] = True
        return result.validate()

    @property
    def cutoffs(self):
        """Per-pair contact cutoff, the same for every pair."""
        return torch.full(self.pair_mask.shape, CONTACT_DISTANCE_ANGSTROM, device=self.pair_mask.device)

    @classmethod
    def from_features(cls, features, *, exclusion_cutoff=None):
        tokens = features.get('token_index', features.get('protein_token_mask'))
        if tokens is None:
            raise ValueError('CONTACT requires a complete token axis')
        keys = (CONTACT_PREFIX + 'pair_indices', CONTACT_PREFIX + 'pair_mask')
        if not any(key in features for key in keys):
            if tokens.ndim != 1:
                raise ValueError('missing CONTACT signature on batched input')
            return cls.empty(tokens.shape[-1], device=tokens.device)
        if not all(key in features for key in keys):
            raise ValueError('incomplete CONTACT signature')
        result = cls(*(features[key] for key in keys), tokens.shape[-1]).validate()
        if tokens.shape[:-1] != result.pair_mask.shape[:-1]:
            raise ValueError('CONTACT batch shape disagrees with tokens')
        # Input-level contradiction detection uses the actual exclusion scope.
        ex = features.get('exclusion_residue_mask')
        partner = features.get('exclusion_partner_token_mask')
        if ex is not None and partner is not None and result.pair_mask.numel():
            bounds = result.cutoffs.reshape(-1, result.pair_mask.shape[-1])
            for indices, mask, a, b, pair_cutoffs in zip(result.pair_indices.reshape(-1, result.pair_mask.shape[-1], 2),
                                           result.pair_mask.reshape(-1, result.pair_mask.shape[-1]),
                                           ex.reshape(-1, tokens.shape[-1]), partner.reshape(-1, tokens.shape[-1]), bounds):
                i, j = indices[mask].unbind(-1)
                cutoff = CONTACT_DISTANCE_ANGSTROM if exclusion_cutoff is None else exclusion_cutoff
                conflict = ((a[i] & b[j]) | (a[j] & b[i])) & (pair_cutoffs[mask] <= cutoff)
                if bool(conflict.any()):
                    raise ValueError('CONTACT pair conflicts with EXCLUSION scope')
        return result

    def validate(self):
        indices, mask = self.pair_indices, self.pair_mask
        if indices.dtype != torch.long or mask.dtype != torch.bool:
            raise TypeError('CONTACT requires int64 indices and boolean mask')
        if mask.ndim < 1 or indices.shape != (*mask.shape, 2) or indices.device != mask.device:
            raise ValueError('CONTACT requires [..., K, 2] indices and [..., K] mask on one device')
        active = indices[mask]
        if active.numel() and bool(((active < 0) | (active >= self.num_tokens)).any()):
            raise ValueError('CONTACT index outside token range')
        if active.numel() and bool((active[:, 0] >= active[:, 1]).any()):
            raise ValueError('CONTACT pairs must have i < j')
        if mask.shape[-1]:
            for rows, valid in zip(indices.reshape(-1, mask.shape[-1], 2), mask.reshape(-1, mask.shape[-1])):
                selected = rows[valid]
                if selected.unique(dim=0).shape[0] != selected.shape[0]:
                    raise ValueError('duplicate CONTACT pair')
        return self

    def as_features(self):
        return {CONTACT_PREFIX + 'pair_indices': self.pair_indices, CONTACT_PREFIX + 'pair_mask': self.pair_mask}


EXCLUSION_PREFIX = "exclusion_"


EXCLUSION_PARAM_PREFIX = "constraint_conditioner.exclusion."


EXCLUSION_FIELDS = ("mask", "partner_token_mask", "residue_mask")


@dataclass(frozen=True)
class TokenSiteConstraintBatch:
    """Protein SITE input with partner tokens of any molecular composition.

    ``conditioned_residue_mask`` contains only the observed subset of native
    interface residues on one source chain. ``partner_token_mask`` contains all
    visible tokens of the partner chain; it is not a native-interface label.
    No biological complex class or antibody/antigen direction is represented.
    """

    constraint_mask: torch.Tensor
    protein_token_mask: torch.Tensor
    partner_token_mask: torch.Tensor
    conditioned_residue_mask: torch.Tensor

    @classmethod
    def null(
        cls,
        n_tokens: int,
        *,
        batch_shape: tuple[int, ...] = (),
        protein_token_mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> "TokenSiteConstraintBatch":
        if n_tokens < 0:
            raise ValueError(f"n_tokens must be non-negative, got {n_tokens}")
        token_shape = (*batch_shape, n_tokens)
        if protein_token_mask is None:
            protein_token_mask = torch.ones(
                token_shape, dtype=torch.bool, device=device
            )
        elif protein_token_mask.shape != token_shape:
            raise ValueError("protein_token_mask shape must be [..., N]")
        elif device is not None:
            protein_token_mask = protein_token_mask.to(device=device)
        device = protein_token_mask.device
        return cls(
            constraint_mask=torch.zeros(
                batch_shape, dtype=torch.bool, device=device
            ),
            protein_token_mask=protein_token_mask,
            partner_token_mask=torch.zeros(
                token_shape, dtype=torch.bool, device=device
            ),
            conditioned_residue_mask=torch.zeros(
                token_shape, dtype=torch.bool, device=device
            ),
        ).validate()

    @classmethod
    def from_feature_dict(
        cls,
        features: dict[str, Any],
    ) -> "TokenSiteConstraintBatch":
        names = (
            "constraint_mask",
            "protein_token_mask",
            "partner_token_mask",
            "conditioned_residue_mask",
        )
        missing = [name for name in names if SITE_FEATURE_KEYS[name] not in features]
        if missing:
            raise KeyError(f"missing constraint model-input features: {missing}")
        return cls(
            constraint_mask=features[SITE_FEATURE_KEYS["constraint_mask"]],
            protein_token_mask=features[SITE_FEATURE_KEYS["protein_token_mask"]],
            partner_token_mask=features[SITE_FEATURE_KEYS["partner_token_mask"]],
            conditioned_residue_mask=features[
                SITE_FEATURE_KEYS["conditioned_residue_mask"]
            ],
        ).validate()

    @property
    def n_tokens(self) -> int:
        return self.conditioned_residue_mask.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        return self.conditioned_residue_mask.shape[:-1]

    def validate(self) -> "TokenSiteConstraintBatch":
        tensors = (
            self.constraint_mask,
            self.protein_token_mask,
            self.partner_token_mask,
            self.conditioned_residue_mask,
        )
        names = (
            "constraint_mask",
            "protein_token_mask",
            "partner_token_mask",
            "conditioned_residue_mask",
        )
        for name, tensor in zip(names, tensors):
            if tensor.dtype is not torch.bool:
                raise TypeError(f"{name} must have dtype torch.bool")
        if any(tensor.device != tensors[0].device for tensor in tensors[1:]):
            raise ValueError("all constraint tensors must be on the same device")

        token_shape = self.conditioned_residue_mask.shape
        if self.conditioned_residue_mask.ndim < 1:
            raise ValueError("constraint masks must have at least one dimension")
        if self.protein_token_mask.shape != token_shape:
            raise ValueError("protein_token_mask must have shape [..., N]")
        if self.partner_token_mask.shape != token_shape:
            raise ValueError("partner_token_mask must have shape [..., N]")
        if self.constraint_mask.shape != token_shape[:-1]:
            raise ValueError("constraint_mask must match leading batch dimensions")

        if torch.any(self.conditioned_residue_mask & ~self.protein_token_mask):
            raise ValueError("conditioned residues must be protein tokens")
        if torch.any(self.partner_token_mask & self.conditioned_residue_mask):
            raise ValueError("partner and conditioned residue masks must be disjoint")

        has_partner = self.partner_token_mask.any(dim=-1)
        has_conditioned = self.conditioned_residue_mask.any(dim=-1)
        if torch.any(self.constraint_mask & (~has_partner | ~has_conditioned)):
            raise ValueError(
                "active constraints require partner and conditioned residues"
            )
        inactive = ~self.constraint_mask
        if torch.any((has_partner | has_conditioned) & inactive):
            raise ValueError("inactive constraints must have zero role masks")
        return self

    def to(self, device: torch.device | str) -> "TokenSiteConstraintBatch":
        return TokenSiteConstraintBatch(
            constraint_mask=self.constraint_mask.to(device=device),
            protein_token_mask=self.protein_token_mask.to(device=device),
            partner_token_mask=self.partner_token_mask.to(device=device),
            conditioned_residue_mask=self.conditioned_residue_mask.to(
                device=device
            ),
        ).validate()

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "constraint_mask": self.constraint_mask,
            "protein_token_mask": self.protein_token_mask,
            "partner_token_mask": self.partner_token_mask,
            "conditioned_residue_mask": self.conditioned_residue_mask,
        }
