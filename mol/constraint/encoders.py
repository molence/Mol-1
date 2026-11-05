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
"""SITE/EXCLUSION and sparse CONTACT pair encoders."""
from __future__ import annotations

from torch import nn
import math
import torch
import mol.constraint.schema as _schema


class ContactMaskEncoder(nn.Module):
    def __init__(self, c_z):
        super().__init__()
        self.contact_embedding = nn.Parameter(torch.empty((c_z,)))
        nn.init.normal_(self.contact_embedding, std=c_z ** -.5)
        self.output_projection = nn.Linear(c_z, c_z, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, condition):
        condition.validate()
        indices = condition.pair_indices.to(self.contact_embedding.device)
        mask = condition.pair_mask.to(indices.device)
        batch_shape, capacity = mask.shape[:-1], mask.shape[-1]
        batches, n = math.prod(batch_shape), condition.num_tokens
        indices = indices.reshape(batches, capacity, 2)
        mask = mask.reshape(batches, capacity)
        value = self.output_projection(self.contact_embedding)
        values = mask[..., None].to(value.dtype) * value
        c_z = self.output_projection.out_features
        output = value.new_zeros(batches * n * n, c_z)
        # Empty index_add can fail on CUDA when the output needs 64-bit indexing.
        if capacity > 0:
            safe = torch.where(mask[..., None], indices, 0)
            offset = torch.arange(batches, device=indices.device)[:, None] * n * n
            for i, j in ((0, 1), (1, 0)):
                output = output.index_add(0, (offset + safe[..., i] * n + safe[..., j]).reshape(-1),
                                          values.reshape(-1, c_z))
        # Preserve an autograd path even for a zero-capacity null input.
        return output.reshape(*batch_shape, n, n, c_z) + value.sum() * 0


class ResidueSiteConstraintEncoder(nn.Module):

    def __init__(
        self,
        c_z: int = 128,
        *,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if c_z <= 0:
            raise ValueError("c_z must be positive")
        self.c_z = c_z
        # The two embeddings describe z matrix orientation only. They do not
        # encode generic-P-P versus antibody-antigen class information.
        self.pair_orientation_embedding = nn.Parameter(torch.empty(2, c_z))
        nn.init.normal_(
            self.pair_orientation_embedding, mean=0.0, std=c_z**-0.5
        )
        self.output_projection = nn.Linear(c_z, c_z, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, constraints: _schema.TokenSiteConstraintBatch) -> torch.Tensor:
        constraints.validate()
        embedding = self.pair_orientation_embedding
        dtype = embedding.dtype
        device = embedding.device
        active = constraints.constraint_mask.to(
            device=device, dtype=dtype
        ).unsqueeze(-1).unsqueeze(-1)
        partner = constraints.partner_token_mask.to(device=device, dtype=dtype)
        conditioned = constraints.conditioned_residue_mask.to(
            device=device, dtype=dtype
        )
        partner_to_site = (
            partner.unsqueeze(-1) * conditioned.unsqueeze(-2) * active
        )
        site_to_partner = (
            conditioned.unsqueeze(-1) * partner.unsqueeze(-2) * active
        )
        update = (
            partner_to_site.unsqueeze(-1) * embedding[0]
            + site_to_partner.unsqueeze(-1) * embedding[1]
        )
        return self.output_projection(update)
