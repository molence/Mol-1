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
"""Project residue and token identities into constraint model features."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import mol.constraint.schema as _schema


@dataclass(frozen=True)
class _TokenContext:
    n_tokens: int
    protein_mask: torch.Tensor
    chain_ids: np.ndarray
    seq_ids: list[int | None]
    observed_residue_ids: set[tuple[str, int]]
    native_res_ids: list[int | None] | None = None


def attach_site_constraint_features(
    input_feature_dict: Mapping[str, Any],
    constraints: _schema.TokenSiteConstraintBatch,
) -> dict[str, Any]:
    """Attach shared token metadata and SITE tensors consumed by the model."""

    output = dict(input_feature_dict)
    overlap = set(_schema.SITE_FEATURE_KEYS.values()) & set(
        output
    )
    if overlap:
        raise ValueError(f"feature dict already contains SITE/shared token keys: {overlap}")
    for key, value in constraints.as_dict().items():
        output[_schema.SITE_FEATURE_KEYS[key]] = value
    token_index = output.get("token_index")
    if token_index is not None and token_index.shape[0] != constraints.n_tokens:
        raise ValueError("constraint tensors must share the existing token axis")
    return output


def contact_project(features, atoms, pairs):
    mapping = features['atom_to_token_idx'].cpu().numpy().astype(int)
    protein = features['protein_token_mask']
    chains, positions = np.asarray(atoms.chain_id).astype(str), np.asarray(atoms.label_seq_id).astype(str)
    projected = []
    for pair in pairs:
        if all(type(i) is int for i in pair):
            projected.append(pair)
            continue
        indices = []
        for chain, position in pair:
            tokens = np.unique(mapping[(chains == chain) & (positions == str(position))])
            if len(tokens) != 1 or not bool(protein[int(tokens[0])]):
                raise ValueError(f'CONTACT residue missing, ambiguous, or nonprotein: {(chain, position)}')
            indices.append(int(tokens[0]))
        projected.append(indices)
    condition = _schema.ContactMask.from_pairs(projected, len(protein), device=protein.device)
    result = dict(features, **condition.as_features())
    _schema.ContactMask.from_features(result)
    audit = {'active': bool(condition.pair_mask.any()), 'requested_pairs': len(pairs),
                    'selected_pairs': int(condition.pair_mask.sum()), 'input_pairs': pairs,
                    'residue_pairs': [p for p in pairs if not all(type(i) is int for i in p)],
                    'token_pairs': condition.pair_indices[condition.pair_mask].cpu().tolist()}
    return result, audit


def exclusion_project(features, atoms, context, residues):

    # Reuse exact sequence numbering and projection, without rewriting positive masks.
    clean = {key: value for key, value in features.items()
             if key not in _schema.SITE_FEATURE_KEYS.values() and not key.startswith("exclusion_")}
    projected, audit = project_sites(clean, atoms, context, residues)
    combined = exclusion_attach(features, projected["site_residue_mask"],
                      projected["site_partner_token_mask"])
    return combined, audit


def project_sites(features, atoms, context, residues):
    """Project explicit sequence identities; fail if any requested site disappears."""
    import numpy as np
    import torch


    n_tokens = int(features["token_index"].shape[0])
    atom_to_token = features["atom_to_token_idx"].cpu().numpy().astype(int)
    protein = torch.zeros(n_tokens, dtype=torch.bool)
    protein[features["atom_to_token_idx"][features["is_protein"].bool()].long()] = True
    selected = torch.zeros_like(protein)
    projected = []
    if residues and context is None:
        raise ValueError("nonempty sites require source_chains and partner_chains context")
    for chain, position in residues:
        mask = (np.asarray(atoms.chain_id).astype(str) == chain) & (np.asarray(atoms.label_seq_id).astype(str) == str(position))
        tokens = np.unique(atom_to_token[mask])
        if not len(tokens) or not bool(protein[tokens].all()):
            raise ValueError(f"site does not map to protein tokens: {(chain, position)}")
        selected[tokens] = True
        projected.append({"residue_id": [chain, position], "token_indices": tokens.tolist()})
    partner = torch.zeros_like(protein)
    if context is not None:
        source_atoms = np.isin(atoms.chain_id, context["source_chains"])
        partner_atoms = np.isin(atoms.chain_id, context["partner_chains"])
        # Check every declared role chain survives native preprocessing.
        if not set(context["source_chains"] + context["partner_chains"]) <= set(atoms.chain_id):
            raise ValueError("a declared role chain disappeared during preprocessing")
        if not source_atoms.any() or not partner_atoms.any():
            raise ValueError("source/partner empty after preprocessing")
        partner[torch.as_tensor(np.unique(atom_to_token[partner_atoms]), dtype=torch.long)] = True
    batch = (_schema.TokenSiteConstraintBatch(torch.tensor(True), protein, partner, selected).validate()
             if residues else _schema.TokenSiteConstraintBatch.null(n_tokens, protein_token_mask=protein))
    # Older featurizers may supply an empty constraint_feature mapping.
    # Nonempty constraints must use the current input schema.
    legacy = features.pop("constraint_feature", {})
    if any(bool(torch.as_tensor(value).any()) for value in legacy.values()):
        raise ValueError("unexpected nonempty legacy constraint features")
    features = attach_site_constraint_features(features, batch)
    return features, {"target_id": context["target_id"] if context else None, "n_tokens": n_tokens,
                      "n_atoms": len(atoms), "requested_sites": len(residues),
                      "projected_sites": len(projected), "selected_tokens": int(selected.sum()), "projected": projected,
                      "active": bool(batch.constraint_mask),
                      "partner_tokens": int(batch.partner_token_mask.sum())}


def exclusion_attach(features, residues=None, partner=None):
    protein = features["protein_token_mask"]
    residues = torch.zeros_like(protein) if residues is None else residues
    partner = torch.zeros_like(protein) if partner is None else partner
    gate = residues.any(-1)
    partner = partner & gate.unsqueeze(-1)
    result = dict(features)
    result.update({_schema.EXCLUSION_PREFIX + "mask": gate,
                   _schema.EXCLUSION_PREFIX + "partner_token_mask": partner,
                   _schema.EXCLUSION_PREFIX + "residue_mask": residues})
    exclusion_view(result)
    return result


def exclusion_view(features):
    keys = {_schema.EXCLUSION_PREFIX + name for name in _schema.EXCLUSION_FIELDS}
    present = keys.intersection(features)
    protein = features["protein_token_mask"]
    if not present:
        return _schema.TokenSiteConstraintBatch.null(protein.shape[-1], batch_shape=tuple(protein.shape[:-1]),
                                               protein_token_mask=protein, device=protein.device)
    if present != keys:
        raise ValueError("incomplete EXCLUSION fixed-mask signature")
    batch = _schema.TokenSiteConstraintBatch(
        features[_schema.EXCLUSION_PREFIX + "mask"], protein,
        features[_schema.EXCLUSION_PREFIX + "partner_token_mask"], features[_schema.EXCLUSION_PREFIX + "residue_mask"]).validate()
    positive = features["site_residue_mask"]
    positive_partner = features["site_partner_token_mask"]
    shared_residues = (positive & batch.conditioned_residue_mask).any(dim=-1)
    shared_partners = (positive_partner & batch.partner_token_mask).any(dim=-1)
    both = features["site_mask"] & batch.constraint_mask
    if torch.any(both & shared_residues & shared_partners):
        raise ValueError("SITE and EXCLUSION conflict on shared residues and partners")
    return batch
