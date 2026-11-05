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
"""Constraint inference weight and metadata compatibility."""
from __future__ import annotations

import mol.constraint.config as _config
import mol.constraint.runtime as _runtime
import mol.constraint.schema as _schema


def contract_without_version(contract):
    """Compare actual semantics; tolerate extra metadata in existing exports."""
    if not isinstance(contract, dict):
        return contract
    normalized = {key: value for key, value in contract.items() if key != "version"}
    # Released checkpoints used this name for the same protein-source geometry.
    if normalized.get("scope") == "unified_protein_source":
        normalized["scope"] = "protein_source"
    return normalized


def contact_checkpoint_state(checkpoint, config):
    contact_validate_checkpoint(checkpoint, config)
    if config.get('constraint_exclusion_enable', False):
        return exclusion_checkpoint_state(checkpoint, config)
    return checkpoint_state(checkpoint, config)


def exclusion_checkpoint_state(checkpoint, config):
    validate_exclusion_checkpoint(checkpoint, _config.ExclusionSettings.from_config(config))
    checkpoint_name = checkpoint.get("model_name", config.model_name)
    if checkpoint_name != config.model_name:
        raise ValueError(f"checkpoint was saved for model {checkpoint_name!r}; "
                         f"the config selects {config.model_name!r}")
    return checkpoint["model"]


def load_model_state(model, state):
    """Validate all supplied tensors before strictly loading inference weights."""
    state = {k.removeprefix("module."): v for k, v in state.items()}
    _runtime.require_finite(state, "checkpoint.model")
    if not any(k.startswith("constraint_conditioner.") for k in state):
        raise RuntimeError("Base checkpoint has no constraint_conditioner keys: "
                           "constraint inference requires mol-1-constraint weights")
    # PyTorch can copy valid tensors before reporting a later mismatch.
    # Preflight the entire state so failed loads leave weights unchanged.
    expected = model.state_dict()
    missing, unexpected = expected.keys() - state.keys(), state.keys() - expected.keys()
    if missing or unexpected:
        raise RuntimeError("checkpoint tensor keys mismatch: "
                           f"missing={sorted(missing)}, unexpected={sorted(unexpected)}; "
                           "use the matching model architecture and conditioner modules")
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise RuntimeError(
                f"checkpoint shape mismatch: {key}: checkpoint {tuple(value.shape)}, "
                f"model {tuple(expected[key].shape)}; use a matching architecture "
                "(including c_z and hidden_scale_up)")
    print("Checkpoint mode: strict constraint checkpoint (all model tensors finite)", flush=True)
    return model.load_state_dict(state, strict=True)


def checkpoint_state(checkpoint, config):
    """Validate compatibility and load the supplied export without switching weights."""

    checkpoint_name = checkpoint.get("model_name", config.model_name)
    if checkpoint_name != config.model_name:
        raise ValueError(f"checkpoint was saved for model {checkpoint_name!r}; "
                         f"the config selects {config.model_name!r}")
    return checkpoint["model"]


def contact_validate_checkpoint(checkpoint, config):
    enabled = config.get('constraint_contact_enable', False)
    saved = checkpoint.get('contact')
    has_weights = any(k.removeprefix('module.').startswith(_schema.CONTACT_PARAM_PREFIX)
                      for k in checkpoint.get('model', {}))
    if not enabled:
        if saved is not None or has_weights:
            raise ValueError('CONTACT checkpoint requires enabled CONTACT')
        return
    if saved is None:
        raise ValueError('CONTACT checkpoint metadata missing')
    if not has_weights:
        raise ValueError('CONTACT checkpoint geometry/weights contract mismatch')
    if contract_without_version(saved.get('contract')) != dict(
            cutoff_angstrom=_schema.CONTACT_DISTANCE_ANGSTROM, geometry='minimum_heavy_atom',
            scope='token_pairs', positive_only=True, zero='unknown'):
        raise ValueError('CONTACT checkpoint geometry/weights contract mismatch')


def validate_exclusion_checkpoint(checkpoint, settings):
    saved = checkpoint.get("exclusion")
    if not isinstance(saved, dict) or contract_without_version(saved.get("contract")) != settings.contract():
        raise ValueError("EXCLUSION checkpoint distance/primitive contract mismatch")
