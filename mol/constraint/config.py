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
"""Configuration for constraint inference."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from mol.config.types import DefaultNoneWithType, ListValue
from pathlib import Path
import argparse
import json
import math
import os
import mol.constraint.runtime as _runtime
import mol.constraint.schema as _schema


def load_inference_config(path, native_args):
    """Use the native defaults/model registry/typed argument merger."""
    from mol.config.base import configs as base
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.parser import ConfigManager

    model_configs = constraint_model_configs()
    overrides = load_config_overrides(path)
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--model_name", default=overrides.get("model_name", MODEL_NAME))
    model_name = selection.parse_known_args(native_args)[0].model_name
    if model_name not in model_configs:
        raise ValueError(f"Unsupported constraint model {model_name!r}; expected {MODEL_NAME!r}")
    defaults = deepcopy({**base, "data": data_configs, **inference_configs})
    deep_update(defaults, model_configs[model_name])
    deep_update(defaults, DEFAULT_CONFIG)
    deep_update(defaults, overrides)
    defaults["model_name"] = model_name
    manager = ConfigManager(defaults, fill_required_with_null=True)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Use --load_checkpoint_path FILE to load the supplied model weights.")
    for key in manager.config_infos:
        parser.add_argument("--" + key, default=argparse.SUPPRESS)
    config = manager.merge_configs(vars(parser.parse_args(native_args)))
    _runtime.verify_runtime_config(config, _runtime.runtime_snapshot(config), phase="inference_config")
    return config


MODEL_NAME = "mol-1-constraint"


# Single source of shipped defaults; callers receive independent copies.
DEFAULT_CONFIG = {'model_name': 'mol-1-constraint',
 'model': {'N_cycle': 20},
 'constraint_exclusion_enable': True,
 'constraint_exclusion_d_gap': 1.5,
 'constraint_contact_enable': True,
 'inference_runtime_checks': True,
 'dtype': 'bf16',
 'mc_dropout_apply_rate': 0.0,
 'mc_dropout_rate': 0.0,
 'enable_tf32': False,
 'sample_diffusion': {'N_step': 200},
 'infer_setting': {'sample_diffusion_chunk_size': 1}}


def constraint_model_configs():
    """Derive the constraint architecture from the public Mol-1 base preset."""
    from copy import deepcopy
    from mol.config.model_type import model_configs

    return {MODEL_NAME: deepcopy(model_configs["mol-1-base"])}


@dataclass(frozen=True)
class ExclusionSettings:
    d_gap: float = DEFAULT_CONFIG["constraint_exclusion_d_gap"]

    @property
    def cutoff(self):
        return _schema.CONTACT_DISTANCE_ANGSTROM + self.d_gap

    @classmethod
    def from_config(cls, config):
        values = {name: config.get("constraint_exclusion_" + name, default)
                  for name, default in asdict(cls()).items()}
        return cls(**values).validate()

    def validate(self):
        if not math.isfinite(self.d_gap) or self.d_gap <= 0:
            raise ValueError("EXCLUSION d_gap must be finite and positive")
        return self

    def contract(self):
        return {"distance": "minimum-heavy-atom", "positive_cutoff": _schema.CONTACT_DISTANCE_ANGSTROM,
                "exclusion_cutoff": self.cutoff, "d_gap": self.d_gap, "scope": "protein_source",
                "violation": "any_distance_le_cutoff"}


def deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict):
            if not isinstance(target.get(key), dict):
                target[key] = {}
            deep_update(target[key], value)
        elif value is None:
            target[key] = DefaultNoneWithType(getattr(target.get(key), "dtype", str))
        elif value == []:
            empty = ListValue(None, dtype=getattr(target.get(key), "dtype", str))
            empty.value = []
            target[key] = empty
        else:
            target[key] = os.path.expandvars(value) if isinstance(value, str) else deepcopy(value)
    return target


def _merge_json_dicts(parent, child):
    """Merge a child JSON config over its parent without changing either input."""
    merged = deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_json_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config_overrides(path=None, *, _chain=()):
    """Return Python defaults, or load optional JSON overrides."""
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    path = Path(path).resolve()
    if path in _chain:
        chain = " -> ".join(str(item) for item in (*_chain, path))
        raise ValueError(f"config inheritance cycle: {chain}")
    with path.open() as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    parents = overrides.pop("extends", None)
    if parents is None:
        return overrides
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not parents:
        raise ValueError(f"extends must be a non-empty relative path or list: {path}")
    merged = {}
    for parent in parents:
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"extends must contain non-empty relative paths: {path}")
        parent_path = Path(parent)
        if parent_path.is_absolute():
            raise ValueError(f"extends must be a relative path: {path}")
        merged = _merge_json_dicts(
            merged,
            load_config_overrides(path.parent / parent_path, _chain=(*_chain, path)),
        )
    return _merge_json_dicts(merged, overrides)
