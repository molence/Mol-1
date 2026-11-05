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
"""ConstraintMol adapter over the unchanged base Mol implementation."""
from __future__ import annotations

from mol.model.mol import Mol
import torch
import mol.constraint.config as _config
import mol.constraint.encoders as _encoders
import mol.constraint.features as _features
import mol.constraint.runtime as _runtime
import mol.constraint.schema as _schema


class ConstraintMol(Mol):
    def __init__(self, configs):
        super().__init__(configs)
        self.inference_max_cycle = _runtime.resolve_n_cycle(configs)
        self._inference_runtime_expected = (_runtime.runtime_snapshot(configs)
                                            if configs.get("inference_runtime_checks", False) else None)
        self.constraint_conditioner = _encoders.ResidueSiteConstraintEncoder(c_z=self.c_z)
        self.exclusion_enabled = bool(configs.get("constraint_exclusion_enable", False))
        if self.exclusion_enabled:
            # Keep positive keys and RNG unchanged; use the same encoder implementation.
            with torch.random.fork_rng(devices=[]):
                self.constraint_conditioner.exclusion = _encoders.ResidueSiteConstraintEncoder(c_z=self.c_z)
        self.contact_enabled = bool(configs.get('constraint_contact_enable', False))
        if self.contact_enabled:
            with torch.random.fork_rng(devices=[]):
                self.constraint_conditioner.contact = _encoders.ContactMaskEncoder(
                    self.c_z)
        chunk_rows = int(configs.get("constraint_inference_transition_chunk_rows", 0))
        if chunk_rows:
            _runtime.configure_inference_transition_chunking(
                self,
                min_tokens=int(configs.get("constraint_inference_transition_min_tokens", 2000)),
                chunk_rows=chunk_rows,
            )
        numerical_config = configs.get("constraint_numerics", {})
        if numerical_config.get("reference_kernels", False):
            _runtime.reference_layer_norms(self)
        self._finite_hooks = _runtime.install_finite_hooks(
            self, detailed=numerical_config.get("detailed", False))

    def forward(self, input_feature_dict, label_full_dict, label_dict, *args, **kwargs):
        _runtime.require_finite(input_feature_dict, "input_feature_dict")
        _runtime.require_finite(label_full_dict, "label_full_dict")
        _runtime.require_finite(label_dict, "label_dict")
        mode = kwargs.get("mode", args[0] if args else "inference")
        if mode == "train":
            raise ValueError("ConstraintMol supports inference only")
        self.N_cycle = self.inference_max_cycle
        # The native forward signature defaults to 0.4; use the request value.
        rate = kwargs.setdefault("mc_dropout_apply_rate", self.configs.mc_dropout_apply_rate)
        expected = self._inference_runtime_expected
        _runtime.verify_runtime_config(
            self.configs, expected, phase="constraint_model_forward",
            execution_plan=kwargs.get("execution_plan"), mc_dropout_apply_rate=rate,
            cuda_execution=bool(expected) and input_feature_dict["token_index"].is_cuda, model=self,
        )
        return super().forward(input_feature_dict, label_full_dict, label_dict, *args, **kwargs)

    def get_pairformer_output(self, *args, **kwargs):
        output = super().get_pairformer_output(*args, **kwargs)
        _runtime.require_finite(output, "pairformer_output")
        return output

    def get_constraint_update(self, input_feature_dict):
        # Require the current constraint input schema.
        if "constraint_feature" in input_feature_dict:
            raise ValueError("legacy constraint_feature cannot accompany the opt-in constraint input")
        batch = _schema.TokenSiteConstraintBatch.from_feature_dict(input_feature_dict)
        update = self.constraint_conditioner(batch)
        if getattr(self, "exclusion_enabled", False):
            update = update + self.constraint_conditioner.exclusion(
                _features.exclusion_view(input_feature_dict)).to(update)
        elif any(key.startswith("exclusion_") for key in input_feature_dict):
            raise ValueError("EXCLUSION inputs require constraint_exclusion_enable and a matching checkpoint")
        if getattr(self, 'contact_enabled', False):
            condition = _schema.ContactMask.from_features(
                input_feature_dict, exclusion_cutoff=_config.ExclusionSettings.from_config(self.configs).cutoff)
            update = update + self.constraint_conditioner.contact(condition).to(update)
        elif any(key.startswith('contact_') for key in input_feature_dict):
            raise ValueError('CONTACT inputs require enabled CONTACT')
        return update
