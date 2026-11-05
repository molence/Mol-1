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
"""Run explicit JSON constraints through the native inference pipeline."""
from __future__ import annotations

from mol.data.inference.infer_dataloader import InferenceDataset
from mol.runner.inference import (
    InferenceRunner,
    download_inference_cache,
    infer_predict,
    resolve_checkpoint,
    update_gpu_compatible_configs,
)
from mol.utils.distributed import DIST_WRAPPER
from mol.utils.torch_utils import collate_fn_identity
from torch.utils.data import DataLoader
import argparse
import logging
import torch
import mol.constraint.checkpoint as _checkpoint
import mol.constraint.config as _config
import mol.constraint.features as _features
import mol.constraint.inputs as _inputs
import mol.constraint.model as _model
import mol.constraint.runtime as _runtime


class ConstraintInferenceDataset(InferenceDataset):
    """Add constraint tensors while retaining native featurization and errors."""

    def __init__(self, configs):
        super().__init__(configs)
        tasks, rows = _inputs.prepare_inputs(
            _inputs.read_json(configs.input_json_path),
            config=configs,
        )
        self.inputs = tasks
        self.conditions = {task['name']: row for task, row in zip(tasks, rows)}

    def process_one(self, single_sample_dict):
        data, atoms, timings = super().process_one(single_sample_dict)
        row = self.conditions[single_sample_dict['name']]
        site = row['site']
        features, _ = _features.project_sites(data['input_feature_dict'], atoms, site,
                                   site['residue_ids'] if site else [])
        if self.configs.constraint_exclusion_enable:
            condition = row['exclusion']
            features, _ = _features.exclusion_project(features, atoms, condition,
                                            condition['residue_ids'] if condition else [])
        if self.configs.constraint_contact_enable:
            features, _ = _features.contact_project(features, atoms, row['contact'] or [])
        _runtime.require_finite(features, 'constraint_input')
        data['input_feature_dict'] = features
        return data, atoms, timings


class ConstraintInferenceRunner(InferenceRunner):
    """Override constraint-specific model, checkpoint and data initialization."""

    def init_model(self):
        _config.ExclusionSettings.from_config(self.configs)
        self.model = _model.ConstraintMol(self.configs).to(self.device)

    def load_checkpoint(self):
        config = self.configs
        path = resolve_checkpoint(config)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if config.constraint_contact_enable:
            state = _checkpoint.contact_checkpoint_state(checkpoint, config)
        elif checkpoint.get('contact') is not None:
            raise ValueError('CONTACT checkpoint requires enabled CONTACT')
        elif config.constraint_exclusion_enable:
            state = _checkpoint.exclusion_checkpoint_state(checkpoint, config)
        else:
            state = _checkpoint.checkpoint_state(checkpoint, config)
        logging.info("Loading checkpoint %s (ema_export=%s)", path, checkpoint.get("ema_export"))
        _checkpoint.load_model_state(self.model, state)
        self.model.eval()

    def build_dataloader(self, configs):
        dataset = ConstraintInferenceDataset(configs)
        # Shard without DistributedSampler padding: do not duplicate candidates.
        assigned = list(range(DIST_WRAPPER.rank, len(dataset), DIST_WRAPPER.world_size))
        return DataLoader(dataset, batch_size=1, sampler=assigned,
                          collate_fn=collate_fn_identity, num_workers=configs.num_workers)


def main(configs):
    runner = ConstraintInferenceRunner(configs)
    try:
        infer_predict(runner, configs)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run(argv=None):
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config', default=None)
    args, native_args = parser.parse_known_args(argv)
    configs = update_gpu_compatible_configs(_config.load_inference_config(args.config, native_args))
    download_inference_cache(configs)
    main(configs)


if __name__ == "__main__":
    run()
