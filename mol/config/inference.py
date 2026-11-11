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
# Upstream file: configs/configs_inference.py

# pylint: disable=C0114
import os
from pathlib import Path

from mol.config.base import model_configs as base_model_configs
from mol.config.types import ListValue, RequiredValue, ValueMaybeNone

MOL_ROOT_DIR = os.environ.get("MOL_ROOT_DIR", str(Path.home()))
inference_configs = {
    "model_name": "mol-1-base",  # inference model selection
    # auto prefers CUDA, then Apple MPS, then CPU. An explicit device that is
    # unavailable is an error, not a silent downgrade to CPU.
    "device": "auto",  # auto | cuda | cuda:N | mps | cpu
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": True,
    "input_json_path": RequiredValue(str),
    "load_checkpoint_dir": os.path.join(MOL_ROOT_DIR, "checkpoint"),
    "num_workers": 0,
    "use_msa": True,
    # FP32 matrix multiplications stay FP32; true runs them on TF32 tensor
    # cores, 4-8% faster with slightly different results.
    "enable_tf32": False,
    # Up to 2560 tokens the confidence head and diffusion sampling run in FP32
    # (BF16 above). True runs them under BF16 autocast at every size: on GB300
    # with 644 tokens the confidence head goes from 0.95 s to 0.32 s, at
    # slightly different results.
    "bf16_confidence_head": False,
    "bf16_sample_diffusion": False,
    "enable_efficient_fusion": True,
    "enable_diffusion_shared_vars_cache": True,
    "diffusion_cache_mode": "full",  # full | shared | off
    "msa_pair_as_unpair": True,
    "use_template": False,
    "use_rna_msa": False,
    "use_seeds_in_json": False,
    # Where prediction differs from evaluation during training: no Monte Carlo
    # dropout in the recycles, and one diffusion sample at a time.
    "mc_dropout_apply_rate": 0.0,
    "mc_dropout_rate": 0.0,
    "infer_setting": {
        **base_model_configs["infer_setting"],
        "sample_diffusion_chunk_size": ValueMaybeNone(1),
    },
}
