#!/usr/bin/env bash
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

# Mol-1 constraint inference: 8 PairMixer blocks (pair width 448) followed by 40 Pairformer
# blocks (pair width 320), 48 trunk blocks in all.
# Input JSON must already contain the required MSA/template features. For raw
# inputs, run `mol prep` first (see docs/infer_json_format.md).
#
# Usage:
#   bash inference_demo.sh  # weights from $MOL_ROOT_DIR/checkpoint/mol-1-constraint.pt
#   CHECKPOINT_PATH=/path/to/model.pt INPUT_JSON_PATH=/path/to/input.json \
#     DUMP_DIR=./output/constraint bash inference_demo.sh
#   DRY_RUN=1 bash inference_demo.sh
# Extra runner flags are appended last, so individual settings can be overridden:
#   CHECKPOINT_PATH=/path/to/model.pt bash inference_demo.sh \
#     --infer_setting.chunk_size 512
# Values reach the config exactly as the shell passes them: quote a path that
# contains spaces, and pass "" to empty a list option.
#
# The runner accepts public inference settings as options. Where mol cpred names an
# option differently:
#   mol cpred          mol/constraint/inference.py
#   --input            --input_json_path
#   --out_dir          --dump_dir
#   --cycle            --model.N_cycle
#   --step             --sample_diffusion.N_step
#   --sample           --sample_diffusion.N_sample
#   --trimul_kernel    --triangle_multiplicative
#   --triatt_kernel    --triangle_attention
# The rest, including the memory settings in docs/inference_memory.md, are
# spelled the same.
# Set MOL_ROOT_DIR to the data/cache root; it also holds the default
# checkpoint directory. Unset, it falls back to the home directory.
# The default 7PZB binding-site input reuses existing MSAs and has no template.
# Default kernels: cuequivariance multiplication and attention.
# Optional FlashPairformer: TRIANGLE_MULTIPLICATIVE=flashpairformer bash inference_demo.sh
# For that backend, install the pinned opt_core package or set OPT_CORE_DIR
# to its source root (see docs/kernels.md).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OPT_CORE_DIR="${OPT_CORE_DIR:-}"
export PYTHONPATH="${SCRIPT_DIR}${OPT_CORE_DIR:+:${OPT_CORE_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

INPUT_JSON_PATH="${INPUT_JSON_PATH:-${SCRIPT_DIR}/examples/7pzb_constraint_binding_site/input.json}"
DUMP_DIR="${DUMP_DIR:-${SCRIPT_DIR}/test_outputs/constraint}"
# Omit the checkpoint option by default so the shared resolver can download
# mol-1-constraint.pt when missing. Explicit paths must already exist.
checkpoint_args=()
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
    checkpoint_args=(--load_checkpoint_path "$CHECKPOINT_PATH")
fi

# Architecture and chunking use the inference config defaults.
# Paths, sampling and feature/kernel switches are passed explicitly.
config_args=()
if [[ -n "${CONFIG:-}" ]]; then
    config_args=(--config "$CONFIG")
fi

inference_command=(
    "${PYTHON:-python3}" "${SCRIPT_DIR}/mol/constraint/inference.py"
    ${config_args[@]+"${config_args[@]}"}
    ${checkpoint_args[@]+"${checkpoint_args[@]}"}
    --input_json_path "${INPUT_JSON_PATH}"
    --dump_dir "${DUMP_DIR}"
    --seeds "${SEEDS:-101}"
    --use_seeds_in_json "${USE_SEEDS_IN_JSON:-false}"
    --model.N_cycle "${N_CYCLE:-20}"
    --sample_diffusion.N_step "${N_STEP:-200}"
    --sample_diffusion.N_sample "${N_SAMPLE:-5}"
    --use_template "${USE_TEMPLATE:-false}"
    --use_rna_msa "${USE_RNA_MSA:-false}"
    --device "${DEVICE:-auto}"
    --triangle_attention "${TRIANGLE_ATTENTION:-cuequivariance}"
    --triangle_multiplicative "${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]; then
    printf '%q ' "${inference_command[@]}"
    printf '\n'
    exit 0
fi

exec "${inference_command[@]}"
