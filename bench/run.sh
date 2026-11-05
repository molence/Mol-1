#!/bin/bash
# Run a bench inside the allocation, one GPU per call.
# Usage: bench/run.sh <gpu> <tag> [bench args / -- runner args]
# Env: JOBID (allocation), EXTRA_ENV ("K=V K=V" exported inside the container)
set -uo pipefail
REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/Mol-1
IMAGE=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/containers/trtllm-pytorch-26.09-py3-sbsa-ubuntu24.04-skip-tritondevel-202611020300-19783.sqsh
GPU=$1; TAG=$2; shift 2
srun --jobid="${JOBID}" --overlap -N1 -n1 --gpus-per-node=4 \
  --container-image=$IMAGE --container-mounts=/scratch:/scratch,/lustre:/lustre --no-container-mount-home \
  bash -c "export HOME=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/compile_home MOL_ROOT_DIR=$REPO/data_cache CUDA_VISIBLE_DEVICES=$GPU ${EXTRA_ENV:-}; cd $REPO && $REPO/data_cache/venv/bin/python bench/mol_bench.py --tag $TAG $*" \
  > $REPO/bench/logs/$TAG.log 2>&1
grep -E "MEDIAN|Error|error" $REPO/bench/logs/$TAG.log | tail -5
