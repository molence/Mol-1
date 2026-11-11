#!/bin/bash
# Run a bench inside the allocation, one GPU per call.
# Usage: bench/run.sh <gpu> <tag> [bench args / -- runner args]
# Env: JOBID (allocation), MOL_IMAGE (container .sqsh), optional MOL_MOUNTS / MOL_COMPILE_HOME, EXTRA_ENV ("K=V K=V" exported inside the container)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${MOL_IMAGE:?set MOL_IMAGE to the container image (.sqsh)}
MOUNTS=${MOL_MOUNTS:-$(dirname "$REPO"):$(dirname "$REPO")}
COMPILE_HOME=${MOL_COMPILE_HOME:-$REPO/../compile_home}
GPU=$1; TAG=$2; shift 2
srun --jobid="${JOBID}" --overlap -N1 -n1 --gpus-per-node=4 \
  --container-image=$IMAGE --container-mounts=$MOUNTS --no-container-mount-home \
  bash -c "export HOME=$COMPILE_HOME MOL_ROOT_DIR=$REPO/data_cache CUDA_VISIBLE_DEVICES=$GPU ${EXTRA_ENV:-}; cd $REPO && $REPO/data_cache/venv/bin/python bench/mol_bench.py --tag $TAG $*" \
  > $REPO/bench/logs/$TAG.log 2>&1
grep -E "MEDIAN|Error|error" $REPO/bench/logs/$TAG.log | tail -5
