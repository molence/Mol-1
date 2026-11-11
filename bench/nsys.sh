#!/bin/bash
# nsys trace of the last bench repeat: bench/nsys.sh <gpu> <tag> [bench args / -- runner args]
# Env: JOBID, MOL_IMAGE (container .sqsh), optional MOL_MOUNTS / MOL_COMPILE_HOME, EXTRA_ENV. Writes bench/results/nsys/<tag>.nsys-rep
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${MOL_IMAGE:?set MOL_IMAGE to the container image (.sqsh)}
MOUNTS=${MOL_MOUNTS:-$(dirname "$REPO"):$(dirname "$REPO")}
COMPILE_HOME=${MOL_COMPILE_HOME:-$REPO/../compile_home}
GPU=$1; TAG=$2; shift 2
mkdir -p $REPO/bench/results/nsys
srun --jobid="${JOBID}" --overlap -N1 -n1 --gpus-per-node=4 \
  --container-image=$IMAGE --container-mounts=$MOUNTS --no-container-mount-home \
  bash -c "export HOME=$COMPILE_HOME MOL_ROOT_DIR=$REPO/data_cache CUDA_VISIBLE_DEVICES=$GPU MOL_BENCH_NVTX=1 ${EXTRA_ENV:-}; cd $REPO && \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
      --sample=none --cpuctxsw=none -f true -o bench/results/nsys/$TAG \
      $REPO/data_cache/venv/bin/python bench/mol_bench.py --tag nsys_$TAG --repeats 1 $*" \
  > $REPO/bench/logs/nsys_$TAG.log 2>&1
grep -E "MEDIAN|Error|error|Generated" $REPO/bench/logs/nsys_$TAG.log | tail -5
