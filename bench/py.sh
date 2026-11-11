#!/bin/bash
# Run a python command inside the container on the allocation (CPU/GPU 0 shared): bench/py.sh <args...>
# Env: JOBID, MOL_IMAGE (container .sqsh), optional MOL_MOUNTS, MOL_COMPILE_HOME
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${MOL_IMAGE:?set MOL_IMAGE to the container image (.sqsh)}
MOUNTS=${MOL_MOUNTS:-$(dirname "$REPO"):$(dirname "$REPO")}
COMPILE_HOME=${MOL_COMPILE_HOME:-$REPO/../compile_home}
srun --jobid=${JOBID} --overlap -N1 -n1 --container-image=$IMAGE \
  --container-mounts=$MOUNTS --no-container-mount-home \
  bash -c "export HOME=$COMPILE_HOME; cd $REPO && data_cache/venv/bin/python $*"
