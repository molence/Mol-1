#!/bin/bash
# Run a python command inside the container on the allocation (CPU/GPU 0 shared): bench/py.sh <args...>
REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/Mol-1
srun --jobid=${JOBID} --overlap -N1 -n1 --container-image=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/containers/trtllm-pytorch-26.09-py3-sbsa-ubuntu24.04-skip-tritondevel-202611020300-19783.sqsh \
  --container-mounts=/scratch:/scratch,/lustre:/lustre --no-container-mount-home \
  bash -c "export HOME=$REPO/../compile_home; cd $REPO && data_cache/venv/bin/python $*"
