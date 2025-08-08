#!/bin/bash

pkill python3 -9 && sleep 3

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu
export TORCHDYNAMO_VERBOSE=1
export NCCL_DEBUG=INFO
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1

if /opt/rocm/bin/rocm_agent_enumerator 2>/dev/null | grep -v gfx000 >/dev/null; then
  SIZE=$(/opt/rocm/bin/rocm_agent_enumerator | grep -v gfx000 | wc -l)
else
  SIZE=$(ls -1 /dev/nvidia[0-9]* | wc -l)
fi

LOCAL_SIZE=${LOCAL_SIZE:-${SIZE}}

echo "*******************************************************"
echo "Using ${LOCAL_SIZE} GPU(s) for LLM Inference (\"-e LOCAL_SIZE=${LOCAL_SIZE}\").."
echo "*******************************************************"

if [[ "${LOCAL_SIZE}" == "1" ]]; then
  python3 $(dirname $0)/llm_moe_tutel.py "$@"
else
  OMP_NUM_THREADS=4 python3 -m torch.distributed.run --nproc_per_node=${LOCAL_SIZE:-8} $(dirname $0)/llm_moe_tutel.py "$@"
fi
