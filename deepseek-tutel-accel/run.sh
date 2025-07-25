#!/bin/bash -e

pkill python3 -9 && sleep 3

export TORCHDYNAMO_VERBOSE=1
export NCCL_DEBUG=INFO

FLUSH=${FLUSH:-32}
python3 -m torch.distributed.run --nproc_per_node=${LOCAL_SIZE:-8} $(dirname $0)/llm_moe_tutel.py --buffer_size ${FLUSH} "$@"
