#!/usr/bin/env bash
set -euo pipefail
NAME=r24-replay
IMAGE=voipmonitor/vllm@sha256:ab4ff9d6fef85c49d372714e89f014fcb66c6b247c0e3f341eb56dc798fdd0cd

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --init \
  --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 128g \
  -v /mnt/2king/models/GLM-5.3-Flash-NVFP4:/model:ro \
  -v /mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8:/draft-mxfp8:ro \
  -v r24-runtime-cache:/cache \
  -v /mnt/2king/lmcache-l2:/lmcache-l2 \
  -e MODEL=/model -e SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4 \
  -e PORT=5001 -e TP=4 -e DCP=4 \
  -e CACHE_MODE=lmcache -e KV_CACHE_QUANT=nvfp4_ds_mla \
  -e LMCACHE_CHUNK_SIZE=4096 -e LMCACHE_TARGET_TOKEN_BUDGET=4096 \
  -e LMCACHE_L1_SIZE_GB=64 -e LMCACHE_L2_ENABLED=1 -e LMCACHE_L2_ROOT=/lmcache-l2 \
  -e CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  -e MAX_MODEL_LEN=1048576 -e MAX_NUM_SEQS=32 \
  -e MAX_NUM_BATCHED_TOKENS=4096 -e PREFILL_SCHEDULE_INTERVAL=1 \
  -e FAIRNESS_ENGINE=compute_share -e PREFILL_COMPUTE_SHARE=0.4 \
  -e GPU_MEMORY_UTILIZATION=0.95 \
  -e SPECULATOR=dflash2 -e DFLASH_DEPTH=7 \
  -e DFLASH_MODEL=/draft-mxfp8 -e DFLASH_MODEL_REVISION= \
  "$IMAGE" >/dev/null
for i in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:5001/health >/dev/null 2>&1; then
    echo READY
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo FAILED
    docker logs "$NAME" 2>&1 | tail -80
    exit 1
  fi
  sleep 10
done
echo FAILED
exit 1
