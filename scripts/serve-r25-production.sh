#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-glm53-prod}"
IMAGE="${IMAGE:-voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804}"
MODEL_DIR="${MODEL_DIR:-/mnt/2king/models/GLM-5.3-Flash-NVFP4}"
DRAFT_DIR="${DRAFT_DIR:-/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8}"
L2_DIR="${L2_DIR:-/mnt/2king/lmcache-l2}"
PORT="${PORT:-5001}"

# Private 128 GiB SHM is intentional. Combining --ipc host with --shm-size
# silently uses the host's smaller /dev/shm and breaks 64 GiB native offload.
docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run -d --name "${NAME}" --init --restart unless-stopped \
  --gpus '"device=0,1,2,3"' --network host --shm-size 128g \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DRAFT_DIR}:/draft-mxfp8:ro" \
  -v r25-runtime-cache:/cache \
  -v "${L2_DIR}:/lmcache-l2" \
  -e MODEL=/model -e SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4 \
  -e PORT="${PORT}" -e TP=4 -e DCP=4 \
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
  "${IMAGE}" >/dev/null

echo "[serve] R25 TP4/DCP4 DFlash K7, packed NVFP4 KV, LMCache, fairness 0.4"
echo "[serve] container=${NAME} port=${PORT} private_shm=128g"
for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[serve] healthy after $((i * 10))s"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "[serve] FATAL: container exited"
    docker logs "${NAME}" 2>&1 | tail -80
    exit 1
  fi
  sleep 10
done

echo "[serve] TIMEOUT waiting for health"
docker logs "${NAME}" 2>&1 | tail -80
exit 1
