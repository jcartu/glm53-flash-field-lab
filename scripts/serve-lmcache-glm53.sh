#!/usr/bin/env bash
set -euo pipefail

# Two-arm server for D-Rock's LMCache validation on GLM-5.3-Flash-NVFP4.
# Drives the community image's own launcher (env contract):
#   voipmonitor/vllm:glm53-flash-nvfp4-dflash2-community-20260830-r5
#   + LMCache integration/glm53-upstream-consolidation @ 6e479686
#
#   LMCACHE=0 -> LMCacheConnectorV1 off (baseline; lmcache installed but inert)
#   LMCACHE=1 -> LMCacheConnectorV1 + L2 local disk (treatment)
#
# Allocator note: image sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
# which LMCacheConnectorV1 rejects unless cumem is enabled; and cumem
# (CuMemAllocator pool, expandable off) OOMs at CUDAGRAPH_MODE=FULL here.
# Arm 1 therefore runs with ALLOC_CONF="" (flag unset) and no cumem.
#
# Usage: LMCACHE=1 L2_GB=2 ALLOC_CONF= ./serve-lmcache-glm53.sh [extra vllm args]

NAME="${NAME:-lmcache-glm53-5002}"
IMAGE="${IMAGE:-local/lmcache-glm53:drock-6e479686}"
MODEL_DIR="${MODEL_DIR:-/mnt/2king/models/GLM-5.3-Flash-NVFP4}"
DRAFT_DIR="${DRAFT_DIR:-/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8}"
L2_DIR="${L2_DIR:-/mnt/2king/lmcache-l2-test}"
PORT="${PORT:-5002}"
LMCACHE="${LMCACHE:-0}"
L2_GB="${L2_GB:-50}"
CHUNK="${CHUNK:-256}"
DCP="${DCP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GMU="${GPU_MEMORY_UTILIZATION:-0.90}"
ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"

kv_args=()
lmc_env=()
if [[ -n "${KV_TRANSFER_CONFIG:-}" ]]; then
  kv_args=(--kv-transfer-config "${KV_TRANSFER_CONFIG}")
elif [[ "${LMCACHE}" == "1" ]]; then
  kv_args=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
  lmc_env=(
    -e LMCACHE_CHUNK_SIZE="${CHUNK}"
    -e LMCACHE_LOCAL_CPU=False
    -e LMCACHE_MAX_LOCAL_CPU_SIZE=10
    -e LMCACHE_MAX_LOCAL_DISK_SIZE="${L2_GB}"
    -e LMCACHE_LOCAL_DISK="file:///l2"
    -e LMCACHE_LOG_LEVEL=INFO
  )
fi
docker rm -f "${NAME}" >/dev/null 2>&1 || true
sleep 2

docker run -d --name "${NAME}" \
  --network host \
  --gpus all \
  --ipc host \
  --shm-size 32g \
  --restart unless-stopped \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DRAFT_DIR}:/draft-mxfp8:ro" \
  -v "${L2_DIR}:/l2" \
  -e PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
  -e MODEL=/model \
  -e SERVED_MODEL_NAME=GLM-5.3 \
  -e PORT="${PORT}" \
  -e TP=4 \
  -e DCP="${DCP}" \
  -e SPECULATOR=dflash \
  -e DFLASH_MODEL=/draft-mxfp8 \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e GPU_MEMORY_UTILIZATION="${GMU}" \
  -e ENABLE_PREFIX_CACHING=0 \
  "${lmc_env[@]}" \
  "${IMAGE}" \
  "${kv_args[@]}" \
  "$@"

echo "[serve] arm=LMCACHE=${LMCACHE} DCP=${DCP} L2_GB=${L2_GB} alloc='${ALLOC_CONF}' port=${PORT} container=${NAME}"
echo "[serve] waiting for health..."
for i in $(seq 1 90); do
  s=$(curl -s -o /dev/null -w "%{http_code}" "localhost:${PORT}/health" 2>/dev/null || true)
  [[ "$s" == "200" ]] && { echo "[serve] healthy after $((i*10))s"; exit 0; }
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[serve] FATAL: container exited early"; docker logs "${NAME}" 2>&1 | tail -40; exit 1
  fi
  sleep 10
done
echo "[serve] TIMEOUT waiting for health"; docker logs "${NAME}" 2>&1 | tail -40; exit 1
