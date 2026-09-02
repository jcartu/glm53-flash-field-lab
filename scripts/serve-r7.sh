#!/usr/bin/env bash
set -euo pipefail

# Unchanged-r7 control per local-inference-lab/rtx6kpro models/glm-5.3-flash.md
# (commit 790db5a7): DFlash2/DCP4+CKV_GATHER, the config D-Rock's published
# control table came from. LMCACHE=1 adds the connector on the overlay image.
# Usage: [LMCACHE=1] ./serve-r7.sh [extra vllm args]

NAME="${NAME:-lmcache-glm53-5002}"
BASE="${BASE:-voipmonitor/vllm@sha256:488ddf752938b5ab17e3083dd7d5bb84f418bc3f8856f93cc514c8b66abbe4c6}"
IMAGE="${IMAGE:-${BASE}}"
MODEL_DIR="${MODEL_DIR:-/mnt/2king/models/GLM-5.3-Flash-NVFP4}"
DRAFT_DIR="${DRAFT_DIR:-/mnt/2king/models/GLM-5.3-Flash-DFlash2}"
L2_DIR="${L2_DIR:-/mnt/2king/lmcache-l2-test}"
PORT="${PORT:-5002}"
LMCACHE="${LMCACHE:-0}"

mkdir -p "${L2_DIR}"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
sleep 2

lmc_env=()
kv_args=()
if [[ "${LMCACHE}" == "1" ]]; then
  if [[ -n "${KV_CONFIG_FILE:-}" ]]; then
    KV_CONFIG="$(cat "${KV_CONFIG_FILE}")"
  else
    KV_CONFIG="${KV_CONFIG:-{\"kv_connector\":\"LMCacheConnectorV1Dynamic\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_connector_v1\",\"kv_role\":\"kv_both\"}}"
  fi
  kv_args=(--kv-transfer-config "${KV_CONFIG}")
  IMAGE="local/lmcache-glm53-r7:drock-6e479686"
  lmc_env=(
    -e LMCACHE_CHUNK_SIZE=4096
    -e LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-64}"
    -e LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-2}"
    -e LMCACHE_INSTANCE_ID=glm53-dflash2-tp4-dcp4
    -e LMCACHE_L2_CONFIG="{\"type\":\"fs_native\",\"base_path\":\"/lmcache-l2\",\"num_workers\":8,\"use_odirect\":false,\"max_capacity_gb\":${LMCACHE_L2_SIZE_GB:-100},\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.2}}"
    -e LMCACHE_LOG_LEVEL=INFO
  )
fi

docker run -d \
  --name "${NAME}" \
  --init \
  --gpus '"device=0,1,2,3"' \
  --network host \
  --ipc host \
  --shm-size 32g \
  --restart unless-stopped \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DRAFT_DIR}:/draft:ro" \
  -v "${L2_DIR}:/lmcache-l2" \
  -e MODEL=/model \
  -e SERVED_MODEL_NAME=GLM-5.3 \
  -e PORT="${PORT}" \
  -e TP=4 \
  -e DCP=4 \
  -e DCP_CKV_GATHER=1 \
  -e SPECULATOR="${SPECULATOR:-dflash2}" \
  -e DFLASH_MODEL="${DFLASH_MODEL:-/draft}" \
  -e DFLASH_MODEL_REVISION= \
  -e NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}" -e MTP="${MTP:-}" \
  -e DFLASH_KV_CACHE_DTYPE=auto \
  -e PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF:-expandable_segments:False}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-128}" \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}" \
  -e B12X_PCIE_ALLREDUCE=1 \
  -e NCCL_MIN_NCHANNELS=32 \
  -e NCCL_MAX_NCHANNELS=32 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e OMP_NUM_THREADS=2 \
  "${lmc_env[@]}" \
  "${IMAGE}" \
  "${kv_args[@]}" \
  "$@"

echo "[serve-r7] arm=LMCACHE=${LMCACHE} image=${IMAGE} port=${PORT}"
echo "[serve-r7] waiting for health (his runbook allows 15m start)..."
for i in $(seq 1 120); do
  s=$(curl -s -o /dev/null -w "%{http_code}" "localhost:${PORT}/health" 2>/dev/null || true)
  [[ "$s" == "200" ]] && { echo "[serve-r7] healthy after $((i*10))s"; exit 0; }
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[serve-r7] FATAL: container exited"; docker logs "${NAME}" 2>&1 | tail -40; exit 1
  fi
  sleep 10
done
echo "[serve-r7] TIMEOUT"; docker logs "${NAME}" 2>&1 | tail -40; exit 1
