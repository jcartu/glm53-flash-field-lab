#!/usr/bin/env bash
set -euo pipefail

# LMCache MP (ZMQ) sidecar server for D-Rock's branch, GLM-5.3-Flash validation.
# Owns L1 + L2 local disk; vLLM workers connect via tcp://localhost:5555.
# chunk-size must be a multiple of (vLLM block 256 * DCP 4 * mamba factor) = 9216
# per LMCacheMPConnector validation under DCP4.

NAME="${NAME:-lmcache-mp-server}"
IMAGE="${IMAGE:-local/lmcache-glm53:drock-6e479686}"
L2_DIR="${L2_DIR:-/mnt/2king/lmcache-l2-test}"
L2_GB="${L2_GB:-50}"
PORT="${PORT:-5555}"
HTTP_PORT="${HTTP_PORT:-8081}"
CHUNK="${CHUNK:-9216}"
L1_GB="${L1_GB:-10}"

mkdir -p "${L2_DIR}"

docker run -d --name "${NAME}" \
  --network host \
  --gpus all \
  --ipc host \
  -v "${L2_DIR}:/l2" \
  -e LMCACHE_LOCAL_DISK="file:///l2" \
  -e LMCACHE_MAX_LOCAL_DISK_SIZE="${L2_GB}" \
  -e LMCACHE_LOG_LEVEL=INFO \
  --entrypoint /opt/venv/bin/python \
  "${IMAGE}" \
  -u -m lmcache.v1.multiprocess.http_server \
    --host localhost --port "${PORT}" \
    --http-host 127.0.0.1 --http-port "${HTTP_PORT}" \
    --chunk-size "${CHUNK}" \
    --l1-size-gb "${L1_GB}" \
    --eviction-policy LRU

echo "[mp-server] waiting for Uvicorn banner on HTTP ${HTTP_PORT}..."
for i in $(seq 1 45); do
  if docker logs "${NAME}" 2>&1 | grep -q "Uvicorn running"; then
    echo "[mp-server] up after $((i*2))s"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "[mp-server] FATAL: exited"; docker logs "${NAME}" 2>&1 | tail -30; exit 1
  fi
  sleep 2
done
echo "[mp-server] TIMEOUT; last logs:"; docker logs "${NAME}" 2>&1 | tail -15
exit 1
