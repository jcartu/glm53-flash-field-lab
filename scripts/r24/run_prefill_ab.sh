#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/josh/omp-workspace/drock-lmcache/r24-battery

docker rm -f r24-smoke r24-replay >/dev/null 2>&1 || true
"/home/josh/omp-workspace/drock-lmcache/r24-smoke.sh"
python3 "$ROOT/prefill_repeat.py" 5001 gpu-only "$ROOT/prefill-gpu-only.json" 8
docker rm -f r24-smoke >/dev/null
sleep 5
"$ROOT/replay_server.sh"
python3 "$ROOT/prefill_repeat.py" 5001 lmcache "$ROOT/prefill-lmcache.json" 8
docker rm -f r24-replay >/dev/null
echo PREFILL_AB_DONE
