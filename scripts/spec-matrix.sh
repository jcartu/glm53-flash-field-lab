#!/usr/bin/env bash
set -uo pipefail
# Speculator matrix for Festr's chart: image × speculator × C1/8/16 at ctx0.
cd /home/josh/omp-workspace/drock-lmcache
run_cell() { # image_label image_ref speculator_env...
  local label="$1"; shift
  local image="$1"; shift
  docker rm -f r15-test >/dev/null 2>&1
  docker run -d --name r15-test --init --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 32g \
    -v /mnt/2king/models/GLM-5.3-Flash-NVFP4:/model:ro \
    -v /mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8:/draft-mxfp8:ro \
    -e MODEL=/model -e SERVED_MODEL_NAME=GLM-5.3-Flash-NVFP4 -e PORT=5002 -e TP=4 -e DCP=1 \
    -e MAX_MODEL_LEN=262144 -e MAX_NUM_SEQS=16 -e MAX_NUM_BATCHED_TOKENS=4096 \
    -e MAX_CUDAGRAPH_CAPTURE_SIZE=128 -e GPU_MEMORY_UTILIZATION=0.90 \
    -e B12X_PCIE_ALLREDUCE=1 -e NCCL_MIN_NCHANNELS=32 -e NCCL_MAX_NCHANNELS=32 \
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS \
    -e NCCL_PROTO=LL,LL128,Simple -e OMP_NUM_THREADS=2 \
    "$@" "${image}" >/dev/null 2>&1
  local ok=0
  for i in $(seq 1 75); do
    s=$(curl -s -o /dev/null -w "%{http_code}" localhost:5002/health 2>/dev/null || echo 000)
    [ "$s" = "200" ] && { ok=1; break; }; sleep 10
  done
  [ $ok = 1 ] || { echo "$label UNHEALTHY"; return; }
  sleep 20
  (cd /home/josh/qwen-project/llm-inference-bench && python3 llm_decode_bench.py --port 5002 \
    --model GLM-5.3-Flash-NVFP4 --concurrency 1,8,16 --contexts 0 --skip-prefill \
    --duration 30 --max-tokens 1024 --output "/home/josh/omp-workspace/drock-lmcache/chart-${label}.json" >/dev/null 2>&1)
  python3 -c "
import json
d = json.load(open('chart-${label}.json'))
vals = [(r.get('concurrency'), r['client_output_tokens']/(r.get('duration_s') or 30)) for r in d['results']]
print('${label}', ' '.join(f'C{c}={v:.0f}' for c,v in vals))"
}
IMG_R8="voipmonitor/vllm@sha256:488ddf752938b5ab17e3083dd7d5bb84f418bc3f8856f93cc514c8b66abbe4c6"
IMG_R12="voipmonitor/vllm:jovian-judgement-community-20260901-r12"
IMG_R15="voipmonitor/vllm:jovian-judgement-community-20260902-r15"
run_cell r8-mtp0  "$IMG_R8"  -e SPECULATOR=mtp -e MTP=0 -e NUM_SPECULATIVE_TOKENS=0
run_cell r12-mtp0 "$IMG_R12" -e SPECULATOR=mtp -e MTP=0 -e NUM_SPECULATIVE_TOKENS=0
run_cell r12-mtp3 "$IMG_R12" -e SPECULATOR=mtp -e MTP=3 -e NUM_SPECULATIVE_TOKENS=3
run_cell r15-mtp3 "$IMG_R15" -e SPECULATOR=mtp -e MTP=3 -e NUM_SPECULATIVE_TOKENS=3
run_cell r15-dflash2 "$IMG_R15" -e SPECULATOR=dflash2 -e DFLASH_MODEL=/draft-mxfp8 -e DFLASH_MODEL_REVISION= -e NUM_SPECULATIVE_TOKENS=7
echo MATRIX-CHUNK-1-DONE
