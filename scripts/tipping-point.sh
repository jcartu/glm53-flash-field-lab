#!/usr/bin/env bash
set -uo pipefail
cd /home/josh/omp-workspace/drock-lmcache
IMG="voipmonitor/vllm:jovian-judgement-community-20260831-r8"
COMMON_ENV=(DRAFT_DIR=/mnt/2king/models/GLM-5.3-Flash-DFlash2-MXFP8 LMCACHE=0 IMAGE="$IMG" MAX_NUM_SEQS=32 MAX_CUDAGRAPH_CAPTURE_SIZE=288)
run() { # label, spec env...
  local label="$1"; shift
  docker rm -f lmcache-glm53-5002 >/dev/null 2>&1
  if ! env "${COMMON_ENV[@]}" "$@" ./serve-r7.sh >/dev/null 2>&1; then echo "$label LAUNCH-FAIL"; return; fi
  for i in $(seq 1 12); do s=$(curl -s -o /dev/null -w "%{http_code}" localhost:5002/health); [ "$s" = 200 ] && break; sleep 10; done
  [ "$s" = 200 ] || { echo "$label UNHEALTHY"; return; }
  sleep 30  # warm
  (cd /home/josh/qwen-project/llm-inference-bench && python3 llm_decode_bench.py --port 5002 --model GLM-5.3 --concurrency 16,24,32 --contexts 0 --skip-prefill --duration 30 --max-tokens 1024 --output "/home/josh/omp-workspace/drock-lmcache/tip-${label}.json" >/dev/null 2>&1)
  python3 -c "
import json
d = json.load(open('tip-${label}.json'))
vals = []
for r in d['results']:
    dur = r.get('duration_s') or 30
    vals.append((r.get('concurrency'), r['client_output_tokens']/dur))
print('$label', ' '.join(f'C{c}={v:.0f}' for c,v in vals))"
}
run dflash5 SPECULATOR=dflash2 NUM_SPECULATIVE_TOKENS=5
run dflash3 SPECULATOR=dflash2 NUM_SPECULATIVE_TOKENS=3
run mtp3 SPECULATOR=mtp MTP=3
run mtp2 SPECULATOR=mtp MTP=2
run mtp0 SPECULATOR=mtp MTP=0
echo MATRIX-DONE
