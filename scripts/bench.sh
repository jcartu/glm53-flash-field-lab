#!/usr/bin/env bash
set -euo pipefail
# Standard decode/prefill matrix for the LMCache A/B. Usage: bench.sh <port> <out.json>
PORT="${1:-5002}"
OUT="${2:?out json required}"
cd /home/josh/qwen-project/llm-inference-bench
python3 llm_decode_bench.py \
  --port "${PORT}" \
  --model GLM-5.3 \
  --concurrency 1,4,8 \
  --contexts 0,16k,32k,64k,128k \
  --duration 30 \
  --max-tokens 8192 \
  --output "${OUT}"
