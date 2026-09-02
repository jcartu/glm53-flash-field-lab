#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash-NVFP4 (local-inference-lab) on 4x RTX PRO 6000 (SM120, TP4/EP4), SGLang.
# Host :5001 (host networking). SOTA config assembled from:
#   - SGLang GLM-5.3-Flash cookbook cells (GB300 low-latency: adaptive MTP 5/1/6, FP8 KV)
#   - LibertAIDAI verified sm_121 recipe (TileLang DSA + bf16 KV fallback, load-bearing flags)
#   - tonyd2wild day-0 fixes (TileLang smem patch for sm_12x, quant map naming)
# Fallback to the verified path: DSA_PREFILL=tilelang DSA_DECODE=tilelang KV_CACHE_DTYPE=bfloat16 MTP=0

NAME="${NAME:-glm53-flash-nvfp4-5001}"
IMAGE="${IMAGE:-local/sglang:glm-5.3-flash-accel}"
MODEL_DIR="${MODEL_DIR:-/mnt/2king/models/GLM-5.3-Flash-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3}"
PORT="${PORT:-5001}"
TP="${TP:-4}"
QUANT="${QUANT:-modelopt_mixed}"
EP="${EP:-4}"
MEM_FRAC="${MEM_FRAC:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
MAMBA_RATIO="${MAMBA_RATIO:-0.30}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"
MAMBA_CACHE_SIZE="${MAMBA_CACHE_SIZE:-128}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
DSA_PREFILL="${DSA_PREFILL:-tilelang}"
DSA_DECODE="${DSA_DECODE:-tilelang}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
MTP="${MTP:-dflash2}"
DRAFT_QUANT="${DRAFT_QUANT:-unquant}"

# Speculative decoding mode: dflash2 (default, fastest+verified), mtp, or 0/off.
SPEC="${MTP}"
mtp_args=()
case "${SPEC}" in
  dflash2)
    DRAFT_DIR="${DRAFT_DIR:-/mnt/2king/models/GLM-5.3-Flash-DFlash2}"
    mtp_args=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path /draft
      --speculative-draft-attention-backend fa4
      --speculative-draft-model-quantization "${DRAFT_QUANT}"
    )
    ;;
  1|mtp)
    # Adaptive MTP 5/1/6 (cookbook low-latency row; NEXTN normalizes to EAGLE).
    mtp_args=(
      --speculative-algorithm EAGLE
      --speculative-num-steps 5
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 6
      --speculative-adaptive
    )
    ;;
esac

for container in glm53-flash-nvfp4-5001 glm53-full-exl3-5001; do
  docker rm -f "${container}" >/dev/null 2>&1 || true
done
sleep 2

draft_mount=()
if [[ -n "${DRAFT_DIR:-}" ]]; then
  draft_mount=(-v "${DRAFT_DIR}:/draft:ro")
fi

exec docker run -d --name "${NAME}" \
  --network host \
  --gpus all \
  --shm-size 32g \
  --ipc host \
  --restart unless-stopped \
  -v "${MODEL_DIR}:/model:ro" \
  "${draft_mount[@]}" \
  -v sglang-glm53-cache:/root/.cache \
  -v /home/josh/omp-workspace/glm53-patches/tilelang_kernel.py:/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py:ro \
  -v /home/josh/omp-workspace/glm53-patches/glm5_next.py:/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py:ro \
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path /model \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --quantization "${QUANT}" \
  --tp-size "${TP}" \
  --ep-size "${EP}" \
  --port "${PORT}" \
  --attention-backend dsa \
  --dsa-prefill-backend "${DSA_PREFILL}" \
  --dsa-decode-backend "${DSA_DECODE}" \
  --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --mamba-full-memory-ratio "${MAMBA_RATIO}" \
  --chunked-prefill-size "${CHUNKED_PREFILL}" \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
  --return-hidden-states-mode "${HIDDEN_STATES_MODE:-full}" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static "${MEM_FRAC}" \
  --context-length "${CONTEXT_LENGTH}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --enable-metrics \
  "${mtp_args[@]}" \
  "$@"
