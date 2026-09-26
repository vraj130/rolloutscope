#!/usr/bin/env bash
# Serve the M1 proxy judge (PROGRESS.md R-007) with vLLM's OpenAI-compatible server on GPU 1.
#
# Default: official meta-llama/Llama-3.1-8B-Instruct in bf16, pinned to a revision.
# Fallback (only if bf16 does not fit or bottlenecks training): set
#   PROXY_HF_MODEL=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 PROXY_REVISION=<sha>
# No on-the-fly bitsandbytes quantization.
#
# The server runs detached (setsid nohup), so it outlives the shell that started it.
# Usage, from experiments/:
#   bash m1/serve_proxy.sh <run_name>     # logs to $ROLLOUTSCOPE_DATA/m1/<run_name>/proxy.log
#   curl -sf 127.0.0.1:8001/health        # 200 once "Application startup complete" is logged
set -euo pipefail

RUN_NAME="${1:?usage: serve_proxy.sh <run_name>}"
: "${ROLLOUTSCOPE_DATA:?set ROLLOUTSCOPE_DATA}"
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="${HF_HOME:-$ROLLOUTSCOPE_DATA/hf-cache}"
# The HF token lives in the default location, not in the NAS HF_HOME. Without this, gated
# downloads go out anonymously and fail with 401.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
# FlashInfer JIT-compiles its sampler with the system nvcc, which is CUDA 11.8 on fourier and
# too old for it. Use vLLM's PyTorch sampler instead (the judge decodes greedily anyway).
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${PROXY_HF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
REVISION="${PROXY_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}"
PORT="${PROXY_PORT:-8001}"
LOG_DIR="$ROLLOUTSCOPE_DATA/m1/$RUN_NAME"
LOG="$LOG_DIR/proxy.log"
mkdir -p "$LOG_DIR"

setsid nohup uv run vllm serve "$MODEL" \
  --revision "$REVISION" \
  --served-model-name proxy-judge \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization "${PROXY_GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --host 127.0.0.1 \
  --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
echo "proxy judge starting: pid $!, log $LOG"
