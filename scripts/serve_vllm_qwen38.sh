#!/usr/bin/env bash
set -euo pipefail

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/mnt/afs2/202608/embedding_models/dci-bench/pretrained_models/MiniCPM5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniCPM5-2B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-minicpm5}"
REASONING_PARSER="${REASONING_PARSER:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

EXTRA_ARGS=()
if [[ -n "$REASONING_PARSER" ]]; then
  EXTRA_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "$KV_CACHE_DTYPE" ]]; then
  EXTRA_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
  EXTRA_ARGS+=(--language-model-only)
fi
if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
  read -r -a USER_EXTRA_ARGS <<<"$VLLM_EXTRA_ARGS"
  EXTRA_ARGS+=("${USER_EXTRA_ARGS[@]}")
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --enable-auto-tool-choice \
  "${EXTRA_ARGS[@]}"
