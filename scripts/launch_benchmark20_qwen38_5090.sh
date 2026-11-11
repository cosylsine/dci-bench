#!/usr/bin/env bash
# Current 20-task Lite/Full smoke. Default: plan only, no model calls.
set -Eeuo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
BENCH_ENV=/mnt/afs/250010100/miniconda/envs/dci-bench
SERVING_ENV=/mnt/afs/250010100/miniconda/envs/dci-vllm-qwen38
BENCH_PY="$BENCH_ENV/bin/python"
SERVING_PY="$SERVING_ENV/bin/python"
SUITE=/mnt/afs2/202609/embedding_models/dci-bench/eval_data/benchmark_v2_20261103_v1

export PATH="$BENCH_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$BENCH_ENV/lib"
export PYTHONPATH="$REPO_DIR" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TSX_DISABLE_CACHE=1
export DCI_EVALUATION_TRACK=document
unset DCI_PROTOCOL_PROFILE DCI_PROTOCOL_CONTRACT_SHA256 DCI_CODE_RUNTIME_CONTRACT
unset PYTHONHOME NODE_PATH NODE_OPTIONS KV_CACHE_DTYPE VLLM_NVFP4_GEMM_BACKEND
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_PATH=/mnt/afs2/202610/Qwen3.8-27B SERVED_MODEL_NAME=Qwen3.8-27B
export TENSOR_PARALLEL_SIZE=4 MAX_MODEL_LEN=65536 DTYPE=bfloat16
export TOOL_CALL_PARSER=qwen3_coder REASONING_PARSER=qwen3 LANGUAGE_MODEL_ONLY=1
export VLLM_HOST=127.0.0.1 VLLM_PORT="${VLLM_PORT:-29099}"
export VLLM_MAX_NUM_SEQS=4 VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
export VLLM_MAX_NUM_BATCHED_TOKENS=2048 VLLM_EXTRA_ARGS=--enforce-eager
export VLLM_WORKER_CLS=dci_bench.backends.vllm_gdn_worker.GDNWarmupWorker
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BRIDGE_PORT="${BRIDGE_PORT:-29131}"

[[ $# == 0 || ( $# == 1 && ( "$1" == --preflight-only || "$1" == --serving-smoke-only ) ) ]] || {
  echo 'Usage: DRY_RUN=0 bash scripts/launch_benchmark20_qwen38_5090.sh [--preflight-only|--serving-smoke-only]' >&2
  exit 2
}
[[ -x "$BENCH_PY" && -x "$SERVING_PY" && -f "$MODEL_PATH/config.json" ]]
runner=("$BENCH_PY" scripts/run_benchmark_v2.py --suite "$SUITE" --split smoke
        --concurrency 4 --token-budget 64000 --bridge-port "$BRIDGE_PORT")
printf '[benchmark20] 20 tasks x Lite/Full = 40 executions; TP=4 BF16; context=65536; budget=64000\n'
if [[ "${DRY_RUN:-1}" != 0 && "${1:-}" != --preflight-only ]]; then
  "${runner[@]}"
  exit 0
fi

RUN_ROOT="$REPO_DIR/results/Qwen3.8-27B/benchmark-v2-smoke-runs"
mkdir -p "$RUN_ROOT"
RUN_DIR="$(mktemp -d "$RUN_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
export RUN_DIR
exec > >(tee "$RUN_DIR/launcher.log") 2>&1
printf '[benchmark20] output=%s\n' "$RUN_DIR"
printf '[benchmark20] worker=%s; gpu_memory_utilization=%s; max_num_batched_tokens=%s\n' \
  "$VLLM_WORKER_CLS" "$VLLM_GPU_MEMORY_UTILIZATION" "$VLLM_MAX_NUM_BATCHED_TOKENS"

# Check data and the actual node sandbox before loading any model weights.
if [[ "${1:-}" != --serving-smoke-only ]]; then
  "${runner[@]}" --preflight-only --output "$RUN_DIR/data-preflight"
  bash scripts/ensure_pool_smoke_tools.sh
  "$BENCH_PY" - "$RUN_DIR" <<'PY'
from pathlib import Path
import sys
from scripts.inspect_local_runtime import prepare
prepare(Path(sys.argv[1]) / 'sandbox-tools')
PY
  # This generic transport probe has no pool metadata. Use its default profile;
  # run_benchmark_v2 selects Lite/Full with verified candidate IDs for real tasks.
  "$BENCH_PY" scripts/check_inspect_parallel.py \
    --runtime "$REPO_DIR" --tools-directory "$RUN_DIR/sandbox-tools" \
    --output "$RUN_DIR/sandbox-preflight" --max-concurrency 4
  if [[ "${1:-}" == --preflight-only ]]; then
    echo '[benchmark20] CPU data and sandbox preflight passed; no GPU/model calls.'
    exit 0
  fi
fi

# Preserve the serving environment separately from the benchmark interpreter.
serving=(env PATH="$SERVING_ENV/bin:$PATH" LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu)
"${serving[@]}" "$SERVING_PY" scripts/harness_pilot_runtime.py --output "$RUN_DIR/runtime.json"
"$BENCH_PY" - "$RUN_DIR/runtime.json" "$VLLM_PORT" "$BRIDGE_PORT" <<'PY'
import json, socket, sys
runtime = json.load(open(sys.argv[1]))
gpus = runtime['cuda']['gpus']
if len(gpus) != 4 or any('5090' not in g['name'] or g['capability'] != [12, 0] for g in gpus):
    raise SystemExit('This launcher requires exactly four visible RTX 5090 / sm120 GPUs')
ports = [int(sys.argv[2]), *range(int(sys.argv[3]), int(sys.argv[3]) + 4)]
if len(set(ports)) != len(ports):
    raise SystemExit('Model and bridge ports overlap')
for port in ports:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
PY
# Compile and execute the failing GDN kernels before reading 55 GB of weights.
# This separate probe does not populate the worker's in-process autotune cache;
# GDNWarmupWorker repeats warmup inside profiling before KV cache allocation.
"${serving[@]}" "$SERVING_PY" scripts/check_gdn_kernels.py \
  --model-config "$MODEL_PATH/config.json" --tensor-parallel-size 4 \
  --output "$RUN_DIR/gdn-kernel-preflight.json"
VLLM_VERSION="$("${serving[@]}" "$SERVING_PY" -c 'from importlib.metadata import version; print(version("vllm"))')"
DCI_PROTOCOL_PROFILE=document-pool-v2-lite "$BENCH_PY" -m dci_bench.backends.manifest \
  --output "$RUN_DIR/backend.json" --model-key Qwen3.8-27B --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" --base-url "http://127.0.0.1:$VLLM_PORT/v1" \
  --vllm-version "$VLLM_VERSION" --tool-call-parser "$TOOL_CALL_PARSER" \
  --reasoning-parser "$REASONING_PARSER" --language-model-only --sampling-backend auto \
  --tensor-parallel-size 4 --context-length 65536 --dtype bfloat16 \
  --max-num-seqs 4 --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --runtime-info "$RUN_DIR/runtime.json"

SERVER_PID=''
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid "${serving[@]}" bash scripts/serve_vllm_qwen38.sh > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for ((attempt=0; attempt<360; attempt++)); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo 'vLLM exited; inspect server.log'; exit 1; }
  if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$VLLM_PORT/v1/models" \
       -o "$RUN_DIR/models.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo 'vLLM startup timed out; inspect server.log'; exit 1; }
"$BENCH_PY" - "$RUN_DIR/models.json" <<'PY'
import json, sys
assert 'Qwen3.8-27B' in [row['id'] for row in json.load(open(sys.argv[1]))['data']]
PY
DCI_PROTOCOL_PROFILE=document-pool-v2-lite "$BENCH_PY" scripts/harness_pilot_warmup.py \
  --base-url "http://127.0.0.1:$VLLM_PORT/v1" --model "$SERVED_MODEL_NAME" \
  --output "$RUN_DIR/warmup.json" --concurrency 4 --deadline-seconds 600 --http-timeout-seconds 600
if [[ "${1:-}" == --serving-smoke-only ]]; then
  echo '[benchmark20] Serving smoke passed: model loaded and synthetic protocol requests passed; benchmark not executed.'
  exit 0
fi
"${runner[@]}" --backend-manifest "$RUN_DIR/backend.json" --execute --output "$RUN_DIR/smoke"
printf '[benchmark20] Finished; inspect %s/smoke/summary.json for completed/valid counts.\n' "$RUN_DIR"
