#!/usr/bin/env bash
# Current 20-task Lite/Full smoke or calibrated full run. Default: plan only.
set -Eeuo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
BENCH_ENV=/mnt/afs/250010100/miniconda/envs/dci-bench
SERVING_ENV=/mnt/afs/250010100/miniconda/envs/dci-vllm-qwen38
BENCH_PY="$BENCH_ENV/bin/python"
SERVING_PY="$SERVING_ENV/bin/python"
SUITE="${BENCHMARK_SUITE:-/mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_v2_20261103_v1}"

FULL=0
MODE=run
FIXED_BUDGET=''
REQUESTED_RUN_DIR=''
RESUME_RUN=''
while (( $# )); do
  arg="$1"
  shift
  case "$arg" in
    --full) FULL=1 ;;
    --run-dir|--resume-run)
      (( $# )) || { echo "$arg requires a directory" >&2; exit 2; }
      if [[ "$arg" == --run-dir ]]; then REQUESTED_RUN_DIR="$1"; else RESUME_RUN="$1"; fi
      shift ;;
    --token-budget)
      (( $# )) || { echo '--token-budget requires 32000, 64000 or 96000' >&2; exit 2; }
      FIXED_BUDGET="$1"
      shift
      case "$FIXED_BUDGET" in 32000|64000|96000) ;; *) echo 'Invalid token budget' >&2; exit 2 ;; esac ;;
    --preflight-only|--serving-smoke-only)
      [[ "$MODE" == run ]] || { echo 'Choose one diagnostic mode' >&2; exit 2; }
      MODE="$arg" ;;
    *) echo 'Usage: DRY_RUN=0 bash scripts/launch_benchmark20_qwen38_5090.sh [--full [--token-budget 64000]] [--run-dir DIR | --resume-run DIR] [--preflight-only|--serving-smoke-only]' >&2; exit 2 ;;
  esac
done
if [[ -n "$RESUME_RUN" ]]; then
  [[ -z "$REQUESTED_RUN_DIR" && "$FULL" == 1 && -n "$FIXED_BUDGET" && "$MODE" == run ]] || {
    echo '--resume-run requires --full --token-budget and cannot be combined with --run-dir or diagnostics' >&2
    exit 2
  }
fi
if [[ -n "$FIXED_BUDGET" && "$FULL" != 1 ]]; then
  echo '--token-budget requires --full' >&2
  exit 2
fi
fixed_args=()
if [[ -n "$FIXED_BUDGET" ]]; then
  fixed_args=(--fixed-budget-reason "User-requested fixed cumulative budget: $FIXED_BUDGET tokens")
fi
CONCURRENCY="${CONCURRENCY:-4}"
case "$CONCURRENCY" in 4|8|16) ;; *) echo 'CONCURRENCY must be 4, 8 or 16' >&2; exit 2 ;; esac
PREFLIGHT_CONCURRENCY="${PREFLIGHT_CONCURRENCY:-4}"
case "$PREFLIGHT_CONCURRENCY" in 1|2|4|8|16) ;; *) echo 'PREFLIGHT_CONCURRENCY must be 1, 2, 4, 8 or 16' >&2; exit 2 ;; esac
INSPECT_PREFLIGHT_TIMEOUT_SECONDS="${INSPECT_PREFLIGHT_TIMEOUT_SECONDS:-600}"
if [[ ! "$INSPECT_PREFLIGHT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'INSPECT_PREFLIGHT_TIMEOUT_SECONDS must be a positive integer' >&2
  exit 2
fi

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
export DCI_TOKEN_COUNT_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
export DCI_TOKEN_COUNT_MODEL="$SERVED_MODEL_NAME" DCI_TOKEN_COUNT_PYTHON="$BENCH_PY"
export VLLM_MAX_NUM_SEQS="$CONCURRENCY" VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
export VLLM_MAX_NUM_BATCHED_TOKENS=2048 VLLM_EXTRA_ARGS=--enforce-eager
export VLLM_WORKER_CLS=dci_bench.backends.vllm_gdn_worker.GDNWarmupWorker
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BRIDGE_PORT="${BRIDGE_PORT:-29131}"

[[ -x "$BENCH_PY" && -x "$SERVING_PY" && -f "$MODEL_PATH/config.json" ]]
runner=("$BENCH_PY" scripts/run_benchmark_v2.py --suite "$SUITE"
        --concurrency "$CONCURRENCY" --bridge-port "$BRIDGE_PORT")
runtime_args=()
if [[ -n "${DCI_RUNTIME_CONFIG:-}" ]]; then
  DCI_RUNTIME_CONFIG="$(realpath -e -- "$DCI_RUNTIME_CONFIG")"
  export DCI_RUNTIME_CONFIG DCI_BENCH_PYTHON="$BENCH_PY"
  runtime_args=(--runtime-config "$DCI_RUNTIME_CONFIG")
  runner+=("${runtime_args[@]}")
fi
SPLIT=smoke
RUN_KIND=smoke
if [[ "$FULL" == 1 ]]; then
  SPLIT=test
  RUN_KIND=full
  if [[ -n "$FIXED_BUDGET" ]]; then
    printf '[benchmark20] User-fixed budget=%s -> 2000 test executions; development selection skipped\n' "$FIXED_BUDGET"
  else
    printf '[benchmark20] 3 x 400 dev executions -> budget selection -> 2000 test executions\n'
  fi
else
  printf '[benchmark20] 20 tasks x Lite/Full = 40 smoke executions; budget=64000\n'
fi
printf '[benchmark20] TP=4 BF16; context=65536; concurrency=%s; preflight_concurrency=%s\n' "$CONCURRENCY" "$PREFLIGHT_CONCURRENCY"
if [[ "${DRY_RUN:-1}" != 0 && "$MODE" != --preflight-only ]]; then
  # The test plan uses a provisional budget; execution selects it from dev.
  "${runner[@]}" --split "$SPLIT" --token-budget "${FIXED_BUDGET:-64000}" "${fixed_args[@]}"
  exit 0
fi

RUN_ROOT="$REPO_DIR/results/Qwen3.8-27B/benchmark-v2-$RUN_KIND-runs"
mkdir -p "$RUN_ROOT"
if [[ -n "$RESUME_RUN" ]]; then
  RUN_DIR="$(realpath -e -- "$RESUME_RUN")"
  [[ -d "$RUN_DIR" ]] || { echo 'Resume directory does not exist' >&2; exit 2; }
  mkdir -p "$RUN_DIR/resume-attempts"
  ATTEMPT_DIR="$(mktemp -d "$RUN_DIR/resume-attempts/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
else
  if [[ -n "$REQUESTED_RUN_DIR" ]]; then
    RUN_DIR="$(realpath -m -- "$REQUESTED_RUN_DIR")"
    mkdir -p "$(dirname -- "$RUN_DIR")"
    mkdir -- "$RUN_DIR" # Refuse to overwrite an existing run; use --resume-run.
  else
    RUN_DIR="$(mktemp -d "$RUN_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
  fi
  ATTEMPT_DIR="$RUN_DIR"
fi
export RUN_DIR
exec > >(tee "$ATTEMPT_DIR/launcher.log") 2>&1
printf '[benchmark20] output=%s\n' "$RUN_DIR"
printf '[benchmark20] attempt=%s\n' "$ATTEMPT_DIR"
printf '[benchmark20] concurrency=%s; preflight_concurrency=%s; inspect_preflight_timeout_seconds=%s\n' \
  "$CONCURRENCY" "$PREFLIGHT_CONCURRENCY" "$INSPECT_PREFLIGHT_TIMEOUT_SECONDS"
printf '[benchmark20] worker=%s; gpu_memory_utilization=%s; max_num_batched_tokens=%s\n' \
  "$VLLM_WORKER_CLS" "$VLLM_GPU_MEMORY_UTILIZATION" "$VLLM_MAX_NUM_BATCHED_TOKENS"

# Host dependencies, Frozen preparation, and data/bridge checks are one
# agent-preparation block. Serving-only retains its pure service-diagnostic
# semantics and skips the entire block.
if [[ "$MODE" != --serving-smoke-only ]]; then
  DRY_RUN=0 bash scripts/ensure_pool_smoke_tools.sh
  if [[ -n "${DCI_RUNTIME_CONFIG:-}" ]]; then
    export DCI_FROZEN_RUNTIME_PROBE_OUTPUT="$ATTEMPT_DIR/frozen-runtime-preflight.json"
    DRY_RUN=0 bash scripts/prepare_frozen_runtime.sh
  fi
  if [[ -n "$RESUME_RUN" ]]; then
    "$BENCH_PY" scripts/check_benchmark20_resume.py --run "$RUN_DIR" --suite "$SUITE" \
      --token-budget "$FIXED_BUDGET" --concurrency "$CONCURRENCY" "${runtime_args[@]}"
  fi
  # Check data and the actual node sandbox before loading any model weights.
  if [[ "$FULL" == 1 ]]; then
    # Dev and test contain different queries. Validate both before model load.
    preflight_splits=(dev test)
    if [[ -n "$FIXED_BUDGET" ]]; then preflight_splits=(test); fi
    for split in "${preflight_splits[@]}"; do
      "${runner[@]}" --split "$split" --token-budget "${FIXED_BUDGET:-64000}" "${fixed_args[@]}" --preflight-only \
        --preflight-concurrency "$PREFLIGHT_CONCURRENCY" --output "$ATTEMPT_DIR/data-preflight-$split"
    done
  else
    "${runner[@]}" --split smoke --token-budget 64000 --preflight-only \
      --preflight-concurrency "$PREFLIGHT_CONCURRENCY" --output "$ATTEMPT_DIR/data-preflight"
  fi
  "$BENCH_PY" - "$ATTEMPT_DIR" <<'PY'
from pathlib import Path
import sys
from scripts.inspect_local_runtime import prepare
prepare(Path(sys.argv[1]) / 'sandbox-tools')
PY
  # This generic transport probe has no pool metadata. Use its default profile;
  # run_benchmark_v2 selects Lite/Full with verified candidate IDs for real tasks.
  "$BENCH_PY" scripts/check_inspect_parallel.py \
    --runtime "$REPO_DIR" --tools-directory "$ATTEMPT_DIR/sandbox-tools" \
    --output "$ATTEMPT_DIR/sandbox-preflight" --max-concurrency "$CONCURRENCY" \
    --sample-timeout-seconds "$INSPECT_PREFLIGHT_TIMEOUT_SECONDS" "${runtime_args[@]}"
  if [[ "$MODE" == --preflight-only ]]; then
    echo '[benchmark20] CPU data and sandbox preflight passed; no GPU/model calls.'
    exit 0
  fi
fi

# Preserve the serving environment separately from the benchmark interpreter.
serving=(env PATH="$SERVING_ENV/bin:$PATH" LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu)
"${serving[@]}" "$SERVING_PY" scripts/harness_pilot_runtime.py --output "$ATTEMPT_DIR/runtime.json"
"$BENCH_PY" - "$ATTEMPT_DIR/runtime.json" "$VLLM_PORT" "$BRIDGE_PORT" "$CONCURRENCY" <<'PY'
import json, socket, sys
runtime = json.load(open(sys.argv[1]))
gpus = runtime['cuda']['gpus']
if len(gpus) != 4 or any('5090' not in g['name'] or g['capability'] != [12, 0] for g in gpus):
    raise SystemExit('This launcher requires exactly four visible RTX 5090 / sm120 GPUs')
ports = [int(sys.argv[2]), *range(int(sys.argv[3]), int(sys.argv[3]) + int(sys.argv[4]))]
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
  --output "$ATTEMPT_DIR/gdn-kernel-preflight.json"
VLLM_VERSION="$("${serving[@]}" "$SERVING_PY" -c 'from importlib.metadata import version; print(version("vllm"))')"
if [[ -n "$RESUME_RUN" && -f "$RUN_DIR/backend.json" ]]; then
  "$BENCH_PY" scripts/check_benchmark20_resume.py --run "$RUN_DIR" --suite "$SUITE" \
    --token-budget "$FIXED_BUDGET" --concurrency "$CONCURRENCY" "${runtime_args[@]}" \
    --serving-version "$VLLM_VERSION"
else
  DCI_PROTOCOL_PROFILE=document-pool-v2-lite "$BENCH_PY" -m dci_bench.backends.manifest \
    --output "$RUN_DIR/backend.json" --model-key Qwen3.8-27B --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" --base-url "http://127.0.0.1:$VLLM_PORT/v1" \
    --vllm-version "$VLLM_VERSION" --tool-call-parser "$TOOL_CALL_PARSER" \
    --reasoning-parser "$REASONING_PARSER" --language-model-only --sampling-backend auto \
    --tensor-parallel-size 4 --context-length 65536 --dtype bfloat16 \
    --max-num-seqs "$CONCURRENCY" --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --runtime-info "$ATTEMPT_DIR/runtime.json"
fi

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
setsid "${serving[@]}" bash scripts/serve_vllm_qwen38.sh > "$ATTEMPT_DIR/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for ((attempt=0; attempt<360; attempt++)); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo 'vLLM exited; inspect server.log'; exit 1; }
  if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$VLLM_PORT/v1/models" \
       -o "$ATTEMPT_DIR/models.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo 'vLLM startup timed out; inspect server.log'; exit 1; }
"$BENCH_PY" - "$ATTEMPT_DIR/models.json" <<'PY'
import json, sys
assert 'Qwen3.8-27B' in [row['id'] for row in json.load(open(sys.argv[1]))['data']]
PY
DCI_PROTOCOL_PROFILE=document-pool-v2-lite "$BENCH_PY" scripts/harness_pilot_warmup.py \
  --base-url "http://127.0.0.1:$VLLM_PORT/v1" --model "$SERVED_MODEL_NAME" \
  --output "$ATTEMPT_DIR/warmup.json" --concurrency "$CONCURRENCY" --deadline-seconds 600 --http-timeout-seconds 600 \
  --backend-manifest "$RUN_DIR/backend.json"
DCI_PROTOCOL_PROFILE=document-pool-v2-lite "$BENCH_PY" scripts/check_budget_tokenizer.py \
  --base-url "$DCI_TOKEN_COUNT_BASE_URL" --model "$SERVED_MODEL_NAME" \
  --output "$ATTEMPT_DIR/tokenizer-preflight.json"
if [[ "$MODE" == --serving-smoke-only ]]; then
  echo '[benchmark20] Serving smoke passed: model loaded and synthetic protocol requests passed; benchmark not executed.'
  exit 0
fi
if [[ "$FULL" == 1 && -n "$FIXED_BUDGET" ]]; then
  resume_args=()
  if [[ -n "$RESUME_RUN" && -d "$RUN_DIR/test" ]]; then resume_args=(--resume); fi
  "${runner[@]}" --split test --token-budget "$FIXED_BUDGET" "${fixed_args[@]}" \
    --backend-manifest "$RUN_DIR/backend.json" --execute --output "$RUN_DIR/test" "${resume_args[@]}"
elif [[ "$FULL" == 1 ]]; then
  for budget in 32000 64000 96000; do
    printf '[benchmark20] Starting dev calibration: budget=%s\n' "$budget"
    "${runner[@]}" --split dev --token-budget "$budget" --backend-manifest "$RUN_DIR/backend.json" \
      --execute --output "$RUN_DIR/dev-$budget"
  done
  "$BENCH_PY" scripts/analyze_benchmark_v2.py select-budget \
    --reports "$RUN_DIR/dev-32000/summary.json" "$RUN_DIR/dev-64000/summary.json" "$RUN_DIR/dev-96000/summary.json" \
    --output "$RUN_DIR/budget-selection.json"
  budget="$("$BENCH_PY" - "$RUN_DIR/budget-selection.json" <<'PY'
import json, sys
selected = json.load(open(sys.argv[1]))['selected_budget']
if selected not in (32000, 64000, 96000):
    raise SystemExit('No dev budget passed the selection gates; formal evaluation was not started')
print(selected)
PY
)"
  printf '[benchmark20] Starting full test: selected budget=%s\n' "$budget"
  "${runner[@]}" --split test --token-budget "$budget" --backend-manifest "$RUN_DIR/backend.json" \
    --calibration "$RUN_DIR/dev-$budget/summary.json" --budget-selection "$RUN_DIR/budget-selection.json" \
    --execute --output "$RUN_DIR/test"
else
  "${runner[@]}" --split smoke --token-budget 64000 --backend-manifest "$RUN_DIR/backend.json" \
    --execute --output "$RUN_DIR/smoke"
fi
printf '[benchmark20] Finished; inspect %s/%s/summary.json for completed/valid counts.\n' "$RUN_DIR" "$SPLIT"
