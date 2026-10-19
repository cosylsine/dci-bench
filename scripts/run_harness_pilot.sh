#!/usr/bin/env bash
# One serving replica, sequential paired profiles, 60 fixed queries (10 per task).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_PATH="${MODEL_PATH:-/mnt/afs/share/Qwen3.8-27B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
export TENSOR_PARALLEL_SIZE="${TP_SIZE:-1}"
export MAX_MODEL_LEN=65536
export DTYPE=bfloat16
export VLLM_PORT="${VLLM_PORT:-28099}"
export VLLM_HOST=127.0.0.1
export TOOL_CALL_PARSER=qwen3_coder
export REASONING_PARSER=qwen3
export LANGUAGE_MODEL_ONLY=1
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
export WARMUP_DEADLINE_SECONDS="${WARMUP_DEADLINE_SECONDS:-600}"
export WARMUP_HTTP_TIMEOUT_SECONDS="${WARMUP_HTTP_TIMEOUT_SECONDS:-600}"
# Resolve the forwarded CLI option before configuring the serving replica.
pilot_args=("$@")
preflight_profiles=(v3)
for ((i=0; i<${#pilot_args[@]}; i+=1)); do
  case "${pilot_args[i]}" in
    --max-concurrency)
      if (( i + 1 >= ${#pilot_args[@]} )); then
        echo "--max-concurrency requires a positive integer" >&2
        exit 2
      fi
      MAX_CONCURRENCY="${pilot_args[i+1]}"
      i=$((i + 1))
      ;;
    --max-concurrency=*) MAX_CONCURRENCY="${pilot_args[i]#*=}" ;;
    --profiles)
      preflight_profiles=()
      while (( i + 1 < ${#pilot_args[@]} )) && [[ "${pilot_args[i+1]}" != --* ]]; do
        i=$((i + 1))
        preflight_profiles+=("${pilot_args[i]}")
      done
      if (( ${#preflight_profiles[@]} == 0 )); then
        echo "--profiles requires at least one profile" >&2
        exit 2
      fi
      ;;
    --profiles=*) preflight_profiles=("${pilot_args[i]#*=}") ;;
  esac
done
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$MAX_CONCURRENCY}"
if [[ ! "$MAX_CONCURRENCY" =~ ^[1-9][0-9]*$ || ! "$VLLM_MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENCY and VLLM_MAX_NUM_SEQS must be positive integers" >&2
  exit 2
fi
if (( VLLM_MAX_NUM_SEQS < MAX_CONCURRENCY )); then
  echo "VLLM_MAX_NUM_SEQS must be at least MAX_CONCURRENCY" >&2
  exit 2
fi
if [[ ! "$WARMUP_DEADLINE_SECONDS" =~ ^([1-9][0-9]*(\.[0-9]+)?|0\.[0-9]+)$ ||
      ! "$WARMUP_HTTP_TIMEOUT_SECONDS" =~ ^([1-9][0-9]*(\.[0-9]+)?|0\.[0-9]+)$ ]]; then
  echo "WARMUP_DEADLINE_SECONDS and WARMUP_HTTP_TIMEOUT_SECONDS must be positive numbers" >&2
  exit 2
fi
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enforce-eager}"
BENCH_PYTHON="${BENCH_PYTHON:-/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python}"
SERVING_PYTHON="${SERVING_PYTHON:-/mnt/afs/250010100/miniconda/envs/dci-vllm-qwen38/bin/python}"
export PATH="$(dirname "$SERVING_PYTHON"):$PATH"
export DCI_PROTOCOL_PROFILE=v3
unset DCI_PROTOCOL_CONTRACT_SHA256
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
PILOT_ROOT="${PILOT_ROOT:-results/harness-pilots/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'model=%s served=%s TP=%s dtype=%s context=65536 total_budget=128000 concurrency=%s max_num_seqs=%s\n' "$MODEL_PATH" "$SERVED_MODEL_NAME" "$TENSOR_PARALLEL_SIZE" "$DTYPE" "$MAX_CONCURRENCY" "$VLLM_MAX_NUM_SEQS"
  printf 'serving_python=%s bench_python=%s output=%s\n' "$SERVING_PYTHON" "$BENCH_PYTHON" "$PILOT_ROOT"
  printf 'pilot arguments:'
  printf ' %q' "$@"
  printf '\n'
  exit 0
fi
if [[ -e "$PILOT_ROOT" ]]; then
  echo "Refusing to reuse existing pilot path: $PILOT_ROOT" >&2
  exit 2
fi
PATH="$(dirname "$BENCH_PYTHON"):$PATH" "$BENCH_PYTHON" scripts/run_harness_pilot.py --preflight-only "$@"
# Match the full-run launchers: container restarts can discard OS packages.
missing_os=()
[[ -x /usr/bin/bwrap ]] || missing_os+=(bubblewrap)
[[ -x /usr/bin/rg ]] || missing_os+=(ripgrep)
if (( ${#missing_os[@]} > 0 )); then
  if [[ "$(id -u)" != "0" ]] || ! command -v apt-get >/dev/null 2>&1; then
    echo "[pilot] Cannot install missing OS packages: ${missing_os[*]}; node image must provide /usr/bin/bwrap and /usr/bin/rg" >&2
    exit 2
  fi
  echo "[pilot] Installing missing OS packages: ${missing_os[*]}"
  if ! apt-get update -o Acquire::Retries=3 ||
     ! DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing_os[@]}"; then
    echo "[pilot] OS package installation failed" >&2
    exit 2
  fi
fi
if [[ ! -x /usr/bin/bwrap || ! -x /usr/bin/rg ]]; then
  echo "[pilot] Trusted /usr/bin/bwrap and /usr/bin/rg are required" >&2
  exit 2
fi
for binary in node npm setsid curl; do
  if ! PATH="$(dirname "$BENCH_PYTHON"):$PATH" command -v "$binary" >/dev/null 2>&1; then
    echo "[pilot] Required command is unavailable: $binary" >&2
    exit 2
  fi
done
# Exercise the real sandbox before probing CUDA or loading model weights.
echo "[pilot] Data and Pi sandbox preflight"
if ! PATH="$(dirname "$BENCH_PYTHON"):$PATH" "$BENCH_PYTHON" scripts/run_all.py --preflight-only; then
  exit 2
fi
for profile in "${preflight_profiles[@]}"; do
  if ! (
    cd pi-dci
    DCI_PROTOCOL_PROFILE="$profile" PATH="$(dirname "$BENCH_PYTHON"):$PATH" npm exec -- tsx scripts/dci-run.ts \
      --preflight-only --workspace "$PWD/../data/workspaces/LLMPublicHealthQA"
  ); then
    echo "[pilot] Pi sandbox preflight failed for $profile; model startup cancelled" >&2
    exit 2
  fi
done
mkdir -p "$PILOT_ROOT"
PILOT_ROOT="$(cd "$PILOT_ROOT" && pwd)"
SERVICE_PID=""
cleanup() {
  if [[ -n "$SERVICE_PID" ]]; then
    kill -TERM -- "-$SERVICE_PID" 2>/dev/null || true
    wait "$SERVICE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$SERVING_PYTHON" scripts/harness_pilot_runtime.py --output "$PILOT_ROOT/runtime.json"
if curl --noproxy '*' -fsS "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $VLLM_PORT already serves a model; choose a fresh VLLM_PORT" >&2
  exit 2
fi
setsid bash scripts/serve_vllm_qwen38.sh >"$PILOT_ROOT/server.log" 2>&1 &
SERVICE_PID=$!
ready=0
for ((attempt=0; attempt<360; attempt+=1)); do
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "Serving failed; inspect $PILOT_ROOT/server.log" >&2
    exit 2
  fi
  if curl --noproxy '*' -fsS "http://127.0.0.1:$VLLM_PORT/v1/models" >"$PILOT_ROOT/models.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" != "1" ]]; then
  echo "Serving readiness timed out; inspect $PILOT_ROOT/server.log" >&2
  exit 2
fi
VLLM_VERSION="$("$SERVING_PYTHON" -c 'from importlib.metadata import version; print(version("vllm"))')"
"$BENCH_PYTHON" -m dci_bench.backends.manifest \
  --output "$PILOT_ROOT/backend.json" --model-key "$SERVED_MODEL_NAME" \
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
  --base-url "http://127.0.0.1:$VLLM_PORT/v1" --vllm-version "$VLLM_VERSION" \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only \
  --sampling-backend auto --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --context-length 65536 --dtype "$DTYPE" --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" --runtime-info "$PILOT_ROOT/runtime.json"
echo "[pilot] synthetic serving warmup: concurrency=$MAX_CONCURRENCY deadline=${WARMUP_DEADLINE_SECONDS}s"
PATH="$(dirname "$BENCH_PYTHON"):$PATH" "$BENCH_PYTHON" scripts/harness_pilot_warmup.py \
  --base-url "http://127.0.0.1:$VLLM_PORT/v1" --model "$SERVED_MODEL_NAME" \
  --output "$PILOT_ROOT/warmup.json" --concurrency "$MAX_CONCURRENCY" \
  --deadline-seconds "$WARMUP_DEADLINE_SECONDS" --http-timeout-seconds "$WARMUP_HTTP_TIMEOUT_SECONDS"
PATH="$(dirname "$BENCH_PYTHON"):$PATH" "$BENCH_PYTHON" scripts/run_harness_pilot.py \
  --backend-manifest "$PILOT_ROOT/backend.json" --output-root "$PILOT_ROOT/evaluation" \
  --backend-artifact "$PILOT_ROOT/runtime.json" --backend-artifact "$PILOT_ROOT/models.json" \
  --backend-artifact "$PILOT_ROOT/warmup.json" "$@"
