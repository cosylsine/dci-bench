#!/usr/bin/env bash
# Run a 4-GPU vLLM replica for a four-node batch or one selected task.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NODE_RANK="${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-}}}}"
WORLD_SIZE="${WORLD_SIZE:-${SENSECORE_PYTORCH_NNODES:-${SLURM_NNODES:-4}}}"
ONLY_TASK="${ONLY_TASK:-}"
if [[ "${WORLD_SIZE}" != "4" && "${WORLD_SIZE}" != "1" ]] ||
   { [[ "${WORLD_SIZE}" == "4" ]] && [[ ! "${NODE_RANK}" =~ ^[0-3]$ ]]; } ||
   { [[ "${WORLD_SIZE}" == "1" ]] && [[ "${NODE_RANK}" != "0" ]]; }; then
  echo "[DCI] use WORLD_SIZE=4 with NODE_RANK=0,1,2,3 or WORLD_SIZE=1 with NODE_RANK=0 and ONLY_TASK" >&2
  exit 2
fi

if [[ "${WORLD_SIZE}" == "1" ]]; then
  case "${ONLY_TASK}" in
    LLMAILAStatutes|LLMFQuADRetrieval|LLMHC3FinanceRetrieval|LLMLegalBenchConsumerContractsQA|LLMPublicHealthQA|LLMTwitterHjerneRetrieval)
      TASKS=("${ONLY_TASK}") ;;
    *) echo "[DCI] WORLD_SIZE=1 requires ONLY_TASK to name one DCI task" >&2; exit 2 ;;
  esac
else
  # Keep one model replica per node. The assignments cover all six tasks.
  case "${NODE_RANK}" in
    0) TASKS=(LLMAILAStatutes LLMFQuADRetrieval) ;;
    1) TASKS=(LLMHC3FinanceRetrieval LLMTwitterHjerneRetrieval) ;;
    2) TASKS=(LLMLegalBenchConsumerContractsQA) ;;
    3) TASKS=(LLMPublicHealthQA) ;;
  esac
  if [[ -n "${ONLY_TASK}" ]]; then
    echo "[DCI] ONLY_TASK is supported with WORLD_SIZE=1; four-node task assignments are fixed" >&2
    exit 2
  fi
fi

JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-${LSB_JOBID:-${PBS_JOBID:-manual}}}}"
JOB_TAG="${JOB_TAG//\//_}"
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_LOG="${RUN_LOG:-${REPO_ROOT}/results/job-logs/qwen38-vllm-${JOB_TAG}-rank${NODE_RANK}-${RUN_TIMESTAMP}.log}"
SERVICE_LOG="${SERVICE_LOG:-${REPO_ROOT}/results/job-logs/qwen38-vllm-service-${JOB_TAG}-rank${NODE_RANK}-${RUN_TIMESTAMP}.log}"
DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-${REPO_ROOT}/results/job-artifacts/qwen38-vllm-${JOB_TAG}-rank${NODE_RANK}-${RUN_TIMESTAMP}}"
mkdir -p "$(dirname "${RUN_LOG}")" "$(dirname "${SERVICE_LOG}")" "${DIAGNOSTIC_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

log() { printf '[DCI][%s][rank=%s] %s\n' "$(date --iso-8601=seconds)" "${NODE_RANK}" "$*"; }
VLLM_PID=""
VLLM_PGID=""
cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "stopping vLLM process group ${VLLM_PGID}"
    if [[ "${VLLM_PGID}" =~ ^[1-9][0-9]*$ ]]; then
      kill -TERM -- "-${VLLM_PGID}" 2>/dev/null || true
    else
      kill -TERM "${VLLM_PID}" 2>/dev/null || true
    fi
    (
      sleep 30
      if kill -0 "${VLLM_PID}" 2>/dev/null; then
        if [[ "${VLLM_PGID}" =~ ^[1-9][0-9]*$ ]]; then
          kill -KILL -- "-${VLLM_PGID}" 2>/dev/null || true
        else
          kill -KILL "${VLLM_PID}" 2>/dev/null || true
        fi
      fi
    ) &
    local watchdog=$!
    wait "${VLLM_PID}" 2>/dev/null || true
    kill "${watchdog}" 2>/dev/null || true
    wait "${watchdog}" 2>/dev/null || true
  fi
  log "finished exit=${exit_code}; run_log=${RUN_LOG}; service_log=${SERVICE_LOG}; diagnostics=${DIAGNOSTIC_DIR}"
  exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CONDA_SH="${CONDA_SH:-/mnt/afs/250010100/miniconda/etc/profile.d/conda.sh}"
VLLM_ENV="${VLLM_ENV:-dci-vllm-qwen38}"
BENCH_ENV="${BENCH_ENV:-dci-bench}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs/share/Qwen3.8-27B}"
MODEL_KEY="${MODEL_KEY:-Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
VLLM_HOST="127.0.0.1"
VLLM_PORT="${VLLM_PORT:-17892}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/data/workspaces}"
METADATA_ROOT="${METADATA_ROOT:-${REPO_ROOT}/data/metadata}"
BACKEND_MANIFEST_PATH="${BACKEND_MANIFEST_PATH:-${REPO_ROOT}/results/backend-manifests/vllm-${MODEL_KEY}-${JOB_TAG}-rank${NODE_RANK}-${RUN_TIMESTAMP}.json}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
BRIDGE_PORT="${BRIDGE_PORT:-18891}"
METRIC_KS="${METRIC_KS:-1 3 5 10 20}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-360}"
WAIT_SECONDS="${WAIT_SECONDS:-5}"
TOOL_SMOKE_TIMEOUT_SECONDS="${TOOL_SMOKE_TIMEOUT_SECONDS:-600}"
FAIL_JOB_ON_SAMPLE_FAILURES="${FAIL_JOB_ON_SAMPLE_FAILURES:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-}"

for name in VLLM_PORT TENSOR_PARALLEL_SIZE MAX_MODEL_LEN MAX_CONCURRENCY VLLM_MAX_NUM_SEQS BRIDGE_PORT WAIT_ATTEMPTS WAIT_SECONDS TOOL_SMOKE_TIMEOUT_SECONDS; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    log "${name} must be a positive integer"
    exit 2
  fi
done
if (( VLLM_MAX_NUM_SEQS < MAX_CONCURRENCY )); then
  log "VLLM_MAX_NUM_SEQS must be at least MAX_CONCURRENCY"
  exit 2
fi
if ! python - "${VLLM_GPU_MEMORY_UTILIZATION}" <<'PY'
import math
import sys

try:
    fraction = float(sys.argv[1])
except ValueError:
    raise SystemExit(2)
if not math.isfinite(fraction) or not 0 < fraction < 1:
    raise SystemExit(2)
PY
then
  log "VLLM_GPU_MEMORY_UTILIZATION must be a number between 0 and 1"
  exit 2
fi
if [[ "${TENSOR_PARALLEL_SIZE}" != "4" || "${LANGUAGE_MODEL_ONLY}" != "1" ]]; then
  log "this 4x5090 text-only job requires TENSOR_PARALLEL_SIZE=4 and LANGUAGE_MODEL_ONLY=1"
  exit 2
fi
if [[ -n "${SENSECORE_ACCELERATE_DEVICE_COUNT:-}" && "${SENSECORE_ACCELERATE_DEVICE_COUNT}" != "4" ]]; then
  log "SENSECORE_ACCELERATE_DEVICE_COUNT must be 4, got ${SENSECORE_ACCELERATE_DEVICE_COUNT}"
  exit 2
fi
if (( BRIDGE_PORT + 6 * MAX_CONCURRENCY - 1 > 65535 )); then
  log "bridge port ranges exceed TCP port 65535"
  exit 2
fi
log "host=${HOSTNAME:-unknown} tasks=${TASKS[*]} model=${MODEL_PATH} tp=4 concurrency=${MAX_CONCURRENCY}"
log "vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS} vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN}"
log "results_root=${RESULTS_ROOT} workspace_root=${WORKSPACE_ROOT} metadata_root=${METADATA_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run complete; service and evaluation were not started"
  exit 0
fi

if [[ ! -r "${CONDA_SH}" ]]; then
  log "missing Conda initialization script: ${CONDA_SH}"
  exit 2
fi
missing_os=()
[[ -x /usr/bin/bwrap ]] || missing_os+=(bubblewrap)
[[ -x /usr/bin/rg ]] || missing_os+=(ripgrep)
if (( ${#missing_os[@]} > 0 )); then
  if [[ "$(id -u)" != "0" ]] || ! command -v apt-get >/dev/null 2>&1; then
    log "node image must provide /usr/bin/bwrap and /usr/bin/rg; missing: ${missing_os[*]}"
    exit 2
  fi
  apt-get update -o Acquire::Retries=3
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing_os[@]}"
fi
if [[ ! -x /usr/bin/bwrap || ! -x /usr/bin/rg ]]; then
  log "trusted /usr/bin/bwrap and /usr/bin/rg are required"
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  log "setsid and curl are required"
  exit 2
fi

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export MODEL_PATH SERVED_MODEL_NAME VLLM_HOST VLLM_PORT TENSOR_PARALLEL_SIZE MAX_MODEL_LEN
export TOOL_CALL_PARSER REASONING_PARSER LANGUAGE_MODEL_ONLY VLLM_EXTRA_ARGS
export VLLM_MAX_NUM_SEQS VLLM_GPU_MEMORY_UTILIZATION

log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L
set +u
source "${CONDA_SH}"
conda activate "${BENCH_ENV}"
set -u
log "data preflight"
python scripts/run_all.py --preflight-only --workspace-root "${WORKSPACE_ROOT}" --metadata-root "${METADATA_ROOT}"
(
  cd pi-dci
  npm exec -- tsx scripts/dci-run.ts --preflight-only --workspace "${WORKSPACE_ROOT}/${TASKS[0]}"
)
python - "${VLLM_HOST}" "${VLLM_PORT}" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    try:
        listener.bind((sys.argv[1], int(sys.argv[2])))
    except OSError as exc:
        raise SystemExit(f"[DCI] vLLM port is already occupied: {exc}")
PY

set +u
conda activate "${VLLM_ENV}"
set -u
log "5090 and Qwen3.5 serving preflight"
python - <<'PY'
import json
import os
from importlib.metadata import version
from pathlib import Path
import torch

config = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
assert config.get("model_type") == "qwen3_5", "unexpected model architecture"
assert list(Path(os.environ["MODEL_PATH"]).glob("*.safetensors")), "model weights missing"
assert tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8), "PyTorch CUDA build is older than 12.8"
assert "sm_120" in torch._C._cuda_getArchFlags().split(), "PyTorch wheel has no sm_120 support"
assert torch.cuda.is_available() and torch.cuda.device_count() == 4, "four usable CUDA devices are required"
for index in range(4):
    props = torch.cuda.get_device_properties(index)
    assert "5090" in props.name and (props.major, props.minor) == (12, 0), f"device {index} is not an RTX 5090: {props.name}"
    print(f"[DCI] GPU {index}: {props.name}, {props.total_memory / 1024**3:.1f} GiB", flush=True)
torch.ones(1, device="cuda:0").item()
vllm_version = version("vllm")
assert tuple(int(x) for x in vllm_version.split(".")[:2]) >= (0, 17), "vLLM 0.17+ required"
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config  # noqa: F401
print(f"[DCI] torch={torch.__version__} CUDA={torch.version.cuda} vllm={vllm_version}", flush=True)
PY
VLLM_VERSION="$(python -c 'from importlib.metadata import version; print(version("vllm"))')"
log "vLLM version=${VLLM_VERSION}; starting service on ${VLLM_HOST}:${VLLM_PORT}"

setsid scripts/serve_vllm_qwen38.sh >"${SERVICE_LOG}" 2>&1 &
VLLM_PID=$!
sleep 1
VLLM_PGID="$(ps -o pgid= -p "${VLLM_PID}" 2>/dev/null | tr -d '[:space:]' || true)"
OWN_PGID="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
if [[ ! "${VLLM_PGID}" =~ ^[1-9][0-9]*$ || "${VLLM_PGID}" == "${OWN_PGID}" ]]; then
  log "vLLM process group was not isolated"
  exit 2
fi

MODELS_RESPONSE_PATH="${DIAGNOSTIC_DIR}/models.json"
TOOL_SMOKE_REQUEST_PATH="${DIAGNOSTIC_DIR}/tool-parser-smoke-request.json"
TOOL_SMOKE_RESPONSE_PATH="${DIAGNOSTIC_DIR}/tool-parser-smoke-response.json"
MODELS_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${VLLM_PORT}/v1/chat/completions"
host_curl() {
  local max_time="$1"
  shift
  env -u LD_LIBRARY_PATH curl --noproxy 127.0.0.1,localhost --connect-timeout 5 --max-time "${max_time}" "$@"
}
ready=0
for (( attempt=1; attempt<=WAIT_ATTEMPTS; attempt++ )); do
  if host_curl 30 -fsS "${MODELS_URL}" -o "${MODELS_RESPONSE_PATH}" 2>/dev/null; then
    if python - "${MODELS_RESPONSE_PATH}" "${SERVED_MODEL_NAME}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert sys.argv[2] in [model.get("id") for model in payload.get("data", [])]
PY
    then
      ready=1
      break
    fi
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "vLLM exited before readiness"
    tail -n 100 "${SERVICE_LOG}" || true
    exit 2
  fi
  if (( attempt == 1 || attempt % 12 == 0 )); then
    log "waiting for vLLM: attempt=${attempt}/${WAIT_ATTEMPTS}"
  fi
  sleep "${WAIT_SECONDS}"
done
if [[ "${ready}" != "1" ]]; then
  log "vLLM readiness timed out"
  tail -n 100 "${SERVICE_LOG}" || true
  exit 2
fi
log "vLLM ready"

set +u
conda activate "${BENCH_ENV}"
set -u
python - "${SERVED_MODEL_NAME}" "${TOOL_SMOKE_REQUEST_PATH}" <<'PY'
import json, sys
from pathlib import Path
model, path = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Call get_weather for Beijing now. Do not answer in prose."}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "tool_choice": "auto", "parallel_tool_calls": False,
    "temperature": 0.2, "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": False},
}
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
log "checking first vLLM chat completion and tool parser; timeout=${TOOL_SMOKE_TIMEOUT_SECONDS}s"
smoke_started=$SECONDS
if host_curl "${TOOL_SMOKE_TIMEOUT_SECONDS}" -fsS -H "Content-Type: application/json" -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  --data-binary "@${TOOL_SMOKE_REQUEST_PATH}" "${CHAT_URL}" -o "${TOOL_SMOKE_RESPONSE_PATH}"; then
  log "first chat completion returned after $((SECONDS - smoke_started))s"
else
  smoke_exit=$?
  log "first chat completion failed: curl_exit=${smoke_exit} elapsed=$((SECONDS - smoke_started))s; response=${TOOL_SMOKE_RESPONSE_PATH}"
  host_curl 10 -fsS "http://127.0.0.1:${VLLM_PORT}/metrics" -o "${DIAGNOSTIC_DIR}/vllm-metrics.txt" || true
  nvidia-smi >"${DIAGNOSTIC_DIR}/nvidia-smi.txt" 2>&1 || true
  log "saved failure diagnostics in ${DIAGNOSTIC_DIR}"
  tail -n 80 "${SERVICE_LOG}" || true
  exit "${smoke_exit}"
fi
python - "${TOOL_SMOKE_RESPONSE_PATH}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
choices = payload.get("choices") or []
assert choices and choices[0].get("finish_reason") == "tool_calls", "tool call was not produced"
calls = choices[0].get("message", {}).get("tool_calls") or []
assert calls and calls[0].get("function", {}).get("name") == "get_weather", "unexpected tool call"
args = calls[0]["function"].get("arguments")
assert isinstance(json.loads(args) if isinstance(args, str) else args, dict), "invalid tool arguments"
assert payload.get("usage", {}).get("total_tokens") is not None, "usage tokens missing"
print("[DCI] vLLM tool parser smoke passed", flush=True)
PY

mkdir -p "$(dirname "${BACKEND_MANIFEST_PATH}")"
if [[ -e "${BACKEND_MANIFEST_PATH}" ]]; then
  log "backend manifest already exists: ${BACKEND_MANIFEST_PATH}"
  exit 2
fi
python -m dci_bench.backends.manifest \
  --output "${BACKEND_MANIFEST_PATH}" --model-key "${MODEL_KEY}" \
  --model-path "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --vllm-version "${VLLM_VERSION}" --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser "${REASONING_PARSER}" --language-model-only \
  --sampling-backend auto --tensor-parallel-size 4 \
  --context-length "${MAX_MODEL_LEN}" --dtype bfloat16 \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"

read -r -a METRIC_K_ARRAY <<<"${METRIC_KS}"
RUNNER_ARGS=(
  python scripts/run_all.py --model-key "${MODEL_KEY}"
  --tasks "${TASKS[@]}" --backend-manifest "${BACKEND_MANIFEST_PATH}"
  --backend-artifact "${MODELS_RESPONSE_PATH}"
  --backend-artifact "${TOOL_SMOKE_REQUEST_PATH}"
  --backend-artifact "${TOOL_SMOKE_RESPONSE_PATH}"
  --results-root "${RESULTS_ROOT}" --workspace-root "${WORKSPACE_ROOT}"
  --metadata-root "${METADATA_ROOT}" --max-concurrency "${MAX_CONCURRENCY}"
  --bridge-port "${BRIDGE_PORT}" --api-key "${OPENAI_API_KEY}"
  --metric-ks "${METRIC_K_ARRAY[@]}"
)
if [[ -n "${RUN_ID}" ]]; then RUNNER_ARGS+=(--run-id "${RUN_ID}"); fi
log "evaluation begins: ${TASKS[*]}"
if "${RUNNER_ARGS[@]}"; then runner_exit=0; else runner_exit=$?; fi
job_exit="${runner_exit}"
if [[ "${runner_exit}" == "1" && "${FAIL_JOB_ON_SAMPLE_FAILURES}" == "0" ]]; then
  job_exit=0
fi
log "evaluation finished: runner_exit=${runner_exit} job_exit=${job_exit} tasks=${TASKS[*]}"
exit "${job_exit}"
