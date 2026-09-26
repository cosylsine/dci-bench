#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-${LSB_JOBID:-${PBS_JOBID:-manual-$$}}}}"
JOB_TAG="${JOB_TAG//\//_}"
RUN_LOG="${RUN_LOG:-${REPO_ROOT}/results/job-logs/phase5-sglang-${JOB_TAG}-${RUN_TIMESTAMP}.log}"
SERVICE_LOG="${SERVICE_LOG:-${REPO_ROOT}/results/job-logs/phase5-sglang-service-${JOB_TAG}-${RUN_TIMESTAMP}.log}"
DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-${REPO_ROOT}/results/job-artifacts/phase5-sglang-${JOB_TAG}-${RUN_TIMESTAMP}}"
mkdir -p "$(dirname "${RUN_LOG}")" "$(dirname "${SERVICE_LOG}")" "${DIAGNOSTIC_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

log() {
  printf '[DCI][%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

report_error() {
  local exit_code="$1"
  local line_number="$2"
  local failed_command="$3"
  log "ERROR exit=${exit_code} line=${line_number} command=${failed_command}"
}
trap 'report_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

SGLANG_PID=""
SGLANG_PGID=""
cleanup() {
  local exit_code=$?
  local attempt
  local stop_watchdog_pid=""
  trap - EXIT
  log "job exiting; exit=${exit_code} sglang_pid=${SGLANG_PID:-not-started}"
  if [[ -n "${SGLANG_PID}" ]] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
    if [[ "${SGLANG_PGID}" =~ ^[1-9][0-9]*$ ]]; then
      log "stopping SGLang process group pgid=${SGLANG_PGID}"
      kill -TERM -- "-${SGLANG_PGID}" 2>/dev/null || true
    else
      log "SGLang process group unavailable; stopping pid=${SGLANG_PID}"
      kill -TERM "${SGLANG_PID}" 2>/dev/null || true
    fi
    (
      for attempt in $(seq 1 "${SERVICE_STOP_WAIT_SECONDS:-30}"); do
        sleep 1
        if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
          exit 0
        fi
      done
      log "SGLang did not stop after ${SERVICE_STOP_WAIT_SECONDS:-30}s; sending SIGKILL"
      if [[ "${SGLANG_PGID}" =~ ^[1-9][0-9]*$ ]]; then
        kill -KILL -- "-${SGLANG_PGID}" 2>/dev/null || true
      else
        kill -KILL "${SGLANG_PID}" 2>/dev/null || true
      fi
    ) &
    stop_watchdog_pid=$!
  fi
  if [[ -n "${SGLANG_PID}" ]]; then
    # Reap the expected TERM/KILL while wait's stderr is redirected.  Polling
    # first leaves the dead background job pending and makes Bash print a
    # misleading "Killed ... serve_sglang_minicpm5.sh" notification.
    wait "${SGLANG_PID}" 2>/dev/null || true
  fi
  if [[ -n "${stop_watchdog_pid}" ]]; then
    wait "${stop_watchdog_pid}" 2>/dev/null || true
  fi
  log "artifacts: run_log=${RUN_LOG} service_log=${SERVICE_LOG} diagnostics=${DIAGNOSTIC_DIR}"
  exit "${exit_code}"
}
trap cleanup EXIT

handle_signal() {
  local signal_name="$1"
  local exit_code="$2"
  log "received ${signal_name}; stopping evaluation and SGLang"
  exit "${exit_code}"
}
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

log "Phase 5 SGLang job started; pid=$$ host=${HOSTNAME:-unknown} user=$(id -un) uid=$(id -u)"
log "repo_root=${REPO_ROOT}"
log "scheduler: SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
log "scheduler: JOB_ID=${JOB_ID:-unset} LSB_JOBID=${LSB_JOBID:-unset} PBS_JOBID=${PBS_JOBID:-unset}"
log "initial CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset}"

if [[ "${DEBUG_TRACE:-0}" == "1" ]]; then
  export PS4='+ [${BASH_SOURCE}:${LINENO}] '
  set -x
fi

CONDA_SH="${CONDA_SH:-/mnt/afs/250010100/miniconda/etc/profile.d/conda.sh}"
BENCH_ENV="${BENCH_ENV:-dci-bench}"
SGLANG_ENV="${SGLANG_ENV:-dci-sglang-minicpm5}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs2/202608/embedding_models/dci-bench/pretrained_models/MiniCPM5-2B}"
MODEL_KEY="${MODEL_KEY:-MiniCPM5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniCPM5-2B}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
# Keep the one-job Phase 5 default away from SGLang's commonly used 30000
# and the usual Linux ephemeral port range (32768-60999). Callers can still
# override it when multiple jobs share a network namespace.
SGLANG_PORT="${SGLANG_PORT:-17891}"
BACKEND_BASE_URL="${BACKEND_BASE_URL:-http://127.0.0.1:${SGLANG_PORT}/v1}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
DTYPE="${DTYPE:-bfloat16}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-minicpm5}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
SGLANG_JIT_CXX="${SGLANG_JIT_CXX:-}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
BACKEND_MANIFEST_PATH="${BACKEND_MANIFEST_PATH:-${REPO_ROOT}/results/backend-manifests/sglang-phase5-${MODEL_KEY}-${JOB_TAG}-${RUN_TIMESTAMP}.json}"
REUSE_BACKEND_MANIFEST="${REUSE_BACKEND_MANIFEST:-0}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-data/workspaces}"
METADATA_ROOT="${METADATA_ROOT:-data/metadata}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
BRIDGE_PORT="${BRIDGE_PORT:-18891}"
METRIC_KS="${METRIC_KS:-1 3 5 10 20}"
RUN_ID="${RUN_ID:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-360}"
WAIT_SECONDS="${WAIT_SECONDS:-5}"
SERVICE_STOP_WAIT_SECONDS="${SERVICE_STOP_WAIT_SECONDS:-30}"
PREPARE_DATA="${PREPARE_DATA:-1}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_JOB_ON_SAMPLE_FAILURES="${FAIL_JOB_ON_SAMPLE_FAILURES:-0}"

for numeric_name in TP_SIZE CONTEXT_LENGTH SGLANG_PORT MAX_CONCURRENCY BRIDGE_PORT WAIT_ATTEMPTS WAIT_SECONDS SERVICE_STOP_WAIT_SECONDS; do
  numeric_value="${!numeric_name}"
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    log "${numeric_name} must be a positive integer, got ${numeric_value@Q}"
    exit 2
  fi
done
for boolean_name in PREPARE_DATA DRY_RUN REUSE_BACKEND_MANIFEST FAIL_JOB_ON_SAMPLE_FAILURES; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    log "${boolean_name} must be 0 or 1, got ${boolean_value@Q}"
    exit 2
  fi
done
if (( BRIDGE_PORT + 6 * MAX_CONCURRENCY - 1 > 65535 )); then
  log "six task bridge port pools exceed TCP port range: base=${BRIDGE_PORT} concurrency=${MAX_CONCURRENCY}"
  exit 2
fi
if [[ ! -r "${CONDA_SH}" ]]; then
  log "missing CONDA_SH=${CONDA_SH}"
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  log "setsid is required to isolate and clean up the SGLang process group"
  exit 2
fi

MISSING_OS_PACKAGES=()
if [[ -x /usr/bin/bwrap ]]; then
  log "bubblewrap already available: /usr/bin/bwrap"
else
  MISSING_OS_PACKAGES+=(bubblewrap)
fi
if [[ -x /usr/bin/rg ]]; then
  log "ripgrep already available: /usr/bin/rg"
else
  MISSING_OS_PACKAGES+=(ripgrep)
fi
if (( ${#MISSING_OS_PACKAGES[@]} > 0 )); then
  log "missing OS packages: ${MISSING_OS_PACKAGES[*]}; preparing installation"
  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get is unavailable; use a cluster image that provides /usr/bin/bwrap and /usr/bin/rg"
    exit 2
  fi
  log "stage=apt-update begin"
  apt-get update -o Acquire::Retries=3
  log "stage=apt-update ok"
  apt-cache policy "${MISSING_OS_PACKAGES[@]}" || true
  log "stage=os-package-install begin; packages=${MISSING_OS_PACKAGES[*]}"
  DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends "${MISSING_OS_PACKAGES[@]}"
  log "stage=os-package-install ok"
fi
if [[ "$(command -v bwrap 2>/dev/null || true)" != "/usr/bin/bwrap" ]]; then
  log "trusted Bubblewrap must be available at /usr/bin/bwrap"
  exit 2
fi
if [[ "$(command -v rg 2>/dev/null || true)" != "/usr/bin/rg" ]]; then
  log "trusted ripgrep must be available at /usr/bin/rg"
  exit 2
fi

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

log "runtime config: model=${MODEL_PATH} model_key=${MODEL_KEY} served_name=${SERVED_MODEL_NAME}"
log "runtime config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} TP_SIZE=${TP_SIZE} context_length=${CONTEXT_LENGTH} dtype=${DTYPE}"
log "runtime config: host=${SGLANG_HOST} port=${SGLANG_PORT} backend_base_url=${BACKEND_BASE_URL}"
log "runtime config: max_concurrency=${MAX_CONCURRENCY} bridge_port=${BRIDGE_PORT} results_root=${RESULTS_ROOT}"
log "runtime config: fail_job_on_sample_failures=${FAIL_JOB_ON_SAMPLE_FAILURES}"
log "runtime config: backend_manifest=${BACKEND_MANIFEST_PATH} run_log=${RUN_LOG} service_log=${SERVICE_LOG}"
if command -v nvidia-smi >/dev/null 2>&1; then
  log "stage=gpu-visible nvidia-smi -L"
  nvidia-smi -L || log "WARNING: nvidia-smi -L failed; SGLang preflight will enforce usable GPUs"
fi

log "stage=bench-env begin; activating ${BENCH_ENV}"
set +u
source "${CONDA_SH}"
conda activate "${BENCH_ENV}"
set -u
log "stage=bench-env ok; python=$(command -v python) node=$(command -v node || printf 'not-found') npm=$(command -v npm || printf 'not-found')"
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  log "Node/npm are required in BENCH_ENV=${BENCH_ENV}"
  exit 2
fi

if [[ "${PREPARE_DATA}" == "1" ]]; then
  log "stage=data-prepare begin"
  scripts/prepare_data.sh
  log "stage=data-prepare ok"
fi
log "stage=phase5-data-preflight begin"
python scripts/run_all.py \
  --preflight-only \
  --workspace-root "${WORKSPACE_ROOT}" \
  --metadata-root "${METADATA_ROOT}"
log "stage=phase5-data-preflight ok"

log "stage=pi-preflight begin"
if [[ "${WORKSPACE_ROOT}" == /* ]]; then
  PI_PREFLIGHT_WORKSPACE="${WORKSPACE_ROOT}/LLMPublicHealthQA"
else
  PI_PREFLIGHT_WORKSPACE="${REPO_ROOT}/${WORKSPACE_ROOT}/LLMPublicHealthQA"
fi
(
  cd pi-dci
  npm exec -- tsx scripts/dci-run.ts \
    --preflight-only \
    --workspace "${PI_PREFLIGHT_WORKSPACE}"
)
log "stage=pi-preflight ok"

python - "${SGLANG_HOST}" "${SGLANG_PORT}" "${BACKEND_BASE_URL}" <<'PY'
import socket
import sys
from urllib.parse import urlparse

host = sys.argv[1]
port = int(sys.argv[2])
endpoint = urlparse(sys.argv[3])
if endpoint.scheme not in {"http", "https"} or endpoint.hostname not in {"127.0.0.1", "localhost"}:
    raise SystemExit(
        "[DCI] the one-job launcher requires a loopback BACKEND_BASE_URL "
        "(http://127.0.0.1:<port>/v1)"
    )
if endpoint.port != port:
    raise SystemExit("[DCI] BACKEND_BASE_URL port must match SGLANG_PORT")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    try:
        listener.bind((host, port))
    except OSError as exc:
        raise SystemExit(f"[DCI] SGLang port {port} is unavailable before launch: {exc}")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  log "stage=sglang-dry-run begin"
  RUN_LOG="${SERVICE_LOG}" \
  CONDA_SH="${CONDA_SH}" \
  SGLANG_ENV="${SGLANG_ENV}" \
  MODEL_PATH="${MODEL_PATH}" \
  MODEL_KEY="${MODEL_KEY}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  SGLANG_HOST="${SGLANG_HOST}" \
  SGLANG_PORT="${SGLANG_PORT}" \
  BACKEND_BASE_URL="${BACKEND_BASE_URL}" \
  BACKEND_MANIFEST_PATH="${BACKEND_MANIFEST_PATH}" \
  REUSE_BACKEND_MANIFEST="${REUSE_BACKEND_MANIFEST}" \
  TP_SIZE="${TP_SIZE}" \
  CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  DTYPE="${DTYPE}" \
  TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
  SAMPLING_BACKEND="${SAMPLING_BACKEND}" \
  SGLANG_JIT_CXX="${SGLANG_JIT_CXX}" \
  SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS}" \
  DRY_RUN=1 \
  scripts/serve_sglang_minicpm5.sh
  log "dry-run complete; service and evaluation were not started"
  exit 0
fi

PREFLIGHT_STARTED="$(date +%s.%N)"
log "stage=sglang-launch begin"
RUN_LOG="${SERVICE_LOG}" \
CONDA_SH="${CONDA_SH}" \
SGLANG_ENV="${SGLANG_ENV}" \
MODEL_PATH="${MODEL_PATH}" \
MODEL_KEY="${MODEL_KEY}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
SGLANG_HOST="${SGLANG_HOST}" \
SGLANG_PORT="${SGLANG_PORT}" \
BACKEND_BASE_URL="${BACKEND_BASE_URL}" \
BACKEND_MANIFEST_PATH="${BACKEND_MANIFEST_PATH}" \
REUSE_BACKEND_MANIFEST="${REUSE_BACKEND_MANIFEST}" \
TP_SIZE="${TP_SIZE}" \
CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
DTYPE="${DTYPE}" \
TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
SAMPLING_BACKEND="${SAMPLING_BACKEND}" \
SGLANG_JIT_CXX="${SGLANG_JIT_CXX}" \
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS}" \
setsid scripts/serve_sglang_minicpm5.sh &
SGLANG_PID=$!
sleep 1
SGLANG_PGID="$(ps -o pgid= -p "${SGLANG_PID}" 2>/dev/null | tr -d '[:space:]' || true)"
WRAPPER_PGID="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
if [[ ! "${SGLANG_PGID}" =~ ^[1-9][0-9]*$ ]] || [[ "${SGLANG_PGID}" == "${WRAPPER_PGID}" ]]; then
  log "failed to isolate SGLang process group: pid=${SGLANG_PID} pgid=${SGLANG_PGID:-unset} wrapper_pgid=${WRAPPER_PGID:-unset}"
  tail -n 200 "${SERVICE_LOG}" || true
  exit 2
fi
log "stage=sglang-launch ok; pid=${SGLANG_PID} pgid=${SGLANG_PGID}"

if ! command -v curl >/dev/null 2>&1; then
  log "curl is required for endpoint readiness checks"
  exit 2
fi
CURL_BIN="$(command -v curl)"
MODELS_URL="${BACKEND_BASE_URL%/}/models"
CHAT_COMPLETIONS_URL="${BACKEND_BASE_URL%/}/chat/completions"
MODELS_RESPONSE_PATH="${DIAGNOSTIC_DIR}/models.json"
TOOL_SMOKE_REQUEST_PATH="${DIAGNOSTIC_DIR}/tool-parser-smoke-request.json"
TOOL_SMOKE_RESPONSE_PATH="${DIAGNOSTIC_DIR}/tool-parser-smoke-response.json"

host_curl() {
  env -u LD_LIBRARY_PATH "${CURL_BIN}" \
    --noproxy "127.0.0.1,localhost" \
    --connect-timeout 5 \
    --max-time 30 \
    "$@"
}

format_json_file() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

models_ready() {
  [[ -r "${BACKEND_MANIFEST_PATH}" ]] || return 1
  host_curl -fsS \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    "${MODELS_URL}" \
    -o "${MODELS_RESPONSE_PATH}" || return 1
  format_json_file "${MODELS_RESPONSE_PATH}" || return 1
  python - "${MODELS_RESPONSE_PATH}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
models = payload.get("data")
model_ids = [
    str(item.get("id", ""))
    for item in models
    if isinstance(item, dict)
] if isinstance(models, list) else []
if expected not in model_ids:
    raise SystemExit(f"expected model {expected!r}; endpoint advertised {model_ids!r}")
PY
}

log "waiting for SGLang readiness at ${MODELS_URL}"
READY=0
for attempt in $(seq 1 "${WAIT_ATTEMPTS}"); do
  if models_ready; then
    READY=1
    log "SGLang ready after attempt=${attempt} elapsed_seconds=$(((attempt - 1) * WAIT_SECONDS))"
    break
  fi
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    log "SGLang exited before readiness; tailing service log"
    tail -n 200 "${SERVICE_LOG}" || true
    exit 2
  fi
  if (( attempt == 1 || attempt % 12 == 0 )); then
    log "SGLang not ready; attempt=${attempt}/${WAIT_ATTEMPTS} elapsed_seconds=$((attempt * WAIT_SECONDS))"
    tail -n 20 "${SERVICE_LOG}" || true
  fi
  sleep "${WAIT_SECONDS}"
done
if [[ "${READY}" != "1" ]]; then
  log "SGLang readiness timed out after $((WAIT_ATTEMPTS * WAIT_SECONDS)) seconds"
  tail -n 200 "${SERVICE_LOG}" || true
  exit 2
fi

python - "${BACKEND_MANIFEST_PATH}" "${MODEL_KEY}" "${BACKEND_BASE_URL}" <<'PY'
import sys
from pathlib import Path

from dci_bench.backends.manifest import load_backend_manifest, sanitize_base_url

path, expected_model_key, expected_url = sys.argv[1:]
manifest = load_backend_manifest(Path(path))
if manifest["model"]["model_key"] != expected_model_key:
    raise SystemExit("backend manifest model_key does not match the requested model")
if manifest["backend"]["base_url"] != sanitize_base_url(expected_url):
    raise SystemExit("backend manifest endpoint does not match the launched service")
print("[DCI] backend manifest identity check passed")
PY

python - "${SERVED_MODEL_NAME}" "${TOOL_SMOKE_REQUEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model, path = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": "Call get_weather for Beijing now. Do not answer in prose.",
    }],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "temperature": 0.2,
    "max_tokens": 128,
}
Path(path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

log "stage=tool-parser-smoke begin"
host_curl -fsS \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -X POST "${CHAT_COMPLETIONS_URL}" \
  --data-binary "@${TOOL_SMOKE_REQUEST_PATH}" \
  -o "${TOOL_SMOKE_RESPONSE_PATH}"
format_json_file "${TOOL_SMOKE_RESPONSE_PATH}"
python - "${TOOL_SMOKE_RESPONSE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
choices = payload.get("choices")
message = choices[0].get("message") if isinstance(choices, list) and choices else None
tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
if not isinstance(tool_calls, list) or not tool_calls:
    raise SystemExit("tool parser smoke did not return tool_calls")
function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
if not isinstance(function, dict) or function.get("name") != "get_weather":
    raise SystemExit("tool parser smoke returned an unexpected function")
arguments = function.get("arguments")
parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
if not isinstance(parsed, dict) or not parsed:
    raise SystemExit("tool parser smoke returned invalid arguments")
if choices[0].get("finish_reason") != "tool_calls":
    raise SystemExit("tool parser smoke finish_reason was not tool_calls")
usage = payload.get("usage")
if not isinstance(usage, dict) or usage.get("total_tokens") is None:
    raise SystemExit("tool parser smoke did not return usage")
print("[DCI] tool parser smoke passed")
PY

PREFLIGHT_SECONDS="$(python - "${PREFLIGHT_STARTED}" <<'PY'
import sys
import time

print(max(0.0, time.time() - float(sys.argv[1])))
PY
)"

read -r -a METRIC_K_ARRAY <<<"${METRIC_KS}"
RUNNER_ARGS=(
  python scripts/run_all.py
  --model-key "${MODEL_KEY}"
  --backend-manifest "${BACKEND_MANIFEST_PATH}"
  --backend-artifact "${MODELS_RESPONSE_PATH}"
  --backend-artifact "${TOOL_SMOKE_REQUEST_PATH}"
  --backend-artifact "${TOOL_SMOKE_RESPONSE_PATH}"
  --results-root "${RESULTS_ROOT}"
  --workspace-root "${WORKSPACE_ROOT}"
  --metadata-root "${METADATA_ROOT}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --bridge-port "${BRIDGE_PORT}"
  --api-key "${OPENAI_API_KEY}"
  --launcher-preflight-seconds "${PREFLIGHT_SECONDS}"
  --metric-ks "${METRIC_K_ARRAY[@]}"
)
if [[ -n "${RUN_ID}" ]]; then
  RUNNER_ARGS+=(--run-id "${RUN_ID}")
fi

log "stage=phase5-evaluation begin; tasks=6"
if "${RUNNER_ARGS[@]}"; then
  RUNNER_EXIT=0
else
  RUNNER_EXIT=$?
fi
JOB_EXIT="${RUNNER_EXIT}"
if [[ "${RUNNER_EXIT}" == "1" && "${FAIL_JOB_ON_SAMPLE_FAILURES}" == "0" ]]; then
  JOB_EXIT=0
  log "WARNING: evaluation completed with invalid/failed samples; preserving runner_exit=1 in the summary and returning job_exit=0"
fi
if [[ "${RUNNER_EXIT}" == "0" ]]; then
  EVAL_STATUS="complete"
elif [[ "${RUNNER_EXIT}" == "1" ]]; then
  EVAL_STATUS="complete_with_sample_failures"
else
  EVAL_STATUS="failed"
fi
log "stage=phase5-evaluation end; status=${EVAL_STATUS} runner_exit=${RUNNER_EXIT} job_exit=${JOB_EXIT}"
exit "${JOB_EXIT}"
