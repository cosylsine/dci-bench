#!/usr/bin/env bash
# Run one 4-GPU vLLM replica per node for the six-task Phase 5 batch.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NODE_RANK="${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-}}}}"
WORLD_SIZE="${WORLD_SIZE:-${SENSECORE_PYTORCH_NNODES:-${SLURM_NNODES:-3}}}"
ONLY_TASK="${ONLY_TASK:-}"
if [[ "${WORLD_SIZE}" != "3" && "${WORLD_SIZE}" != "4" && "${WORLD_SIZE}" != "1" ]] ||
   { [[ "${WORLD_SIZE}" != "1" ]] && [[ ! "${NODE_RANK}" =~ ^[0-9]+$ ]]; } ||
   { [[ "${WORLD_SIZE}" == "3" ]] && [[ ! "${NODE_RANK}" =~ ^[0-2]$ ]]; } ||
   { [[ "${WORLD_SIZE}" == "4" ]] && [[ ! "${NODE_RANK}" =~ ^[0-3]$ ]]; } ||
   { [[ "${WORLD_SIZE}" == "1" ]] && [[ "${NODE_RANK}" != "0" ]]; }; then
  echo "[DCI] use WORLD_SIZE=3 with NODE_RANK=0,1,2 or WORLD_SIZE=1 with NODE_RANK=0 and ONLY_TASK" >&2
  exit 2
fi

if [[ "${WORLD_SIZE}" == "1" ]]; then
  case "${ONLY_TASK}" in
      LLMAILAStatutes|LLMFQuADRetrieval|LLMHC3FinanceRetrieval|LLMLegalBenchConsumerContractsQA|LLMPublicHealthQA|LLMTwitterHjerneRetrieval)
      TASKS=("${ONLY_TASK}") ;;
    *) echo "[DCI] WORLD_SIZE=1 requires ONLY_TASK to name one DCI task" >&2; exit 2 ;;
  esac
else
  # Keep one model replica per node. The three-node assignment covers all six
  # tasks; the old four-node layout remains available for existing jobs.
  case "${NODE_RANK}" in
    0) TASKS=(LLMAILAStatutes LLMFQuADRetrieval) ;;
    1) TASKS=(LLMHC3FinanceRetrieval LLMTwitterHjerneRetrieval) ;;
    2)
      if [[ "${WORLD_SIZE}" == "3" ]]; then
        TASKS=(LLMLegalBenchConsumerContractsQA LLMPublicHealthQA)
      else
        TASKS=(LLMLegalBenchConsumerContractsQA)
      fi
      ;;
    3) TASKS=(LLMPublicHealthQA) ;;
  esac
  if [[ -n "${ONLY_TASK}" ]]; then
    echo "[DCI] ONLY_TASK is supported with WORLD_SIZE=1; cluster task assignments are fixed" >&2
    exit 2
  fi
fi

SCHEDULER_NAME="none"
SCHEDULER_JOB_ID=""
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCHEDULER_NAME="slurm"
  SCHEDULER_JOB_ID="${SLURM_JOB_ID}"
elif [[ -n "${SENSECORE_JOB_ID:-}" ]]; then
  SCHEDULER_NAME="sensecore"
  SCHEDULER_JOB_ID="${SENSECORE_JOB_ID}"
elif [[ -n "${SENSECORE_PYTORCH_JOB_ID:-}" ]]; then
  SCHEDULER_NAME="sensecore"
  SCHEDULER_JOB_ID="${SENSECORE_PYTORCH_JOB_ID}"
elif [[ -n "${JOB_ID:-}" ]]; then
  SCHEDULER_NAME="generic"
  SCHEDULER_JOB_ID="${JOB_ID}"
elif [[ -n "${LSB_JOBID:-}" ]]; then
  SCHEDULER_NAME="lsf"
  SCHEDULER_JOB_ID="${LSB_JOBID}"
elif [[ -n "${PBS_JOBID:-}" ]]; then
  SCHEDULER_NAME="pbs"
  SCHEDULER_JOB_ID="${PBS_JOBID}"
fi
JOB_TAG="${SCHEDULER_JOB_ID:-manual-${BASHPID}}"
JOB_TAG="${JOB_TAG//\//_}"
JOB_TAG="${JOB_TAG//:/_}"
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

DRY_RUN="${DRY_RUN:-0}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs/share/Qwen3.8-27B}"
MODEL_KEY="${MODEL_KEY:-Qwen3.8-27B}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/data/workspaces}"
METADATA_ROOT="${METADATA_ROOT:-${REPO_ROOT}/data/metadata}"
VLLM_ENV="${VLLM_ENV:-dci-vllm-qwen38}"
BENCH_ENV="${BENCH_ENV:-dci-bench}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010100/miniconda}"
BENCH_PYTHON="${BENCH_PYTHON:-${CONDA_ROOT}/envs/${BENCH_ENV}/bin/python}"

# DCI_RUN_ID is the public cluster knob. RUN_ID remains accepted for older
# launch snippets, while scheduler IDs keep all ranks in one immutable group.
EXPLICIT_RUN_ID="${DCI_RUN_ID:-${RUN_ID:-}}"
if [[ -n "${EXPLICIT_RUN_ID}" && ! "${EXPLICIT_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "[DCI] DCI_RUN_ID/RUN_ID must be one safe path component" >&2
  exit 2
fi
RUN_ID="${EXPLICIT_RUN_ID:-${SCHEDULER_JOB_ID:-manual-${RUN_TIMESTAMP}-${BASHPID}}}"
RUN_ID="${RUN_ID//\//_}"
RUN_ID="${RUN_ID//:/_}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "[DCI] DCI_RUN_ID/RUN_ID must be one safe path component" >&2
  exit 2
fi

RUN_LOG_EXPLICIT="${RUN_LOG+x}"
RUN_LOG="${RUN_LOG:-${RESULTS_ROOT}/job-logs/qwen38-vllm-${JOB_TAG}-rank${NODE_RANK}-${RUN_TIMESTAMP}.log}"
SERVICE_LOG="${SERVICE_LOG:-${RESULTS_ROOT}/job-logs/qwen38-vllm-service-${JOB_TAG}-rank${NODE_RANK}-${RUN_TIMESTAMP}.log}"
DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-${RESULTS_ROOT}/job-artifacts/qwen38-vllm-${JOB_TAG}-rank${NODE_RANK}-${RUN_TIMESTAMP}}"
CLUSTER_SYNC_DIR="${CLUSTER_SYNC_DIR:-${RESULTS_ROOT}/cluster-runs/${RUN_ID}}"
BACKEND_MANIFEST_PATH="${BACKEND_MANIFEST_PATH:-${RESULTS_ROOT}/backend-manifests/vllm-${MODEL_KEY}-${RUN_ID}-rank${NODE_RANK}-${RUN_TIMESTAMP}.json}"

log() { printf '[DCI][%s][rank=%s] %s\n' "$(date --iso-8601=seconds)" "${NODE_RANK}" "$*"; }
if [[ "${DRY_RUN}" != "1" ]]; then
  for immutable_path in "${RUN_LOG}" "${SERVICE_LOG}" "${DIAGNOSTIC_DIR}"; do
    if [[ -e "${immutable_path}" ]]; then
      log "refusing to reuse existing launcher artifact path: ${immutable_path}"
      exit 2
    fi
  done
  mkdir -p "$(dirname "${RUN_LOG}")" "$(dirname "${SERVICE_LOG}")" "${DIAGNOSTIC_DIR}"
  exec > >(tee -a "${RUN_LOG}") 2>&1
elif [[ -n "${RUN_LOG_EXPLICIT}" ]]; then
  # Tests and local audits may provide a temporary log directory.  A dry run
  # never creates the default results, diagnostic, or synchronization paths.
  mkdir -p "$(dirname "${RUN_LOG}")"
  exec > >(tee -a "${RUN_LOG}") 2>&1
fi

log "Phase 5 Qwen3.8 node launcher; host=${HOSTNAME:-unknown} scheduler=${SCHEDULER_NAME} scheduler_job_id=${SCHEDULER_JOB_ID:-unset}"
log "world_size=${WORLD_SIZE} node_rank=${NODE_RANK} tasks=${TASKS[*]} model=${MODEL_PATH} model_key=${MODEL_KEY} run_id=${RUN_ID}"
log "results_root=${RESULTS_ROOT} sync_dir=${CLUSTER_SYNC_DIR} backend_manifest=${BACKEND_MANIFEST_PATH}"

if [[ "${WORLD_SIZE}" != "1" && -z "${DCI_RUN_ID:-}" && -z "${SCHEDULER_JOB_ID}" && "${DRY_RUN}" != "1" ]]; then
  log "multi-node runs outside a scheduler require explicit DCI_RUN_ID shared by all ranks"
  exit 2
fi

VLLM_PID=""
VLLM_PGID=""
EVAL_PID=""
EVAL_PGID=""
HEARTBEAT_PID=""
SYNC_INITIALIZED=0
TERMINAL_WRITTEN=0
SIGNAL_EXIT=0
EVAL_ABORTED=0
EVAL_STOP_GRACE_SECONDS="${EVAL_STOP_GRACE_SECONDS:-30}"

process_group_exists() {
  local pgid="$1"
  [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] && kill -0 -- "-${pgid}" 2>/dev/null
}

stop_evaluation_tree() {
  local pid="${EVAL_PID}"
  local pgid="${EVAL_PGID}"
  local own_pgid
  own_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ -n "${pgid}" && "${pgid}" != "${own_pgid}" ]] && process_group_exists "${pgid}"; then
    log "stopping evaluation process group pgid=${pgid}"
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for ((attempt=0; attempt<EVAL_STOP_GRACE_SECONDS; attempt+=1)); do
      process_group_exists "${pgid}" || break
      sleep 1
    done
    if process_group_exists "${pgid}"; then
      log "evaluation process group exceeded ${EVAL_STOP_GRACE_SECONDS}s; sending SIGKILL"
      kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
  elif [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    log "stopping evaluation pid=${pid}"
    kill -TERM "${pid}" 2>/dev/null || true
    for ((attempt=0; attempt<EVAL_STOP_GRACE_SECONDS; attempt+=1)); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  if [[ -n "${pid}" ]]; then
    wait "${pid}" 2>/dev/null || true
  fi
  EVAL_PID=""
  EVAL_PGID=""
}

cleanup() {
  local exit_code=$?
  trap - EXIT
  stop_evaluation_tree
  if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  if [[ "${SYNC_INITIALIZED}" == "1" && "${TERMINAL_WRITTEN}" != "1" ]]; then
    local terminal_status="failure"
    if [[ "${SIGNAL_EXIT}" != "0" ]]; then terminal_status="aborted"; fi
    if "${BENCH_PYTHON}" scripts/cluster_run_sync.py terminal "${SYNC_ARGS[@]}" \
      --status "${terminal_status}" --exit-code "${exit_code}" \
      --reason "launcher exited before publishing a terminal result"; then
      terminal_marker_exit=0
    else
      terminal_marker_exit=$?
    fi
    if [[ "${terminal_marker_exit}" != "0" ]]; then
      log "unable to publish ${terminal_status} terminal marker; helper_exit=${terminal_marker_exit}"
      exit_code=2
    else
      TERMINAL_WRITTEN=1
    fi
  fi
  if [[ -n "${VLLM_PID}" ]] && {
    kill -0 "${VLLM_PID}" 2>/dev/null || process_group_exists "${VLLM_PGID}"
  }; then
    log "stopping vLLM process group ${VLLM_PGID}"
    if process_group_exists "${VLLM_PGID}"; then
      kill -TERM -- "-${VLLM_PGID}" 2>/dev/null || true
    elif kill -0 "${VLLM_PID}" 2>/dev/null; then
      kill -TERM "${VLLM_PID}" 2>/dev/null || true
    fi
    for ((attempt=0; attempt<30; attempt+=1)); do
      if ! process_group_exists "${VLLM_PGID}" && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if process_group_exists "${VLLM_PGID}"; then
      log "vLLM process group exceeded 30s; sending SIGKILL"
      kill -KILL -- "-${VLLM_PGID}" 2>/dev/null || true
    elif kill -0 "${VLLM_PID}" 2>/dev/null; then
      kill -KILL "${VLLM_PID}" 2>/dev/null || true
    fi
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  if [[ "${SYNC_INITIALIZED}" == "1" && "${SIGNAL_EXIT}" == "0" ]]; then
    if "${BENCH_PYTHON}" scripts/cluster_run_sync.py wait "${SYNC_ARGS[@]}" \
      --timeout "${BARRIER_TIMEOUT_SECONDS}" --poll-seconds "${BARRIER_POLL_SECONDS}"; then
      barrier_exit=0
    else
      barrier_exit=$?
    fi
    if [[ "${barrier_exit}" != "0" && "${exit_code}" == "0" ]]; then
      exit_code="${barrier_exit}"
    fi
  fi
  log "finished exit=${exit_code}; run_log=${RUN_LOG}; service_log=${SERVICE_LOG}; diagnostics=${DIAGNOSTIC_DIR}"
  exit "${exit_code}"
}
trap cleanup EXIT
handle_signal() {
  local signal_name="$1"
  local signal_code="$2"
  SIGNAL_EXIT="${signal_code}"
  log "received ${signal_name}; publishing abort and stopping local work"
  if [[ "${SYNC_INITIALIZED}" == "1" && "${TERMINAL_WRITTEN}" != "1" ]]; then
    if "${BENCH_PYTHON}" scripts/cluster_run_sync.py terminal "${SYNC_ARGS[@]}" \
      --status aborted --exit-code "${signal_code}" --reason "${signal_name}"; then
      TERMINAL_WRITTEN=1
    fi
  fi
  stop_evaluation_tree
  exit "${signal_code}"
}
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

SYNC_HELPER="${REPO_ROOT}/scripts/cluster_run_sync.py"
sync_call() {
  "${BENCH_PYTHON}" "${SYNC_HELPER}" "$1" "${SYNC_ARGS[@]}" "${@:2}"
}

cluster_abort_requested() {
  local helper_exit
  if sync_call check-abort >/dev/null 2>&1; then
    helper_exit=0
  else
    helper_exit=$?
  fi
  if [[ "${helper_exit}" == "0" ]]; then
    return 0
  fi
  if [[ "${helper_exit}" != "1" ]]; then
    log "cluster synchronization check failed; treating it as an abort (helper_exit=${helper_exit})"
    return 0
  fi
  return 1
}

publish_terminal() {
  local status="$1"
  local code="$2"
  local reason="$3"
  if [[ "${SYNC_INITIALIZED}" != "1" || "${TERMINAL_WRITTEN}" == "1" ]]; then
    return 0
  fi
  local helper_exit
  if sync_call terminal --status "${status}" --exit-code "${code}" --reason "${reason}"; then
    helper_exit=0
  else
    helper_exit=$?
  fi
  if [[ "${helper_exit}" == "0" ]]; then
    TERMINAL_WRITTEN=1
    return 0
  fi
  log "failed to publish terminal status=${status}; helper_exit=${helper_exit}"
  return "${helper_exit}"
}

start_heartbeat() {
  if [[ "${SYNC_INITIALIZED}" != "1" ]]; then return 0; fi
  (
    while :; do
      sync_call heartbeat --state running >/dev/null 2>&1 || exit 0
      sleep "${HEARTBEAT_SECONDS}"
    done
  ) &
  HEARTBEAT_PID=$!
}

run_evaluation_with_watchdog() {
  setsid "${RUNNER_ARGS[@]}" &
  EVAL_PID=$!
  local evaluation_pid="${EVAL_PID}"
  EVAL_PGID="$(ps -o pgid= -p "${EVAL_PID}" 2>/dev/null | tr -d '[:space:]' || true)"
  runner_exit=2
  while kill -0 "${EVAL_PID}" 2>/dev/null; do
    if cluster_abort_requested; then
      log "peer rank published a terminal failure; stopping local evaluation"
      EVAL_ABORTED=1
      stop_evaluation_tree
      runner_exit=143
      return 0
    fi
    sleep 2
  done
  if process_group_exists "${EVAL_PGID}"; then
    stop_evaluation_tree
  fi
  if wait "${evaluation_pid}" 2>/dev/null; then
    runner_exit=0
  else
    runner_exit=$?
  fi
  # Reap the leader and clean up any bridge descendants left in its group.
  stop_evaluation_tree
}

CONDA_SH="${CONDA_SH:-${CONDA_ROOT}/etc/profile.d/conda.sh}"
VLLM_ENV="${VLLM_ENV:-dci-vllm-qwen38}"
BENCH_ENV="${BENCH_ENV:-dci-bench}"
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
BARRIER_TIMEOUT_SECONDS="${BARRIER_TIMEOUT_SECONDS:-172800}"
INIT_WAIT_TIMEOUT_SECONDS="${INIT_WAIT_TIMEOUT_SECONDS:-1800}"
BARRIER_HEARTBEAT_TIMEOUT_SECONDS="${BARRIER_HEARTBEAT_TIMEOUT_SECONDS:-1800}"
BARRIER_POLL_SECONDS="${BARRIER_POLL_SECONDS:-2}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-15}"
RUNTIME_ARTIFACT_PATH="${RUNTIME_ARTIFACT_PATH:-${DIAGNOSTIC_DIR}/serving-runtime-rank${NODE_RANK}.json}"
SYNC_ARGS=(
  --state-dir "${CLUSTER_SYNC_DIR}" --world-size "${WORLD_SIZE}" --rank "${NODE_RANK}"
  --run-id "${RUN_ID}" --model-key "${MODEL_KEY}" --tasks "${TASKS[@]}"
  --heartbeat-timeout "${BARRIER_HEARTBEAT_TIMEOUT_SECONDS}"
)

for name in VLLM_PORT TENSOR_PARALLEL_SIZE MAX_MODEL_LEN MAX_CONCURRENCY VLLM_MAX_NUM_SEQS BRIDGE_PORT WAIT_ATTEMPTS WAIT_SECONDS TOOL_SMOKE_TIMEOUT_SECONDS BARRIER_TIMEOUT_SECONDS INIT_WAIT_TIMEOUT_SECONDS BARRIER_HEARTBEAT_TIMEOUT_SECONDS BARRIER_POLL_SECONDS HEARTBEAT_SECONDS EVAL_STOP_GRACE_SECONDS; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    log "${name} must be a positive integer"
    exit 2
  fi
done
if (( VLLM_MAX_NUM_SEQS < MAX_CONCURRENCY )); then
  log "VLLM_MAX_NUM_SEQS must be at least MAX_CONCURRENCY"
  exit 2
fi
if [[ ! "${VLLM_GPU_MEMORY_UTILIZATION}" =~ ^0\.[0-9]+$ ]] ||
   [[ "${VLLM_GPU_MEMORY_UTILIZATION}" =~ ^0\.0+$ ]]; then
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
if [[ "${FAIL_JOB_ON_SAMPLE_FAILURES}" != "0" && "${FAIL_JOB_ON_SAMPLE_FAILURES}" != "1" ]]; then
  log "FAIL_JOB_ON_SAMPLE_FAILURES must be 0 or 1"
  exit 2
fi
log "host=${HOSTNAME:-unknown} tasks=${TASKS[*]} model=${MODEL_PATH} tp=4 concurrency=${MAX_CONCURRENCY}"
log "vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS} vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN}"
log "results_root=${RESULTS_ROOT} workspace_root=${WORKSPACE_ROOT} metadata_root=${METADATA_ROOT}"
log "runtime_artifact=${RUNTIME_ARTIFACT_PATH}"
log "cluster_init_timeout=${INIT_WAIT_TIMEOUT_SECONDS}s barrier_timeout=${BARRIER_TIMEOUT_SECONDS}s heartbeat_stale_timeout=${BARRIER_HEARTBEAT_TIMEOUT_SECONDS}s"
for task in "${TASKS[@]}"; do
  log "task_path task=${task} run_dir=${RESULTS_ROOT}/${MODEL_KEY}/${task}/runs/${RUN_ID}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run complete; service, evaluation, synchronization markers, and result directories were not started or created"
  exit 0
fi

if [[ "${WORLD_SIZE}" != "1" ]]; then
  if [[ ! -x "${BENCH_PYTHON}" ]]; then
    log "required benchmark Python is missing or not executable: ${BENCH_PYTHON}"
    exit 2
  fi
  if sync_call init --timeout "${INIT_WAIT_TIMEOUT_SECONDS}" --poll-seconds "${BARRIER_POLL_SECONDS}"; then
    sync_init_exit=0
  else
    sync_init_exit=$?
  fi
  if [[ "${sync_init_exit}" != "0" ]]; then
    log "cluster synchronization initialization failed; helper_exit=${sync_init_exit}"
    exit 2
  fi
  SYNC_INITIALIZED=1
  start_heartbeat
  if cluster_abort_requested; then
    log "another rank failed before this node began its preflight"
    exit 2
  fi
fi

for task in "${TASKS[@]}"; do
  existing_run_dir="${RESULTS_ROOT}/${MODEL_KEY}/${task}/runs/${RUN_ID}"
  if [[ -e "${existing_run_dir}" ]]; then
    log "refusing to reuse existing immutable result path: ${existing_run_dir}"
    exit 2
  fi
done

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
"${BENCH_PYTHON}" scripts/run_all.py --preflight-only --workspace-root "${WORKSPACE_ROOT}" --metadata-root "${METADATA_ROOT}"
(
  cd pi-dci
  npm exec -- tsx scripts/dci-run.ts --preflight-only --workspace "${WORKSPACE_ROOT}/${TASKS[0]}"
)
if [[ "${SYNC_INITIALIZED}" == "1" ]] && cluster_abort_requested; then
  log "peer rank failed during data preflight"
  exit 2
fi
"${BENCH_PYTHON}" - "${VLLM_HOST}" "${VLLM_PORT}" <<'PY'
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
VLLM_PYTHON="${VLLM_PYTHON:-$(command -v python)}"
log "5090 and Qwen3.8 serving preflight; vllm_env=${VLLM_ENV} python=${VLLM_PYTHON}"
"${VLLM_PYTHON}" - <<'PY'
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
for index in range(4):
    # A device-count query alone can succeed while the first kernel later
    # fails on an incompatible 5090 build.  Exercise and synchronize one
    # actual CUDA kernel on every visible card before loading vLLM.
    torch.ones(1, device=f"cuda:{index}").add_(1).item()
    torch.cuda.synchronize(index)
    print(f"[DCI] CUDA kernel preflight passed on GPU {index}", flush=True)
vllm_version = version("vllm")
assert tuple(int(x) for x in vllm_version.split(".")[:2]) >= (0, 17), "vLLM 0.17+ required"
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config  # noqa: F401
print(f"[DCI] torch={torch.__version__} CUDA={torch.version.cuda} vllm={vllm_version}", flush=True)
PY
VLLM_VERSION="$("${VLLM_PYTHON}" -c 'from importlib.metadata import version; print(version("vllm"))')"
if [[ -e "${RUNTIME_ARTIFACT_PATH}" ]]; then
  log "serving runtime artifact already exists: ${RUNTIME_ARTIFACT_PATH}"
  exit 2
fi
"${VLLM_PYTHON}" - "${RUNTIME_ARTIFACT_PATH}" "${VLLM_ENV}" "${VLLM_VERSION}" <<'PY'
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

path = Path(sys.argv[1])
runtime_env = sys.argv[2]
vllm_version = sys.argv[3]
torch = __import__("torch")
try:
    driver_output = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    driver_versions = sorted({line.strip() for line in driver_output.splitlines() if line.strip()})
except (OSError, subprocess.CalledProcessError):
    driver_versions = []

packages = {}
for package in ("torch", "vllm", "transformers", "xformers", "flashinfer", "numpy"):
    try:
        packages[package] = version(package)
    except PackageNotFoundError:
        packages[package] = None

gpus = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    gpus.append(
        {
            "index": index,
            "name": properties.name,
            "capability": {"major": properties.major, "minor": properties.minor},
            "total_memory_bytes": properties.total_memory,
        }
    )
payload = {
    "schema_version": "dci-serving-runtime-v1",
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "python": {"version": platform.python_version(), "executable": sys.executable},
    "environment": runtime_env,
    "torch": {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compiled_arch_flags": list(torch.cuda.get_arch_list()),
    },
    "vllm": {"version": vllm_version},
    "cuda": {
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_count": torch.cuda.device_count(),
        "driver_versions": driver_versions,
        "kernel_probe": "passed-on-every-visible-device",
        "gpus": gpus,
    },
    "packages": packages,
}
path.parent.mkdir(parents=True, exist_ok=True)
data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
    temporary = Path(handle.name)
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
try:
    # Hard-link publication refuses to replace an artifact from an earlier run.
    os.link(temporary, path)
finally:
    temporary.unlink(missing_ok=True)
print(path, flush=True)
PY
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
  if [[ "${SYNC_INITIALIZED}" == "1" ]] && cluster_abort_requested; then
    log "peer rank failed while waiting for vLLM readiness"
    exit 2
  fi
  if host_curl 30 -fsS "${MODELS_URL}" -o "${MODELS_RESPONSE_PATH}" 2>/dev/null; then
    if "${BENCH_PYTHON}" - "${MODELS_RESPONSE_PATH}" "${SERVED_MODEL_NAME}" <<'PY'
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
if [[ "${SYNC_INITIALIZED}" == "1" ]] && cluster_abort_requested; then
  log "peer rank failed before tool parser smoke"
  exit 2
fi

set +u
conda activate "${BENCH_ENV}"
set -u
"${BENCH_PYTHON}" - "${SERVED_MODEL_NAME}" "${TOOL_SMOKE_REQUEST_PATH}" <<'PY'
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
"${BENCH_PYTHON}" - "${TOOL_SMOKE_RESPONSE_PATH}" <<'PY'
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
"${BENCH_PYTHON}" -m dci_bench.backends.manifest \
  --output "${BACKEND_MANIFEST_PATH}" --model-key "${MODEL_KEY}" \
  --model-path "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --vllm-version "${VLLM_VERSION}" --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser "${REASONING_PARSER}" --language-model-only \
  --sampling-backend auto --tensor-parallel-size 4 \
  --context-length "${MAX_MODEL_LEN}" --dtype bfloat16 \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --runtime-info "${RUNTIME_ARTIFACT_PATH}"

read -r -a METRIC_K_ARRAY <<<"${METRIC_KS}"
RUNNER_ARGS=(
  "${BENCH_PYTHON}" scripts/run_all.py --model-key "${MODEL_KEY}"
  --tasks "${TASKS[@]}" --backend-manifest "${BACKEND_MANIFEST_PATH}"
  --backend-artifact "${MODELS_RESPONSE_PATH}"
  --backend-artifact "${TOOL_SMOKE_REQUEST_PATH}"
  --backend-artifact "${TOOL_SMOKE_RESPONSE_PATH}"
  --backend-artifact "${RUNTIME_ARTIFACT_PATH}"
  --results-root "${RESULTS_ROOT}" --workspace-root "${WORKSPACE_ROOT}"
  --metadata-root "${METADATA_ROOT}" --max-concurrency "${MAX_CONCURRENCY}"
  --bridge-port "${BRIDGE_PORT}" --api-key "${OPENAI_API_KEY}"
  --metric-ks "${METRIC_K_ARRAY[@]}"
)
if [[ -n "${RUN_ID}" ]]; then RUNNER_ARGS+=(--run-id "${RUN_ID}"); fi
log "evaluation begins: ${TASKS[*]}"
if [[ "${SYNC_INITIALIZED}" == "1" ]]; then
  run_evaluation_with_watchdog
else
  if "${RUNNER_ARGS[@]}"; then runner_exit=0; else runner_exit=$?; fi
fi
job_exit="${runner_exit}"
terminal_status="success"
terminal_reason="all assigned tasks completed"
if [[ "${EVAL_ABORTED}" == "1" ]]; then
  terminal_status="aborted"
  terminal_reason="peer rank failed; local evaluation was stopped"
  job_exit=2
elif [[ "${runner_exit}" == "1" && "${FAIL_JOB_ON_SAMPLE_FAILURES}" == "0" ]]; then
  job_exit=0
  terminal_status="complete_with_sample_failures"
  terminal_reason="assigned tasks completed with sample failures"
elif [[ "${runner_exit}" != "0" ]]; then
  terminal_status="failure"
  terminal_reason="run_all exited with configuration or integrity failure"
fi
if [[ "${SYNC_INITIALIZED}" == "1" ]]; then
  if ! publish_terminal "${terminal_status}" "${job_exit}" "${terminal_reason}"; then
    job_exit=2
  fi
fi
log "evaluation finished: runner_exit=${runner_exit} job_exit=${job_exit} terminal_status=${terminal_status} tasks=${TASKS[*]}"
exit "${job_exit}"
