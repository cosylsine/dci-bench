#!/usr/bin/env bash
set -Eeuo pipefail

report_error() {
  local exit_code="$1"
  local line_number="$2"
  local failed_command="$3"
  printf '[DCI][%s] ERROR exit=%s source=%s line=%s command=%s\n' \
    "$(date --iso-8601=seconds)" \
    "${exit_code}" \
    "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}" \
    "${line_number}" \
    "${failed_command}" >&2
}
trap 'report_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-${LSB_JOBID:-${PBS_JOBID:-manual-$$}}}}"
JOB_TAG="${JOB_TAG//\//_}"
RUN_LOG="${RUN_LOG:-${REPO_ROOT}/results/job-logs/phase3-${JOB_TAG}-${RUN_TIMESTAMP}.log}"
mkdir -p "$(dirname "${RUN_LOG}")"
exec > >(tee -a "${RUN_LOG}") 2>&1

log() {
  printf '[DCI][%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

VLLM_PID=""
cleanup() {
  local exit_code=$?
  log "job exiting; exit=${exit_code} vllm_pid=${VLLM_PID:-not-started}"
  if [[ -n "${VLLM_PID}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    log "vLLM cleanup complete; pid=${VLLM_PID}"
  fi
}
trap cleanup EXIT

log "job started; pid=$$ host=${HOSTNAME:-unknown} user=$(id -un) uid=$(id -u)"
log "repo_root=${REPO_ROOT}"
log "run_log=${RUN_LOG}"
log "scheduler: SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
log "scheduler: JOB_ID=${JOB_ID:-unset} LSB_JOBID=${LSB_JOBID:-unset} PBS_JOBID=${PBS_JOBID:-unset}"
log "initial CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset}"

if [[ "${DEBUG_TRACE:-0}" == "1" ]]; then
  export PS4='+ [${BASH_SOURCE}:${LINENO}] '
  set -x
fi

if command -v bwrap >/dev/null 2>&1; then
  log "bubblewrap already available: $(command -v bwrap)"
else
  log "bubblewrap missing; preparing OS package installation"
  log "apt-get=$(command -v apt-get || printf 'not-found') user=$(id -un) uid=$(id -u)"
  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get is unavailable; use a cluster image that provides /usr/bin/bwrap"
    exit 2
  fi
  log "stage=apt-update begin"
  apt-get update -o Acquire::Retries=3
  log "stage=apt-update ok"
  apt-cache policy bubblewrap || true
  log "stage=bubblewrap-install begin"
  DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends bubblewrap
  log "stage=bubblewrap-install ok; bwrap=$(command -v bwrap || printf 'not-found')"
fi

CONDA_SH="${CONDA_SH:-/mnt/afs/250010100/miniconda/etc/profile.d/conda.sh}"
VLLM_ENV="${VLLM_ENV:-dci-vllm-qwen38}"
BENCH_ENV="${BENCH_ENV:-dci-bench}"

MODEL_PATH="${MODEL_PATH:-/mnt/afs/share/Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
# TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
QUERY_IDS="${QUERY_IDS:-Q71 Q72}"
TASK="${TASK:-LLMPublicHealthQA}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase3_qwen38_27b_f1_brwap}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-240}"
WAIT_SECONDS="${WAIT_SECONDS:-5}"

if ! command -v bwrap >/dev/null 2>&1; then
  echo "[DCI] bubblewrap is required and this container loses apt packages after restart." >&2
  echo "[DCI] install it before running: apt install -y bubblewrap" >&2
  exit 2
fi
log "stage=bubblewrap-check ok; path=$(command -v bwrap)"

log "stage=conda-init begin; conda_sh=${CONDA_SH}"
if [[ ! -r "${CONDA_SH}" ]]; then
  log "conda initialization script is missing or unreadable: ${CONDA_SH}"
  exit 2
fi
source "${CONDA_SH}"
log "stage=conda-init ok; conda=$(command -v conda || printf 'not-found')"

log "stage=bench-env begin; activating ${BENCH_ENV} for Node/npm preflight"
conda activate "${BENCH_ENV}"
log "stage=bench-env activated; python=$(command -v python)"
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  log "Node/npm are unavailable in BENCH_ENV=${BENCH_ENV}; expected them under the activated Conda environment"
  exit 2
fi
log "stage=pi-preflight begin; node=$(command -v node) npm=$(command -v npm) workspace=${REPO_ROOT}/data/workspaces/${TASK}"
node --version
npm --version
(
  cd pi-dci
  npm exec -- tsx scripts/dci-run.ts \
    --preflight-only \
    --workspace "${REPO_ROOT}/data/workspaces/${TASK}"
)
log "stage=pi-preflight ok"

mkdir -p results/vllm-logs
VLLM_LOG="${VLLM_LOG:-results/vllm-logs/${SERVED_MODEL_NAME//\//_}-$(date +%Y%m%d-%H%M%S).log}"

export MODEL_PATH SERVED_MODEL_NAME VLLM_HOST VLLM_PORT TENSOR_PARALLEL_SIZE MAX_MODEL_LEN
export TOOL_CALL_PARSER REASONING_PARSER LANGUAGE_MODEL_ONLY
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log "runtime config: model=${MODEL_PATH} served_name=${SERVED_MODEL_NAME} task=${TASK} query_ids=${QUERY_IDS} output_dir=${OUTPUT_DIR}"
log "runtime config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} TP=${TENSOR_PARALLEL_SIZE} max_model_len=${MAX_MODEL_LEN} host=${VLLM_HOST} port=${VLLM_PORT}"
log "runtime config: VLLM_ENV=${VLLM_ENV} BENCH_ENV=${BENCH_ENV} NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
if command -v nvidia-smi >/dev/null 2>&1; then
  log "stage=gpu-preflight nvidia-smi -L"
  nvidia-smi -L || log "WARNING: nvidia-smi -L failed"
else
  log "WARNING: nvidia-smi is not available"
fi

log "stage=vllm-env begin; activating ${VLLM_ENV}"
conda activate "${VLLM_ENV}"
log "stage=vllm-env activated; python=$(command -v python)"
python - <<'PY'
import json
import os
import re
import sys
from importlib import metadata
from pathlib import Path


def version_tuple(text):
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


model_path = Path(os.environ["MODEL_PATH"])
config_path = model_path / "config.json"
model_type = None
required_transformers = None
if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    required_transformers = config.get("transformers_version")

try:
    vllm_version = metadata.version("vllm")
    transformers_version = metadata.version("transformers")
except metadata.PackageNotFoundError as exc:
    print(f"[DCI] missing package in VLLM_ENV: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    import torch
except Exception as exc:
    print(f"[DCI] torch import failed: {exc}", file=sys.stderr)
    sys.exit(2)

print(
    f"[DCI] CUDA probe: torch={torch.__version__}, built_cuda={torch.version.cuda}, "
    f"available={torch.cuda.is_available()}, visible_devices={torch.cuda.device_count()}"
)
visible_devices = torch.cuda.device_count()
for index in range(visible_devices):
    props = torch.cuda.get_device_properties(index)
    print(
        f"[DCI] CUDA device {index}: name={props.name}, "
        f"total_memory_gib={props.total_memory / 1024**3:.2f}"
    )

requested_tp = int(os.environ["TENSOR_PARALLEL_SIZE"])
if not torch.cuda.is_available() or visible_devices < requested_tp:
    print(
        f"[DCI] insufficient visible GPUs: requested tensor parallel size={requested_tp}, "
        f"visible_devices={visible_devices}, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        file=sys.stderr,
    )
    sys.exit(2)

if model_type == "qwen3_5":
    failures = []
    if version_tuple(vllm_version) < (0, 17, 0):
        failures.append(f"vllm {vllm_version} < 0.17.0")
    try:
        from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config  # noqa: F401
    except Exception as exc:
        failures.append(f"vLLM Qwen3_5Config import failed: {exc}")
    if failures:
        print("[DCI] Qwen3.8/Qwen3.5 checkpoint requires a newer serving stack.", file=sys.stderr)
        print(f"[DCI] model_type={model_type}, checkpoint_transformers={required_transformers}", file=sys.stderr)
        print(f"[DCI] current: vllm={vllm_version}, transformers={transformers_version}", file=sys.stderr)
        print(f"[DCI] failing requirements: {', '.join(failures)}", file=sys.stderr)
        print("[DCI] create/use a separate VLLM_ENV with Qwen3.8-capable vLLM.", file=sys.stderr)
        sys.exit(2)
    if version_tuple(transformers_version) < (5, 8, 0):
        print(
            "[DCI] warning: transformers < 5.8.0; continuing because vLLM has Qwen3_5Config "
            "and LANGUAGE_MODEL_ONLY is intended for this text-only run.",
            file=sys.stderr,
        )

print(f"[DCI] preflight ok: model_type={model_type}, vllm={vllm_version}, transformers={transformers_version}")
PY
log "stage=vllm-env preflight ok"
echo "[DCI] starting vLLM ${SERVED_MODEL_NAME} on ${CUDA_VISIBLE_DEVICES}; log=${VLLM_LOG}"
scripts/serve_vllm_qwen38.sh >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
log "vLLM process launched; pid=${VLLM_PID}"

echo "[DCI] waiting for http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
for attempt in $(seq 1 "${WAIT_ATTEMPTS}"); do
  if curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    log "vLLM ready after attempt=${attempt}"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[DCI] vLLM exited early; tailing log"
    tail -n 200 "${VLLM_LOG}" || true
    exit 1
  fi
  if (( attempt == 1 || attempt % 12 == 0 )); then
    log "vLLM not ready yet; attempt=${attempt}/${WAIT_ATTEMPTS} elapsed_seconds=$((attempt * WAIT_SECONDS)) pid=${VLLM_PID}"
    tail -n 20 "${VLLM_LOG}" || true
  fi
  sleep "${WAIT_SECONDS}"
done

if ! curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "[DCI] vLLM readiness timeout; tailing log"
  tail -n 200 "${VLLM_LOG}" || true
  exit 1
fi

log "stage=bench-env begin; activating ${BENCH_ENV}"
conda activate "${BENCH_ENV}"
log "stage=bench-env activated; python=$(command -v python)"
# FIRST=1
for QUERY_ID in ${QUERY_IDS}; do
  echo "[DCI] phase3 ${TASK}/${QUERY_ID}"
  PREPARE=()
  # if [[ "${FIRST}" == "1" ]]; then
  #   PREPARE=(--prepare)
  #   FIRST=0
  # fi
  python scripts/phase3_single_sample.py \
    "${PREPARE[@]}" \
    --task "${TASK}" \
    --query-id "${QUERY_ID}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --vllm-base-url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --output-dir "${OUTPUT_DIR}"
done

log "done; results=${OUTPUT_DIR}/${TASK}; vllm_log=${VLLM_LOG}; run_log=${RUN_LOG}"
