#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_SH="${CONDA_SH:-/mnt/afs/250010100/miniconda/etc/profile.d/conda.sh}"
VLLM_ENV="${VLLM_ENV:-dci-vllm-qwen38}"
BENCH_ENV="${BENCH_ENV:-dci-bench}"

MODEL_PATH="${MODEL_PATH:-/mnt/afs/share/Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
QUERY_IDS="${QUERY_IDS:-Q71 Q72}"
TASK="${TASK:-LLMPublicHealthQA}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase3_minicpm5_2b}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-240}"
WAIT_SECONDS="${WAIT_SECONDS:-5}"

source "${CONDA_SH}"

mkdir -p results/vllm-logs
VLLM_LOG="${VLLM_LOG:-results/vllm-logs/${SERVED_MODEL_NAME//\//_}-$(date +%Y%m%d-%H%M%S).log}"

export MODEL_PATH SERVED_MODEL_NAME VLLM_HOST VLLM_PORT TENSOR_PARALLEL_SIZE MAX_MODEL_LEN
export TOOL_CALL_PARSER REASONING_PARSER LANGUAGE_MODEL_ONLY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

conda activate "${VLLM_ENV}"
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
echo "[DCI] starting vLLM ${SERVED_MODEL_NAME} on ${CUDA_VISIBLE_DEVICES}; log=${VLLM_LOG}"
scripts/serve_model.sh >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "[DCI] stopping vLLM pid=${VLLM_PID}"
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[DCI] waiting for http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
for _ in $(seq 1 "${WAIT_ATTEMPTS}"); do
  if curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[DCI] vLLM ready"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[DCI] vLLM exited early; tailing log"
    tail -n 200 "${VLLM_LOG}" || true
    exit 1
  fi
  sleep "${WAIT_SECONDS}"
done

if ! curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "[DCI] vLLM readiness timeout; tailing log"
  tail -n 200 "${VLLM_LOG}" || true
  exit 1
fi

conda activate "${BENCH_ENV}"
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

echo "[DCI] done; results=${OUTPUT_DIR}/${TASK}; vllm_log=${VLLM_LOG}"
