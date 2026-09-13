#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Keep the GPU dynamic-library selection explicit so this script behaves the
# same way in an interactive shell and in a batch job.
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_SH="${CONDA_SH:-/mnt/afs/250010100/miniconda/etc/profile.d/conda.sh}"
SGLANG_ENV="${SGLANG_ENV:-dci-sglang-minicpm5}"

MODEL_PATH="${MODEL_PATH:-/mnt/afs2/202608/embedding_models/dci-bench/pretrained_models/MiniCPM5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniCPM5-2B}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
DTYPE="${DTYPE:-bfloat16}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-minicpm5}"
# The environment's FlashInfer sampling extension may require a runtime JIT
# build. Use SGLang's native PyTorch sampler by default so serving does not
# depend on an exactly matched nvcc/header wheel pair.
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
# An optional path (or executable name) for the compiler used by SGLang's
# runtime CUDA JIT. SGLang's local source forwards CXX to nvcc with -ccbin and
# also uses it for the corresponding host link step.
SGLANG_JIT_CXX="${SGLANG_JIT_CXX:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -r "${CONDA_SH}" ]]; then
  echo "[DCI] missing CONDA_SH=${CONDA_SH}" >&2
  exit 2
fi
if [[ ! "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[DCI] TP_SIZE must be a positive integer, got ${TP_SIZE@Q}" >&2
  exit 2
fi
if [[ ! "${CONTEXT_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[DCI] CONTEXT_LENGTH must be a positive integer, got ${CONTEXT_LENGTH@Q}" >&2
  exit 2
fi
if [[ ! "${SGLANG_PORT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[DCI] SGLANG_PORT must be a positive integer, got ${SGLANG_PORT@Q}" >&2
  exit 2
fi

# Conda compiler activation hooks may read unset optional variables.
set +u
source "${CONDA_SH}"
conda activate "${SGLANG_ENV}"
set -u

resolve_compiler() {
  local candidate="$1"

  if [[ -z "${candidate}" ]]; then
    return 1
  fi
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || return 1
    printf '%s\n' "${candidate}"
  else
    command -v "${candidate}" 2>/dev/null
  fi
}

resolve_nvcc() {
  local resolved configured

  configured="${CUDA_HOME:-${CUDA_PATH:-}}"
  if [[ -n "${configured}" ]]; then
    [[ -x "${configured}/bin/nvcc" ]] || return 1
    printf '%s\n' "${configured}/bin/nvcc"
    return 0
  fi
  resolved="$(python - <<'PY'
import sysconfig
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
if (root / "bin" / "nvcc").is_file():
    print(root / "bin" / "nvcc")
PY
)"
  if [[ -n "${resolved}" && -x "${resolved}" ]]; then
    printf '%s\n' "${resolved}"
    return 0
  fi
  resolved="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "${resolved}" ]]; then
    printf '%s\n' "${resolved}"
    return 0
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    printf '%s\n' /usr/local/cuda/bin/nvcc
    return 0
  fi
  return 1
}

NVCC_BIN="$(resolve_nvcc || true)"
if [[ -z "${NVCC_BIN}" ]]; then
  echo "[DCI] cannot find nvcc; SGLang needs it to JIT-compile MiniCPM5 kernels." >&2
  exit 2
fi

# Keep preflight and all runtime JIT backends on the same toolkit.
export CUDA_HOME="$(dirname "$(dirname "${NVCC_BIN}")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "[DCI] CUDA toolkit: ${CUDA_HOME}"

select_cuda_jit_compiler() {
  local candidate resolved

  for candidate in "$@"; do
    [[ -n "${candidate}" ]] || continue
    resolved="$(resolve_compiler "${candidate}" || true)"
    [[ -n "${resolved}" ]] || continue

    # This verifies both requirements that caused the startup failure: the
    # host compiler understands C++20 <concepts>, and this nvcc accepts it.
    if printf '#include <concepts>\nint main() { return 0; }\n' | \
      "${NVCC_BIN}" -std=c++20 -ccbin "${resolved}" -x cu -E -o /dev/null - >/dev/null 2>&1; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done
  return 1
}

if [[ -n "${SGLANG_JIT_CXX}" ]]; then
  if ! SELECTED_JIT_CXX="$(select_cuda_jit_compiler "${SGLANG_JIT_CXX}")"; then
    echo "[DCI] SGLANG_JIT_CXX=${SGLANG_JIT_CXX@Q} cannot compile SGLang's C++20 CUDA JIT probe." >&2
    exit 2
  fi
else
  if ! SELECTED_JIT_CXX="$(select_cuda_jit_compiler \
    "${CUDAHOSTCXX:-}" \
    "${CXX:-}" \
    "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" \
    "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" \
    "${CONDA_PREFIX}/bin/g++" \
    g++-13 g++-12 g++-11 g++-10 g++ clang++ c++)"; then
    cat >&2 <<'EOF'
[DCI] SGLang's CUDA JIT needs a host compiler that supports -std=c++20 and <concepts>.
[DCI] No suitable compiler was found before model loading.
[DCI] Set SGLANG_JIT_CXX to a compatible g++ path, or install GCC 12 in this environment:
      conda install -y -c conda-forge "gxx_linux-64=12"
EOF
    exit 2
  fi
fi

export CXX="${SELECTED_JIT_CXX}"
JIT_COMPILER_VERSION="$("${SELECTED_JIT_CXX}" --version 2>&1 | sed -n '1p')"
echo "[DCI] JIT host compiler: ${SELECTED_JIT_CXX} (${JIT_COMPILER_VERSION})"

export MODEL_PATH TOOL_CALL_PARSER
python - <<'PY'
import json
import os
import sys
from pathlib import Path

model_path = Path(os.environ["MODEL_PATH"])
required_files = ("config.json", "tokenizer_config.json", "chat_template.jinja")
missing = [name for name in required_files if not (model_path / name).is_file()]
weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
if missing or not weight_files:
    details = []
    if missing:
        details.append(f"missing files: {', '.join(missing)}")
    if not weight_files:
        details.append("no .safetensors or .bin weight file found")
    print(f"[DCI] invalid MODEL_PATH={model_path}: {'; '.join(details)}", file=sys.stderr)
    raise SystemExit(2)

from sglang.srt.function_call.function_call_parser import FunctionCallParser

parser_name = os.environ["TOOL_CALL_PARSER"]
if parser_name not in FunctionCallParser.ToolCallParserEnum:
    available = ", ".join(sorted(FunctionCallParser.ToolCallParserEnum))
    print(
        f"[DCI] SGLang does not register tool parser {parser_name!r}; available: {available}",
        file=sys.stderr,
    )
    raise SystemExit(2)

config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
print(
    "[DCI] SGLang preflight ok: "
    f"model_type={config.get('model_type')} "
    f"architectures={config.get('architectures')} "
    f"parser={parser_name}"
)
PY

CMD=(
  sglang serve
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${SGLANG_HOST}"
  --port "${SGLANG_PORT}"
  --tp-size "${TP_SIZE}"
  --context-length "${CONTEXT_LENGTH}"
  --dtype "${DTYPE}"
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --sampling-backend "${SAMPLING_BACKEND}"
)
if [[ -n "${SGLANG_EXTRA_ARGS}" ]]; then
  read -r -a USER_EXTRA_ARGS <<<"${SGLANG_EXTRA_ARGS}"
  CMD+=("${USER_EXTRA_ARGS[@]}")
fi

echo "[DCI] serving ${SERVED_MODEL_NAME} on ${SGLANG_HOST}:${SGLANG_PORT}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; TP_SIZE=${TP_SIZE}"
printf '[DCI] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

exec "${CMD[@]}"
