#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-${JOB_ID:-${LSB_JOBID:-${PBS_JOBID:-manual-$$}}}}"
JOB_TAG="${JOB_TAG//\//_}"
RUN_LOG="${RUN_LOG:-${REPO_ROOT}/results/job-logs/phase4-sglang-${JOB_TAG}-${RUN_TIMESTAMP}.log}"
mkdir -p "$(dirname "${RUN_LOG}")"
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

CONDA_SH="${CONDA_SH:-/mnt/afs/250010100/miniconda/etc/profile.d/conda.sh}"
BENCH_ENV="${BENCH_ENV:-dci-bench}"
TASK="${TASK:-LLMPublicHealthQA}"
MODEL_KEY="${MODEL_KEY:-MiniCPM5-2B}"
QUERY_IDS="${QUERY_IDS:-Q71 Q72}"
ALL_QUERIES="${ALL_QUERIES:-0}"
METADATA_ROOT="${METADATA_ROOT:-data/metadata}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
BRIDGE_PORT="${BRIDGE_PORT:-13131}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RUN_ID="${RUN_ID:-}"
BACKEND_MANIFEST="${BACKEND_MANIFEST:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-180}"
WAIT_SECONDS="${WAIT_SECONDS:-5}"

if [[ ! -r "${CONDA_SH}" ]]; then
  log "missing CONDA_SH=${CONDA_SH}"
  exit 2
fi
if [[ -z "${BACKEND_MANIFEST}" || ! -r "${BACKEND_MANIFEST}" ]]; then
  log "BACKEND_MANIFEST must point to the dci-backend-manifest-v1 emitted by serve_sglang_minicpm5.sh"
  exit 2
fi
if [[ ! "${MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${BRIDGE_PORT}" =~ ^[1-9][0-9]*$ ]]; then
  log "MAX_CONCURRENCY and BRIDGE_PORT must be positive integers"
  exit 2
fi
if [[ "${ALL_QUERIES}" != "0" && "${ALL_QUERIES}" != "1" ]]; then
  log "ALL_QUERIES must be 0 or 1"
  exit 2
fi
if ! command -v bwrap >/dev/null 2>&1; then
  log "trusted /usr/bin/bwrap is required for the corpus-only tool sandbox"
  exit 2
fi

PREFLIGHT_STARTED="$(date +%s.%N)"
log "phase4 wrapper start task=${TASK} model_key=${MODEL_KEY} max_concurrency=${MAX_CONCURRENCY}"
log "backend_manifest=${BACKEND_MANIFEST} run_log=${RUN_LOG}"

set +u
source "${CONDA_SH}"
conda activate "${BENCH_ENV}"
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

(
  cd pi-dci
  npm exec -- tsx scripts/dci-run.ts \
    --preflight-only \
    --workspace "${REPO_ROOT}/data/workspaces/${TASK}"
)

MANIFEST_VALUES="$(python - "${BACKEND_MANIFEST}" "${MODEL_KEY}" <<'PY'
import sys
from pathlib import Path

from dci_bench.backends.manifest import load_backend_manifest

manifest = load_backend_manifest(Path(sys.argv[1]))
expected_key = sys.argv[2]
if manifest["model"]["model_key"] != expected_key:
    raise SystemExit(
        f"backend manifest model_key={manifest['model']['model_key']!r}, expected {expected_key!r}"
    )
print(manifest["backend"]["base_url"])
print(manifest["model"]["served_model_name"])
print(manifest["backend"]["openai_service"])
PY
)"
mapfile -t MANIFEST_ARRAY <<<"${MANIFEST_VALUES}"
OPENAI_BASE_URL="${MANIFEST_ARRAY[0]%/}"
SERVED_MODEL_NAME="${MANIFEST_ARRAY[1]}"
OPENAI_SERVICE="${MANIFEST_ARRAY[2]}"
MODELS_URL="${OPENAI_BASE_URL}/models"
CHAT_COMPLETIONS_URL="${OPENAI_BASE_URL}/chat/completions"

# Cluster/container proxy settings are not always standards-compliant (for
# example, NO_PROXY may contain a host:port entry).  The SGLang wrapper is a
# loopback service, so make the bypass explicit for curl and every child
# process, including Inspect and the Pi bridge.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

# Resolve the exact selection before readiness or tool-parser smoke performs
# any model request. This calls the same host-side selector as run_task.py, so
# duplicate, unknown, and qrels-less IDs fail closed with identical semantics.
QUERY_ID_ARRAY=()
if [[ "${ALL_QUERIES}" != "1" ]]; then
  read -r -a QUERY_ID_ARRAY <<<"${QUERY_IDS}"
fi
if ! python - "${TASK}" "${METADATA_ROOT}" "${ALL_QUERIES}" "${QUERY_ID_ARRAY[@]}" <<'PY'
import sys
from pathlib import Path

from dci_bench.tasks.mteb_llm_retrieval import _select_query_ids

task, metadata_root, all_queries, *query_ids = sys.argv[1:]
selected = _select_query_ids(
    task,
    metadata_root=Path(metadata_root),
    query_ids=None if all_queries == "1" else query_ids,
    all_queries=all_queries == "1",
)
print(f"[DCI] query selection preflight passed: {len(selected)} sample(s)")
PY
then
  log "query selection preflight failed"
  exit 2
fi

DIAGNOSTIC_DIR="$(mktemp -d -t dci-phase4-sglang.XXXXXXXX)"
cleanup() {
  if [[ -n "${DIAGNOSTIC_DIR:-}" && -d "${DIAGNOSTIC_DIR}" ]]; then
    rm -r -- "${DIAGNOSTIC_DIR}"
  fi
}
trap cleanup EXIT
MODELS_RESPONSE_PATH="${DIAGNOSTIC_DIR}/models.json"
TOOL_SMOKE_REQUEST_PATH="${DIAGNOSTIC_DIR}/tool-parser-smoke-request.json"
TOOL_SMOKE_RESPONSE_PATH="${DIAGNOSTIC_DIR}/tool-parser-smoke-response.json"

if ! command -v curl >/dev/null 2>&1; then
  log "curl is required for endpoint preflight"
  exit 2
fi
CURL_BIN="$(command -v curl)"
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
  host_curl -fsS "${MODELS_URL}" -o "${MODELS_RESPONSE_PATH}" || return 1
  format_json_file "${MODELS_RESPONSE_PATH}" || return 1
  python - "${MODELS_RESPONSE_PATH}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
model_ids = [str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)]
if expected not in model_ids:
    raise SystemExit(f"expected model {expected!r}; endpoint advertised {model_ids!r}")
PY
}

log "waiting for SGLang endpoint ${MODELS_URL}"
READY=0
for _ in $(seq 1 "${WAIT_ATTEMPTS}"); do
  if models_ready; then
    READY=1
    break
  fi
  sleep "${WAIT_SECONDS}"
done
if [[ "${READY}" != "1" ]]; then
  log "endpoint readiness timed out"
  exit 2
fi

python - "${SERVED_MODEL_NAME}" "${TOOL_SMOKE_REQUEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model, path = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Call get_weather for Beijing now. Do not answer in prose."}],
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
print("[DCI] non-greedy tool parser smoke passed")
PY

PREFLIGHT_SECONDS="$(python - "${PREFLIGHT_STARTED}" <<'PY'
import sys
import time
print(max(0.0, time.time() - float(sys.argv[1])))
PY
)"

RUNNER_ARGS=(
  python scripts/run_task.py
  --model-key "${MODEL_KEY}"
  --task "${TASK}"
  --backend-manifest "${BACKEND_MANIFEST}"
  --backend-artifact "${MODELS_RESPONSE_PATH}"
  --backend-artifact "${TOOL_SMOKE_REQUEST_PATH}"
  --backend-artifact "${TOOL_SMOKE_RESPONSE_PATH}"
  --results-root "${RESULTS_ROOT}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --bridge-port "${BRIDGE_PORT}"
  --api-key "${OPENAI_API_KEY}"
  --launcher-preflight-seconds "${PREFLIGHT_SECONDS}"
)
if [[ -n "${RUN_ID}" ]]; then
  RUNNER_ARGS+=(--run-id "${RUN_ID}")
fi
if [[ "${ALL_QUERIES}" == "1" ]]; then
  RUNNER_ARGS+=(--all-queries)
else
  if (( ${#QUERY_ID_ARRAY[@]} == 0 )); then
    log "QUERY_IDS must contain at least one query id"
    exit 2
  fi
  RUNNER_ARGS+=(--query-ids "${QUERY_ID_ARRAY[@]}")
fi

log "starting one Inspect eval for task=${TASK} service=${OPENAI_SERVICE}"
"${RUNNER_ARGS[@]}"
