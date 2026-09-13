#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_SH="${CONDA_SH:-/mnt/afs/250010100/miniconda/etc/profile.d/conda.sh}"
BENCH_ENV="${BENCH_ENV:-dci-bench}"

TASK="${TASK:-LLMPublicHealthQA}"
QUERY_IDS="${QUERY_IDS:-Q71 Q72}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs2/202608/embedding_models/dci-bench/pretrained_models/MiniCPM5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniCPM5-2B}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:30000/v1}"
OPENAI_SERVICE="${OPENAI_SERVICE:-sglang}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-180}"
WAIT_SECONDS="${WAIT_SECONDS:-5}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/phase3_sglang_minicpm5_2b}"
TASK_OUTPUT_DIR="${TASK_OUTPUT_DIR:-${OUTPUT_DIR}/${TASK}}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${TASK_OUTPUT_DIR}/_runs/${RUN_ID}}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

if [[ ! -r "${CONDA_SH}" ]]; then
  echo "[DCI] missing CONDA_SH=${CONDA_SH}" >&2
  exit 2
fi
if [[ ! "${WAIT_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[DCI] WAIT_ATTEMPTS and WAIT_SECONDS must be positive integers" >&2
  exit 2
fi
read -r -a QUERY_ID_ARRAY <<<"${QUERY_IDS}"
if (( ${#QUERY_ID_ARRAY[@]} == 0 )); then
  echo "[DCI] QUERY_IDS must contain at least one query id" >&2
  exit 2
fi
if [[ -e "${RUN_OUTPUT_DIR}" && "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
  echo "[DCI] run artifact directory already exists: ${RUN_OUTPUT_DIR}; choose a new RUN_ID or set ALLOW_EXISTING_OUTPUT=1" >&2
  exit 2
fi
if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
  for QUERY_ID in "${QUERY_ID_ARRAY[@]}"; do
    SAMPLE_OUTPUT_DIR="${TASK_OUTPUT_DIR}/${QUERY_ID}"
    if [[ -e "${SAMPLE_OUTPUT_DIR}/final.json" || -e "${SAMPLE_OUTPUT_DIR}/trace.json" ]]; then
      echo "[DCI] canonical result already exists under ${SAMPLE_OUTPUT_DIR}; set ALLOW_EXISTING_OUTPUT=1 to replace it" >&2
      exit 2
    fi
  done
fi

OPENAI_BASE_URL="${OPENAI_BASE_URL%/}"
MODELS_URL="${OPENAI_BASE_URL}/models"
CHAT_COMPLETIONS_URL="${OPENAI_BASE_URL}/chat/completions"
PHASE3_OUTPUT_DIR="${PHASE3_OUTPUT_DIR:-${OUTPUT_DIR}}"
LOG_DIR="${LOG_DIR:-${RUN_OUTPUT_DIR}/inspect-logs}"
SUMMARY_DIR="${SUMMARY_DIR:-${RUN_OUTPUT_DIR}/summaries}"
MODELS_RESPONSE_PATH="${MODELS_RESPONSE_PATH:-${RUN_OUTPUT_DIR}/models.json}"
TOOL_SMOKE_REQUEST_PATH="${TOOL_SMOKE_REQUEST_PATH:-${RUN_OUTPUT_DIR}/tool-parser-smoke-request.json}"
TOOL_SMOKE_RESPONSE_PATH="${TOOL_SMOKE_RESPONSE_PATH:-${RUN_OUTPUT_DIR}/tool-parser-smoke-response.json}"
RUN_SUMMARY_PATH="${RUN_SUMMARY_PATH:-${RUN_OUTPUT_DIR}/run-summary.json}"

mkdir -p "${TASK_OUTPUT_DIR}" "${RUN_OUTPUT_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}"

source "${CONDA_SH}"
# Conda compiler packages may install deactivate hooks that read unset backup
# variables. Temporarily disable nounset while switching away from the serving
# environment, then restore the script's strict mode.
set +u
conda activate "${BENCH_ENV}"
set -u
# Inspect and SQLite in this environment are linked against Conda's newer
# libstdc++; keep this process self-contained without changing the environment.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if ! command -v curl >/dev/null 2>&1; then
  echo "[DCI] curl is required for SGLang readiness and tool-call checks" >&2
  exit 2
fi
CURL_BIN="$(command -v curl)"
host_curl() {
  # The benchmark environment's libstdc++ is required by Python, but injecting
  # all of CONDA_PREFIX/lib into the system curl can mix incompatible libffi
  # and p11-kit builds. Run curl with the host dynamic-library search path.
  env -u LD_LIBRARY_PATH "${CURL_BIN}" "$@"
}

export TASK QUERY_IDS
python - <<'PY'
import json
import os
import sys
from pathlib import Path

task = os.environ["TASK"]
query_ids = os.environ["QUERY_IDS"].split()
workspace = Path("data/workspaces") / task
metadata = Path("data/metadata") / task
required = (workspace / "README.md", metadata / "queries.jsonl", metadata / "qrels.json")
missing = [str(path) for path in required if not path.is_file()]
if missing:
    print(f"[DCI] missing prepared workspace or metadata: {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(2)

queries = {
    json.loads(line)["_id"]
    for line in (metadata / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
}
qrels = json.loads((metadata / "qrels.json").read_text(encoding="utf-8"))
missing_ids = [query_id for query_id in query_ids if query_id not in queries or not qrels.get(query_id)]
if missing_ids:
    print(f"[DCI] unknown or unscored query ids: {', '.join(missing_ids)}", file=sys.stderr)
    raise SystemExit(2)
print(f"[DCI] workspace preflight ok: task={task}; query_ids={' '.join(query_ids)}")
PY

models_ready() {
  host_curl -fsS "${MODELS_URL}" -o "${MODELS_RESPONSE_PATH}" || return 1
  python - "${MODELS_RESPONSE_PATH}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
ids = [str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)]
if expected not in ids:
    print(f"[DCI] /v1/models did not advertise {expected!r}; got {ids!r}", file=sys.stderr)
    raise SystemExit(1)
PY
}

echo "[DCI] waiting for existing SGLang endpoint ${MODELS_URL}"
READY=0
for _ in $(seq 1 "${WAIT_ATTEMPTS}"); do
  if models_ready; then
    READY=1
    break
  fi
  sleep "${WAIT_SECONDS}"
done
if [[ "${READY}" != "1" ]]; then
  echo "[DCI] SGLang readiness timeout at ${MODELS_URL}" >&2
  exit 1
fi
echo "[DCI] SGLang is ready: ${SERVED_MODEL_NAME}"

python - "${SERVED_MODEL_NAME}" "${TOOL_SMOKE_REQUEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model_name, output_path = sys.argv[1:]
payload = {
    "model": model_name,
    "messages": [
        {
            "role": "user",
            "content": "Call get_weather for Beijing now. Do not answer in prose.",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "temperature": 0,
    "max_tokens": 128,
}
Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "[DCI] checking MiniCPM5 tool-call parser"
if ! host_curl -fsS \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -X POST "${CHAT_COMPLETIONS_URL}" \
  --data-binary "@${TOOL_SMOKE_REQUEST_PATH}" \
  -o "${TOOL_SMOKE_RESPONSE_PATH}"; then
  echo "[DCI] tool-call smoke request failed; response=${TOOL_SMOKE_RESPONSE_PATH}" >&2
  exit 1
fi
python - "${TOOL_SMOKE_RESPONSE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
choices = payload.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit("[DCI] tool-call smoke response has no choices")
choice = choices[0]
message = choice.get("message") if isinstance(choice, dict) else None
tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
if not isinstance(tool_calls, list) or not tool_calls:
    raise SystemExit("[DCI] tool-call smoke response did not contain parsed tool_calls")
first = tool_calls[0]
function = first.get("function") if isinstance(first, dict) else None
if not isinstance(function, dict) or function.get("name") != "get_weather":
    raise SystemExit("[DCI] tool-call smoke response returned an unexpected function")
arguments = function.get("arguments")
try:
    parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
except json.JSONDecodeError as exc:
    raise SystemExit(f"[DCI] tool-call smoke arguments are not JSON: {exc}") from exc
if not isinstance(parsed_arguments, dict) or not parsed_arguments:
    raise SystemExit("[DCI] tool-call smoke arguments are not a non-empty JSON object")
if choice.get("finish_reason") != "tool_calls":
    raise SystemExit(f"[DCI] tool-call smoke finish_reason={choice.get('finish_reason')!r}, expected 'tool_calls'")
print("[DCI] tool-call smoke passed")
PY

FAILED_QUERY_IDS=()
for QUERY_ID in "${QUERY_ID_ARRAY[@]}"; do
  SUMMARY_PATH="${SUMMARY_DIR}/${QUERY_ID}.json"
  echo "[DCI] phase3 ${TASK}/${QUERY_ID}"
  if python scripts/phase3_single_sample.py \
    --task "${TASK}" \
    --query-id "${QUERY_ID}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --openai-base-url "${OPENAI_BASE_URL}" \
    --openai-service "${OPENAI_SERVICE}" \
    --api-key "${OPENAI_API_KEY}" \
    --log-dir "${LOG_DIR}" \
    --output-dir "${PHASE3_OUTPUT_DIR}" \
    --summary-path "${SUMMARY_PATH}" \
    --require-valid-output; then
    echo "[DCI] phase3 ${TASK}/${QUERY_ID} passed"
  else
    echo "[DCI] phase3 ${TASK}/${QUERY_ID} failed; continuing with remaining queries" >&2
    FAILED_QUERY_IDS+=("${QUERY_ID}")
  fi
done

export SUMMARY_DIR TASK SERVED_MODEL_NAME OPENAI_BASE_URL OPENAI_SERVICE TOOL_SMOKE_RESPONSE_PATH
python - "${RUN_SUMMARY_PATH}" "${QUERY_ID_ARRAY[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

output_path = Path(sys.argv[1])
query_ids = sys.argv[2:]
summary_dir = Path(os.environ["SUMMARY_DIR"])
per_query = {}
for query_id in query_ids:
    path = summary_dir / f"{query_id}.json"
    if path.is_file():
        try:
            per_query[query_id] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            per_query[query_id] = {"valid_output": False, "failure_reason": f"invalid summary JSON: {exc}"}
    else:
        per_query[query_id] = {"valid_output": False, "failure_reason": "summary was not produced"}

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": os.environ["TASK"],
    "served_model_name": os.environ["SERVED_MODEL_NAME"],
    "openai_base_url": os.environ["OPENAI_BASE_URL"],
    "openai_service": os.environ["OPENAI_SERVICE"],
    "tool_smoke_response_path": os.environ["TOOL_SMOKE_RESPONSE_PATH"],
    "per_query": per_query,
    "valid_outputs": sum(bool(item.get("valid_output")) for item in per_query.values()),
    "invalid_outputs": sum(not bool(item.get("valid_output")) for item in per_query.values()),
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[DCI] run summary: {output_path}")
PY

if (( ${#FAILED_QUERY_IDS[@]} > 0 )); then
  echo "[DCI] invalid outputs: ${FAILED_QUERY_IDS[*]}; results retained under ${TASK_OUTPUT_DIR}" >&2
  exit 1
fi

echo "[DCI] all samples passed; results=${TASK_OUTPUT_DIR}; run_artifacts=${RUN_OUTPUT_DIR}"
