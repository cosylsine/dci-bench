#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EMBED_PY="/mnt/afs/250010100/miniconda/envs/lychee_reranker_5090_kalm451/bin/python"

if [[ ! -x "$EMBED_PY" ]]; then
  printf 'required Python is not executable: %s\n' "$EMBED_PY" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# The scheduler may provide a physical-device mask.  When it does not, use
# the first four physical devices; the Python preflight still requires exactly
# four RTX 5090/sm120 devices.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ "${DRY_RUN:-0}" == "1" || " $* " == *" --check-only "* ]]; then
  exec "$EMBED_PY" "$REPO_DIR/scripts/run_gapfill_5090_4gpu.py" "$@"
fi

# Keep the controller's errors as well as the existing per-worker logs.
LOG_DIR="${DCI_GAPFILL_LOG_DIR:-$REPO_DIR/results/gapfill-controller-logs}"
mkdir -p "$LOG_DIR"
LOG_PATH="$(mktemp "$LOG_DIR/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")"
printf 'gapfill controller log: %s\n' "$LOG_PATH"
set +e
"$EMBED_PY" "$REPO_DIR/scripts/run_gapfill_5090_4gpu.py" "$@" 2>&1 | tee "$LOG_PATH"
PIPE_CODES=("${PIPESTATUS[@]}")
set -e
printf 'gapfill controller exit_code=%s log=%s\n' "${PIPE_CODES[0]}" "$LOG_PATH" | tee -a "$LOG_PATH"
if [[ "${PIPE_CODES[0]}" != "0" ]]; then
  exit "${PIPE_CODES[0]}"
fi
exit "${PIPE_CODES[1]}"
