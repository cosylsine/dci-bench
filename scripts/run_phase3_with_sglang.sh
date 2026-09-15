#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[DCI] run_phase3_with_sglang.sh is a compatibility alias for the Phase 4 batch wrapper." >&2
exec "${REPO_ROOT}/scripts/run_phase4_with_sglang.sh" "$@"
