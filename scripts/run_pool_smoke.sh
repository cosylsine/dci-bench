#!/usr/bin/env bash
set -Eeuo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$python"):$PATH"
cpu_data_only=false
for arg in "$@"; do
  case "$arg" in
    --seal|--dry-run|--data-preflight-only) cpu_data_only=true ;;
  esac
done
if [[ "${DRY_RUN:-0}" != 1 && "$cpu_data_only" == false ]]; then
  if [[ -n "${DCI_POOL_SMOKE_LOG_DIR:-}" ]]; then
    mkdir -p "$DCI_POOL_SMOKE_LOG_DIR"
    driver_log="$(mktemp "$DCI_POOL_SMOKE_LOG_DIR/pool-smoke-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")"
    exec > >(tee "$driver_log") 2>&1
    echo "[smoke] Cluster bootstrap and driver log: $driver_log"
  fi
  bash "$repo/scripts/ensure_pool_smoke_tools.sh"
fi
export LD_LIBRARY_PATH=/mnt/afs/250010100/miniconda/envs/dci-bench/lib
exec "$python" -B "$repo/scripts/run_pool_smoke.py" "$@"
