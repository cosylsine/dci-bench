#!/usr/bin/env bash
set -Eeuo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
unset LD_LIBRARY_PATH LD_PRELOAD
export PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$python"):$PATH"
# This launcher re-executes an immutable, previously sealed source bundle.
# Frozen RootFS evaluations use the current v2/task entrypoints instead.
if [[ -n "${DCI_RUNTIME_CONFIG:-}" ]]; then
  echo '[smoke] This legacy sealed-bundle launcher is host-tools only. Use run_benchmark_v2.py or run_document_pool.py --runtime-config with a new output directory.' >&2
  exit 2
fi
cpu_data_only=false
for arg in "$@"; do
  case "$arg" in
    --runtime-config|--runtime-config=*)
      echo '[smoke] Frozen RootFS is not supported by this legacy sealed-bundle launcher; use the current v2/task entrypoint.' >&2
      exit 2 ;;
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
