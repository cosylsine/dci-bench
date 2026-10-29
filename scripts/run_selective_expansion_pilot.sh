#!/usr/bin/env bash
set -Eeuo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
export LD_LIBRARY_PATH="/mnt/afs/250010100/miniconda/envs/dci-bench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ "${DRY_RUN:-0}" != 1 && " $* " != *" --dry-run "* ]]; then
  log_dir="$repo/results/Qwen3.8-27B/job-logs"
  mkdir -p "$log_dir"
  log="$(mktemp "$log_dir/expansion-pilot-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")"
  exec > >(tee "$log") 2>&1
  echo "[pilot] driver log: $log"
  missing=()
  [[ -x /usr/bin/bwrap ]] || missing+=(bubblewrap)
  [[ -x /usr/bin/rg ]] || missing+=(ripgrep)
  if (( ${#missing[@]} )); then
    if [[ "$(id -u)" != 0 ]] || ! command -v apt-get >/dev/null; then
      echo 'Node image must provide /usr/bin/bwrap and /usr/bin/rg.' >&2
      exit 2
    fi
    # System APT helpers must not load Conda's libffi/libstdc++ dependencies.
    env -u LD_LIBRARY_PATH -u LD_PRELOAD apt-get update -o Acquire::Retries=3
    env -u LD_LIBRARY_PATH -u LD_PRELOAD DEBIAN_FRONTEND=noninteractive \
      apt-get install -y --no-install-recommends "${missing[@]}"
  fi
fi
exec "$python" -B "$repo/scripts/run_selective_expansion_pilot.py" "$@"
