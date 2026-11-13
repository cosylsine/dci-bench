#!/usr/bin/env bash
# Import/reuse the explicitly pinned RootFS and run the real no-model probe.
# This is an offline evaluation prerequisite; it never retrieves a rootfs.
set -Eeuo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="${DCI_RUNTIME_CONFIG:-}"
[[ -n "$config" ]] || {
  echo '[frozen-runtime] DCI_RUNTIME_CONFIG is required' >&2
  exit 2
}
config="$(realpath -e -- "$config")"
python="${DCI_BENCH_PYTHON:-/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python}"
[[ -x "$python" ]] || {
  echo "[frozen-runtime] benchmark Python is unavailable: $python" >&2
  exit 2
}

# A caller that is only planning must not unpack a bundle or mutate a cache.
if [[ "${DRY_RUN:-0}" != 0 ]]; then
  echo '[frozen-runtime] dry run: skipped offline prepare and sandbox probe'
  exit 0
fi

bundle_args=()
if [[ -n "${DCI_RUNTIME_BUNDLE:-}" ]]; then
  # Keep the caller's explicit path untouched. RuntimeManager validates it on
  # a cold cache, while a warm cache may legitimately outlive the source file.
  bundle_args=(--bundle "$DCI_RUNTIME_BUNDLE")
fi

# RuntimeManager reuses a valid warm cache. On a cold cache it fails closed
# unless the caller supplied this explicit local bundle; --offline prevents any
# registry or package-manager fallback. --preflight runs the bwrap tool probes.
if "$python" -m dci_bench.runtime prepare \
    --config "$config" --offline --preflight "${bundle_args[@]}"; then
  :
else
  status=$?
  if [[ -z "${DCI_RUNTIME_BUNDLE:-}" ]]; then
    echo '[frozen-runtime] if this is a cold cache, set DCI_RUNTIME_BUNDLE to the matching local rootfs.tar' >&2
  fi
  exit "$status"
fi

probe_args=("$python" "$repo/scripts/check_frozen_runtime.py" --runtime-config "$config")
if [[ -n "${DCI_FROZEN_RUNTIME_PROBE_OUTPUT:-}" ]]; then
  probe_args+=(--output "$DCI_FROZEN_RUNTIME_PROBE_OUTPUT")
fi
"${probe_args[@]}"
