#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  set -- --all-tasks --refresh-existing
elif [[ "$1" != --* ]]; then
  task="$1"
  shift
  set -- --task "$task" "$@"
fi

python -m dci_bench.data.workspace_builder \
  "$@" \
  --sample-queries "${SAMPLE_QUERIES:-5}"
