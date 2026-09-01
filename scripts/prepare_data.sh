#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-LLMPublicHealthQA}"

python -m dci_bench.data.workspace_builder \
  --task "$TASK" \
  --overwrite \
  --sample-queries 5
