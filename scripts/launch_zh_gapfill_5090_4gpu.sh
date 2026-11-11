#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ZH_DATA="${ZH_DATA_ROOT:-/mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v4}"
EMBED_PY="/mnt/afs/250010100/miniconda/envs/lychee_reranker_5090_kalm451/bin/python"
CPU_ENV="/mnt/afs/250010100/miniconda/envs/dci-bench"
BASE_SUITE="/mnt/afs2/202609/embedding_models/dci-bench/eval_data/benchmark_v2_20261102_v2"
BASE_CATALOG="/mnt/afs2/202609/embedding_models/dci-bench/eval_data/benchmark_gapfill_20261102_v2/catalog.json"

# Source preparation must finish before any GPU process is launched.
"$EMBED_PY" "$REPO_DIR/scripts/prepare_zh_gapfill_20261102.py" check --data-root "$ZH_DATA"
LD_LIBRARY_PATH="$CPU_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$CPU_ENV/bin/python" "$REPO_DIR/scripts/run_benchmark_v2.py" --help >/dev/null
bash "$REPO_DIR/scripts/launch_gapfill_5090_4gpu.sh" \
  --run-root "$ZH_DATA/run" \
  --state-root "$ZH_DATA/run/.zh_5090_4gpu_state" \
  --source gapfill_t2ranking_software_zh \
  --source gapfill_cfqa_enterprise_zh "$@"

if [[ "${DRY_RUN:-0}" == "1" || " $* " == *" --check-only "* ]]; then
  exit 0
fi

"$EMBED_PY" "$REPO_DIR/scripts/accept_zh_gapfill_20261102.py" \
  --data-root "$ZH_DATA" \
  --output-suite "$ZH_DATA/suites/benchmark_v2_20261102_zh_v1" \
  --evidence-dir "$ZH_DATA/evidence/suite-acceptance" \
  --config "$BASE_SUITE/config.json" \
  --base-catalog "$BASE_CATALOG" --base-suite "$BASE_SUITE"

LD_LIBRARY_PATH="$CPU_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$CPU_ENV/bin/python" "$REPO_DIR/scripts/run_benchmark_v2.py" \
  --suite "$ZH_DATA/suites/benchmark_v2_20261102_zh_v1" \
  --split smoke --allow-partial --preflight-only \
  --output "$ZH_DATA/evidence/data-preflight"

LD_LIBRARY_PATH="$CPU_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$CPU_ENV/bin/python" "$REPO_DIR/scripts/publish_zh_gapfill_report_20261102.py" \
  --data-root "$ZH_DATA"
