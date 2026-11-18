#!/usr/bin/env bash
# Pool the two new legal sources, then freeze all 20 resampled tasks.
set -Eeuo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATA_ROOT="${LEGAL_POS10_DATA_ROOT:-/mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_legal_pos10_20261116_v1}"
SUITE="${BENCHMARK_SUITE:-/mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_v2_20261116_pos10_v1}"
CPU_ENV=/mnt/afs/250010100/miniconda/envs/dci-bench
cpu=(env LD_LIBRARY_PATH="$CPU_ENV/lib" PYTHONDONTWRITEBYTECODE=1 "$CPU_ENV/bin/python")
export PYTHONDONTWRITEBYTECODE=1
case "$*" in ''|--check-only) ;; *) echo 'Usage: DRY_RUN=0 bash scripts/launch_legal_pos10_5090_4gpu.sh [--check-only]' >&2; exit 2 ;; esac

# Downloads and source conversion must already be complete. These checks are read-only.
"${cpu[@]}" scripts/prepare_privacyqa_pos10.py --data-root "$DATA_ROOT" --check-only
"${cpu[@]}" scripts/prepare_stard_pos10.py --data-root "$DATA_ROOT" --check-only
DRY_RUN="${DRY_RUN:-1}" bash scripts/launch_gapfill_5090_4gpu.sh \
  --run-root "$DATA_ROOT/run" --state-root "$DATA_ROOT/run/.legal_pos10_5090_state" \
  --source gapfill_privacyqa_legal_en --source gapfill_stard_legal_zh "$@"
if [[ "${DRY_RUN:-1}" != 0 || " $* " == *" --check-only "* ]]; then
  exit 0
fi

if [[ -e "$SUITE" ]]; then
  "${cpu[@]}" scripts/prepare_benchmark_v2.py audit --output "$SUITE"
  "${cpu[@]}" - "$SUITE" <<'PY'
import json
from pathlib import Path
import sys
suite = Path(sys.argv[1])
if json.loads((suite / 'config.json').read_text()) != json.loads(Path('configs/benchmark_v2.json').read_text()):
    raise SystemExit('Existing suite has another frozen config; choose a new BENCHMARK_SUITE')
if not json.loads((suite / 'manifest.json').read_text())['ready_for_formal']:
    raise SystemExit('Existing suite is incomplete; preserve it and use a new BENCHMARK_SUITE')
PY
else
  "${cpu[@]}" scripts/prepare_benchmark_pos10.py freeze --output "$SUITE" \
    --legal-en-catalog "$DATA_ROOT/run/gapfill_privacyqa_legal_en/catalog-row.json" \
    --legal-zh-catalog "$DATA_ROOT/run/gapfill_stard_legal_zh/catalog-row.json"
fi
printf 'New 20-task suite is frozen and audited: %s\n' "$SUITE"
printf 'Next: DRY_RUN=0 bash scripts/launch_benchmark_pos10_qwen38_5090.sh --full\n'
