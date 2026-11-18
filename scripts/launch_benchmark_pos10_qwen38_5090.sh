#!/usr/bin/env bash
# New positive-count-limited suite, with the previous formal Frozen settings.
set -Eeuo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export BENCHMARK_SUITE="${BENCHMARK_SUITE:-/mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_v2_20261116_pos10_v1}"
export CONCURRENCY="${CONCURRENCY:-8}"
export DCI_RUNTIME_CONFIG="${DCI_RUNTIME_CONFIG:-$REPO_DIR/configs/runtime/frozen-debian13.json}"
export DCI_RUNTIME_BUNDLE="${DCI_RUNTIME_BUNDLE:-$REPO_DIR/results/frozen-runtime/20261111-xaDmpt/release-v2/rootfs.tar}"
export PYTHONDONTWRITEBYTECODE=1
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python - "$BENCHMARK_SUITE" <<'PY'
import json
from pathlib import Path
import sys

suite = Path(sys.argv[1])
if not (suite / 'benchmark-manifest.json').is_file():
    raise SystemExit(f'New suite is not complete; finish source pooling and suite acceptance first: {suite}')
config = json.loads((suite / 'config.json').read_text())
expected = {
    'version': 'positive-count-1-to-10-v1',
    'min_positive_count': 1, 'max_positive_count': 10,
    'positive_definition': 'canonical_qrel_score_gt_zero',
    'overflow_policy': 'exclude_query_before_sampling',
}
if config.get('query_eligibility') != expected:
    raise SystemExit('This launcher requires the frozen positive-count-1-to-10 policy')
PY
exec bash scripts/launch_benchmark20_qwen38_5090.sh "$@"
