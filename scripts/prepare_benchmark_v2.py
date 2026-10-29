#!/usr/bin/env python3
"""Prepare or audit the metadata-only benchmark v2 suite.

Examples::

    python scripts/prepare_benchmark_v2.py prepare \
      --catalog /mnt/afs2/202609/embedding_models/dci-bench/eval_data/retrieval_pools/catalog.json \
      --output /mnt/afs2/202609/embedding_models/dci-bench/eval_data/benchmark_v2_20261029

    python scripts/prepare_benchmark_v2.py audit --output /path/to/benchmark_v2_20261029

The prepare command only reads source manifests, compact pool metadata, and
the catalog.  It writes selected query/pool metadata plus references to the
existing shared corpus; document bodies are never copied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.benchmark_v2 import (  # noqa: E402
    DEFAULT_SEED,
    audit_benchmark_v2,
    load_benchmark_config,
    prepare_benchmark_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare a new immutable metadata suite")
    prepare.add_argument("--catalog", type=Path, required=True)
    prepare.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "benchmark_v2.json",
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)

    audit = subparsers.add_parser("audit", help="audit an existing v2 suite")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--config", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_benchmark_v2(
            output_dir=args.output,
            config_path=args.config,
            catalog_path=args.catalog,
            seed=args.seed,
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.config is not None:
        load_benchmark_config(args.config)
    result = audit_benchmark_v2(args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

