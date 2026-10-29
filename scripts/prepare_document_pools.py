#!/usr/bin/env python3
"""Prepare immutable bounded document candidate-pool workspaces.

This command is an offline data transformation.  It reads an existing
prepared suite and, for large corpora, a saved original-ID ranking export.  It
never loads an embedding model or performs retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Direct ``python scripts/prepare_document_pools.py`` execution puts only the
# scripts directory on sys.path; add the repository root before importing the
# package so the documented command works without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dci_bench.data.document_pools import (
    POOL_SIZES,
    prepare_document_pools,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-suite",
        "--source-root",
        dest="source_suite",
        type=Path,
        required=True,
        help="existing prepared suite containing workspaces/<task>/corpus and metadata",
    )
    parser.add_argument(
        "--output-suite",
        "--output-root",
        dest="output_suite",
        type=Path,
        required=True,
        help="new suite root; an existing path is never overwritten",
    )
    parser.add_argument("--task", required=True)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--pool-size", choices=tuple(POOL_SIZES))
    size.add_argument("--capacity", type=int, choices=tuple(POOL_SIZES.values()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rankings",
        "--retrieval-rankings",
        dest="rankings",
        type=Path,
        help="saved original-ID retrieval ranking JSON (required when corpus exceeds capacity)",
    )
    parser.add_argument("--source-qrels", type=Path)
    parser.add_argument("--source-doc-ids", type=Path)
    parser.add_argument("--source-workspace", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_document_pools(
        args.source_suite,
        args.output_suite,
        args.task,
        pool_size=args.pool_size,
        capacity=args.capacity,
        seed=args.seed,
        rankings_path=args.rankings,
        source_qrels=args.source_qrels,
        source_doc_ids=args.source_doc_ids,
        source_workspace=args.source_workspace,
    )
    print(
        json.dumps(
            {
                "suite_root": str(result.suite_root),
                "task": result.task,
                "metadata_dir": str(result.metadata_dir),
                "workspace_dir": str(result.workspace_dir),
                "candidate_pools": str(result.candidate_pools_path),
                "completion": str(result.completion_path),
                "num_queries": result.num_queries,
                "corpus_size": result.corpus_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
