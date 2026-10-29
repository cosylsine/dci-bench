#!/usr/bin/env python3
"""Prepare auditable DCI data; model inference is always a separate explicit stage."""
from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.core import (
    DEFAULT_EVAL_ROOT, DEFAULT_RAW_ROOT, build_source, code_identity, digest,
    export_dci, load_bundle, load_config, read_json, slug, verify_release,
    write_dashboard,
)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/data_pipeline/sources.json")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="Show adapters, configured sources, and their readiness")
    sub.add_parser("doctor", help="Read-only check of paths, dependencies, and local model files")
    for name in ("fetch", "build"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--source", nargs="+", required=True, help="Explicit source IDs from the catalog")
        cmd.add_argument("--raw-root", type=Path)
        if name == "build":
            cmd.add_argument("--eval-root", type=Path)
            cmd.add_argument("--release-name", default="v1")
            cmd.add_argument("--snapshot", type=Path, help="Existing raw snapshot (single source only)")
    cmd = sub.add_parser("verify")
    cmd.add_argument("release", type=Path)
    cmd = sub.add_parser("audit")
    cmd.add_argument("release", type=Path)
    cmd = sub.add_parser("report")
    cmd.add_argument("--releases", nargs="+", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd = sub.add_parser("export")
    cmd.add_argument("--release", type=Path, required=True)
    cmd.add_argument("--output-suite", type=Path, required=True)
    cmd.add_argument("--task", required=True)
    cmd.add_argument("--allow-automatic-labels", action="store_true")
    cmd = sub.add_parser("pools")
    cmd.add_argument("--source-suite", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd.add_argument("--task", required=True)
    cmd.add_argument("--capacities", nargs="+", type=int, choices=[200, 500, 1000], default=[200, 500, 1000])
    cmd.add_argument("--rankings", type=Path)
    cmd.add_argument("--seed", type=int, default=20261021)
    cmd = sub.add_parser("estimate", help="Plan model call counts without launching models")
    cmd.add_argument("--release", type=Path, required=True)
    cmd.add_argument("--generation-documents", type=int, default=20)
    cmd.add_argument("--judged-documents-per-query", type=int, default=50)
    cmd = sub.add_parser("mine-review", help="Union lexical/dense candidates for adjudication")
    cmd.add_argument("--release", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd.add_argument("--top-k", type=int, default=20)
    cmd.add_argument("--dense-files", nargs="*", type=Path, default=[])
    cmd = sub.add_parser("partition-languages", help="Publish explicit monolingual views")
    cmd.add_argument("--release", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    for name in ("generate", "judge"):
        cmd = sub.add_parser(name, help="Explicit bounded local-model inference")
        cmd.add_argument("--release", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--models-config", type=Path, default=ROOT / "configs/data_pipeline/models.json")
        cmd.add_argument("--stage-config", type=Path, required=True)
        cmd.add_argument("--cache-dir", type=Path, required=True)
        cmd.add_argument("--max-calls", type=int, default=50)
        cmd.add_argument("--max-total-tokens", type=int, default=200000)
        if name == "judge":
            cmd.add_argument("--candidates", type=Path, required=True)
    cmd = sub.add_parser("rank", help="Compute local dense scores, including every seed positive")
    group = cmd.add_mutually_exclusive_group(required=True)
    group.add_argument("--release", type=Path)
    group.add_argument("--source-suite", type=Path)
    cmd.add_argument("--task")
    cmd.add_argument("--output", type=Path, required=True)
    cmd.add_argument("--cache-dir", type=Path, required=True)
    cmd.add_argument("--model-path")
    cmd.add_argument("--audit-model", action="store_true", help="Unprompted BGE-style audit encoding; not official pools")
    cmd.add_argument("--device", default="cpu")
    cmd.add_argument("--batch-size", type=int, default=8)
    cmd.add_argument("--max-length", type=int, default=8192)
    cmd.add_argument("--top-k", type=int, default=1000)
    cmd.add_argument("--query-prompt")
    cmd.add_argument("--document-prompt")
    cmd.add_argument("--cache-shard-size", type=int, default=0)
    cmd.add_argument("--search-device", default="cpu")
    cmd.add_argument("--search-query-batch-size", type=int, default=32)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "partition-languages":
        from dci_bench.data.pipeline.stages import partition_languages
        emit(partition_languages(args.release, args.output))
        return
    if args.command == "mine-review":
        from dci_bench.data.pipeline.stages import mine_review
        emit(mine_review(args.release, args.output, top_k=args.top_k, dense_files=args.dense_files))
        return
    if args.command in ("generate", "judge"):
        from dci_bench.data.pipeline.stages import run_model_stage
        emit(run_model_stage(args.release, args.output,
                             role="generator" if args.command == "generate" else "judge",
                             models_config=args.models_config, cache_dir=args.cache_dir,
                             stage_config=read_json(args.stage_config), max_calls=args.max_calls,
                             max_total_tokens=args.max_total_tokens, candidates=getattr(args, "candidates", None)))
        return
    if args.command == "rank":
        from dci_bench.data.pipeline.stages import rank_source
        emit(rank_source(args.output, suite=args.source_suite, task=args.task, release=args.release,
                         cache_dir=args.cache_dir, model_path=args.model_path, device=args.device,
                         batch_size=args.batch_size, max_length=args.max_length, top_k=args.top_k,
                         audit_model=args.audit_model, query_prompt=args.query_prompt,
                         document_prompt=args.document_prompt, cache_shard_size=args.cache_shard_size,
                         search_device=args.search_device,
                         search_query_batch_size=args.search_query_batch_size))
        return
    if args.command == "verify":
        emit(verify_release(args.release))
        return
    if args.command == "audit":
        from dci_bench.data.pipeline.quality import audit_bundle
        emit(audit_bundle(load_bundle(args.release)))
        return
    if args.command == "report":
        emit(write_dashboard(args.releases, args.output))
        return
    if args.command == "export":
        emit(export_dci(args.release, args.output_suite, args.task,
                        allow_automatic_labels=args.allow_automatic_labels))
        return
    if args.command == "pools":
        from dci_bench.data.document_pools import prepare_document_pools
        if len(set(args.capacities)) != len(args.capacities):
            raise ValueError("Duplicate capacities")
        for capacity in args.capacities:
            result = prepare_document_pools(args.source_suite, args.output / str(capacity), args.task,
                                            capacity=capacity, seed=args.seed, rankings_path=args.rankings)
            emit({"capacity": capacity, "output": str(result.suite_root), "queries": result.num_queries})
        return
    if args.command == "estimate":
        if min(args.generation_documents, args.judged_documents_per_query) <= 0:
            raise ValueError("Planning counts must be positive")
        m = verify_release(args.release)
        q = m["num_queries"] or min(m["num_documents"], args.generation_documents)
        emit({"boundary": "call-count plan; no inference or GPU allocation",
              "documents": m["num_documents"], "existing_queries": m["num_queries"],
              "generation_document_calls": 0 if m["num_queries"] else q,
              "planned_judgment_pairs": q * min(args.judged_documents_per_query, m["num_documents"]),
              "exhaustive_judgment_pairs": q * m["num_documents"],
              "tokens_and_throughput": "measure on a small model smoke before projecting cost"})
        return
    config = load_config(args.config)
    if args.command == "catalog":
        from dci_bench.data.pipeline.connectors import list_connectors
        emit({"connectors": list_connectors(), "sources": config["sources"]})
        return
    if args.command == "doctor":
        models = read_json(ROOT / "configs/data_pipeline/models.json")
        emit({"python": sys.executable,
              "roots": {k: str(config.get(k, d)) for k, d in (("raw_root", DEFAULT_RAW_ROOT), ("eval_root", DEFAULT_EVAL_ROOT))},
              "dependencies": {n: bool(importlib.util.find_spec(n)) for n in
                               ("jsonschema", "pyarrow", "numpy", "sentence_transformers", "pytest")},
              "embedding_environment": {
                  "verified_python": "/mnt/afs/250010100/miniconda/envs/retriever/bin/python",
                  "required_transformers_major": 4,
                  "current_transformers_version": importlib.metadata.version("transformers") if importlib.util.find_spec("transformers") else None,
                  "reason": "Pinned KaLM custom Qwen2 code is incompatible with Transformers 5"},
              "models": {role: {"path": value.get("path"), "local_path_exists": Path(value.get("path", "/not-configured")).is_dir()}
                         for role, value in models["roles"].items()},
              "code_identity": code_identity(), "gpu_validation": "not performed by doctor"})
        return
    sources = {s["id"]: s for s in config["sources"]}
    if len(set(args.source)) != len(args.source):
        raise ValueError("Duplicate requested sources")
    unknown = set(args.source) - set(sources)
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    raw = args.raw_root or Path(config.get("raw_root", DEFAULT_RAW_ROOT))
    if args.command == "build":
        clean = args.eval_root or Path(config.get("eval_root", DEFAULT_EVAL_ROOT))
        from dci_bench.data.pipeline.core import check_roots
        check_roots(raw, clean)
        slug(args.release_name)
        if args.snapshot and len(args.source) != 1:
            raise ValueError("--snapshot requires exactly one source")
    for name in args.source:
        spec = sources[name]
        if args.command == "fetch":
            from dci_bench.data.pipeline.connectors import fetch_source
            emit({"source": name, "snapshot": str(fetch_source(spec, raw))})
        else:
            output = clean / "pipeline_v1" / name / args.release_name
            if output.exists():
                manifest = verify_release(output)
                if manifest["specification"] != spec or manifest["code_identity"] != code_identity():
                    raise ValueError(f"Existing release has different configuration/code; use a new --release-name: {output}")
                emit({"source": name, "verified_existing": str(output), "state": manifest["state"]})
            else:
                result = build_source(spec, raw, output, snapshot=args.snapshot,
                                      policy=config.get("quality_policy"))
                emit({"source": name, "release": str(output), **result})


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"data-pipeline: {exc}", file=sys.stderr)
        raise SystemExit(2)
