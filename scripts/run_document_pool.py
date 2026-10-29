#!/usr/bin/env python3
"""Run or CPU-preflight a prepared document pool with its declared tier profile.

Pass --task, --metadata-root, --workspace-root and a query selection as for
run_task.py. --preflight-only verifies data/routing without a model endpoint.
Other options are forwarded to run_task.py; normal runs require a backend
manifest with a 65536-token context window.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pool_profile(pool: dict[str, object]) -> tuple[str, str]:
    pool_size = pool.get("pool_size")
    if not isinstance(pool_size, str):
        raise ValueError("candidate-pool tier must be a string")
    if pool_size in {"small", "medium", "large"}:
        return "document-pool-v1-" + pool_size, pool_size
    if pool_size in {"lite", "full"}:
        return "document-pool-v2-" + pool_size, pool_size
    if pool_size in {"document-pool-v2-lite", "document-pool-v2-full"}:
        return pool_size, pool_size.removeprefix("document-pool-v2-")
    raise ValueError("candidate-pool tier must be small, medium, large, lite or full")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    try:
        if not args.task or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.task):
            raise ValueError("unsafe task name")
        # Select the profile before importing any task, adapter or contracts.
        metadata = args.metadata_root / args.task
        pool = json.loads((metadata / "candidate_pools.json").read_text(encoding="utf-8"))
        if not isinstance(pool, dict):
            raise ValueError("candidate-pool metadata must be an object")
        profile, tier_name = _pool_profile(pool)
        explicit_profile = next(
            (pool[key] for key in ("protocol_profile", "pool_profile", "profile") if key in pool),
            None,
        )
        if explicit_profile is not None:
            if not isinstance(explicit_profile, str):
                raise ValueError("candidate-pool profile must be a string")
            equivalent = {profile, tier_name}
            if profile.startswith("document-pool-v1-"):
                equivalent.add(profile.removeprefix("document-pool-v1-"))
            else:
                equivalent.add(profile.removeprefix("document-pool-v2-"))
            if explicit_profile not in equivalent:
                raise ValueError("candidate-pool profile conflicts with its tier")
        existing_profile = os.environ.get("DCI_PROTOCOL_PROFILE")
        if existing_profile is not None:
            equivalent = {profile, tier_name}
            if profile.startswith("document-pool-v1-"):
                equivalent.add(profile.removeprefix("document-pool-v1-"))
            else:
                equivalent.add(profile.removeprefix("document-pool-v2-"))
            if existing_profile not in equivalent:
                raise ValueError("DCI_PROTOCOL_PROFILE conflicts with the prepared pool tier")
        if "dci_bench.protocol.contracts" in sys.modules:
            raise ValueError("run_document_pool must select the profile in a fresh process")
        os.environ["DCI_PROTOCOL_PROFILE"] = profile
        from dci_bench.protocol.document_pools import get_tier
        from dci_bench.data.registry import TASKS, TaskSpec, validate_task_manifest

        tier = get_tier(profile)
        if pool["capacity"] != tier["capacity"]:
            raise ValueError("candidate-pool capacity does not match its tier")
        manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
        task = TaskSpec(args.task, manifest["dataset"], manifest["revision"], args.task,
                        manifest.get("task_instruction", ""))
        validate_task_manifest(task, manifest)
        if args.task in TASKS:
            validate_task_manifest(TASKS[args.task], manifest)
        else:
            TASKS[args.task] = task

        from scripts import run_task
        runner_argv = ["--task", args.task, "--metadata-root", str(args.metadata_root),
                       "--workspace-root", str(args.workspace_root), *forwarded]
        if not args.preflight_only:
            return run_task.main(runner_argv)

        options = run_task.parse_args(runner_argv)
        if options.prepare:
            raise ValueError("pool preflight cannot prepare or rebuild source data")
        from dci_bench.results.phase4_runner import data_audit
        from dci_bench.tasks.mteb_llm_retrieval import _make_samples
        from dci_bench.protocol.contracts import CONTRACT_SHA256, protocol_contract
        from dci_bench.results.protocol import build_run_manifest

        identity = data_audit(args.metadata_root, args.task)
        samples = _make_samples(args.task, options.query_ids, args.metadata_root,
                                args.workspace_root, all_queries=options.all_queries)
        # Validate the data/contract through the same publication schema used by
        # execute_phase4, before allocating a model server or making model calls.
        build_run_manifest(
            run_id='data-preflight', model_key=options.model_key, task=args.task,
            query_ids=[sample.id for sample in samples], data=identity,
            benchmark_contract=protocol_contract(task_name=args.task, task_instruction=task.task_instruction),
            max_total_model_tokens=tier['max_total_model_tokens'], max_concurrency=options.max_concurrency,
        )
        result = {"status": "data_preflight_passed", "task": args.task,
                  "profile": tier["profile"], "capacity": tier["capacity"],
                  "max_total_model_tokens": tier["max_total_model_tokens"],
                  "context_window": tier["context_window"],
                  "contract_sha256": CONTRACT_SHA256, "data": identity,
                  "run_manifest_data_validated": True,
                  "query_ids": [sample.id for sample in samples],
                  "model_calls": 0, "gpu_inference_started": False,
                  "sandbox_verified": False}
        if tier["schema_version"] == "document-pool-v2":
            result.update({key: tier[key] for key in (
                "max_search_turns", "max_tool_invocations", "max_final_repair_turns",
                "max_agent_steps", "tool_result_max_bytes", "sample_timeout_seconds",
            )})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[DCI] Document-pool configuration/integrity failure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
