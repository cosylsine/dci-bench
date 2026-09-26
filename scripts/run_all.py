#!/usr/bin/env python
"""Run all six pinned DCI-Bench retrieval tasks sequentially."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dci_bench.data.registry import TASK_NAMES, get_task, validate_task_manifest
from dci_bench.scoring.retrieval import DEFAULT_METRIC_KS, normalize_metric_ks


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("must be in TCP port range 1..65535")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="MiniCPM5-2B")
    parser.add_argument("--backend-manifest", type=Path)
    parser.add_argument(
        "--backend-artifact",
        action="append",
        type=Path,
        default=[],
        help="backend diagnostic file copied into every task run (repeatable)",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--workspace-root", type=Path, default=Path("data/workspaces"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/metadata"))
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument("--run-id")
    runs.add_argument("--resume-run", metavar="RUN_ID")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--bridge-port", type=_port, default=18891)
    parser.add_argument("--max-concurrency", type=_positive_int, default=1)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_NAMES,
        metavar="TASK",
        help="run only these registry tasks (default: all six)",
    )
    parser.add_argument("--launcher-preflight-seconds", type=_nonnegative_float)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate all six prepared tasks without invoking a model backend",
    )
    parser.add_argument(
        "--metric-ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_METRIC_KS),
        metavar="K",
    )
    return parser.parse_args(argv)


def task_bridge_port(args: argparse.Namespace, task_index: int) -> int:
    """Return the base of a task-specific, non-overlapping bridge port range."""

    port = args.bridge_port + task_index * args.max_concurrency
    if port + args.max_concurrency - 1 > 65535:
        raise ValueError(
            "Phase 5 bridge port ranges exceed TCP port 65535: "
            f"base={args.bridge_port}, tasks={len(TASK_NAMES)}, "
            f"max_concurrency={args.max_concurrency}"
        )
    return port


def task_command(args: argparse.Namespace, task_name: str, task_index: int = 0) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_task.py")),
        "--model-key",
        args.model_key,
        "--task",
        task_name,
        "--all-queries",
        "--backend-manifest",
        str(args.backend_manifest),
        "--results-root",
        str(args.results_root),
        "--workspace-root",
        str(args.workspace_root),
        "--metadata-root",
        str(args.metadata_root),
        "--api-key",
        args.api_key,
        "--bridge-port",
        str(task_bridge_port(args, task_index)),
        "--max-concurrency",
        str(args.max_concurrency),
        "--metric-ks",
        *(str(value) for value in normalize_metric_ks(args.metric_ks)),
    ]
    if args.run_id is not None:
        command.extend(("--run-id", args.run_id))
    if args.resume_run is not None:
        command.extend(("--resume-run", args.resume_run))
    if args.launcher_preflight_seconds is not None:
        command.extend(
            ("--launcher-preflight-seconds", str(args.launcher_preflight_seconds))
        )
    for artifact in args.backend_artifact:
        command.extend(("--backend-artifact", str(artifact)))
    return command


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_process_group(pgid: int, *, grace_seconds: float = 5.0) -> None:
    """Clean up only descendants of one isolated task process group."""

    if not _process_group_exists(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return
        time.sleep(0.1)
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_task_process(command: Sequence[str], *, cwd: Path, check: bool = False):
    """Run a task in its own session so leaked bridge children can be contained."""

    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return_code = process.wait()
    finally:
        _stop_process_group(process.pid)
    if check and return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return subprocess.CompletedProcess(command, return_code)


def preflight_all_tasks(args: argparse.Namespace) -> None:
    """Validate all six workspaces and manifests before the first model call."""

    for task_name in TASK_NAMES:
        task = get_task(task_name)
        workspace = args.workspace_root / task_name / "corpus"
        metadata = args.metadata_root / task_name
        required = (
            workspace,
            metadata / "manifest.json",
            metadata / "queries.jsonl",
            metadata / "qrels.json",
            metadata / "doc_id_files.json",
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Phase 5 preflight missing files for {task_name}: "
                + ", ".join(str(path) for path in missing)
            )
        manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"data manifest must be an object: {metadata / 'manifest.json'}")
        validate_task_manifest(task, manifest)
        query_ids: list[str] = []
        seen: set[str] = set()
        for line_number, line in enumerate(
            (metadata / "queries.jsonl").read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            query = json.loads(line)
            query_id = query.get("_id") if isinstance(query, dict) else None
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"invalid query id at {task_name}/queries.jsonl:{line_number}")
            if query_id in seen:
                raise ValueError(f"duplicate query id {task_name}/{query_id}")
            if not isinstance(query.get("text"), str):
                raise ValueError(f"query {task_name}/{query_id} has no text")
            seen.add(query_id)
            query_ids.append(query_id)
        qrels = json.loads((metadata / "qrels.json").read_text(encoding="utf-8"))
        doc_id_files = json.loads(
            (metadata / "doc_id_files.json").read_text(encoding="utf-8")
        )
        if not isinstance(qrels, dict) or not isinstance(doc_id_files, dict):
            raise ValueError(f"qrels and doc_id_files must be objects for {task_name}")
        missing_qrels = [
            query_id
            for query_id in query_ids
            if not isinstance(qrels.get(query_id), dict) or not qrels[query_id]
        ]
        if missing_qrels:
            raise ValueError(f"missing qrels for {task_name}: {missing_qrels[:5]}")
        unknown_docs = sorted(
            {
                str(doc_id)
                for query_id in query_ids
                for doc_id in qrels[query_id]
                if str(doc_id) not in doc_id_files
            }
        )
        if unknown_docs:
            raise ValueError(f"qrels reference unknown docs for {task_name}: {unknown_docs[:5]}")
        corpus_files = list(workspace.glob("*.md"))
        if len(query_ids) != manifest.get("num_queries"):
            raise ValueError(f"query count does not match manifest for {task_name}")
        if len(corpus_files) != manifest.get("num_docs") or len(doc_id_files) != len(corpus_files):
            raise ValueError(f"corpus count does not match manifest for {task_name}")


def run_all(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., Any] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Run registry tasks in order, continuing past completed benchmark failures.

    A task exit code of 1 means its immutable run completed with invalid or
    failed samples, so the remaining tasks still run.  Configuration or
    integrity failures stop the sequence before additional model work.
    """

    if run_command is None:
        run_command = _run_task_process
    preflight_all_tasks(args)
    task_bridge_port(args, len(TASK_NAMES) - 1)
    selected_tasks = TASK_NAMES if args.tasks is None else tuple(args.tasks)
    if len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("--tasks contains duplicate task names")
    repo_root = Path(__file__).resolve().parents[1]
    records: list[dict[str, Any]] = []
    overall_exit_code = 0
    for ordinal, task_name in enumerate(selected_tasks, start=1):
        task_index = TASK_NAMES.index(task_name)
        bridge_port = task_bridge_port(args, task_index)
        print(
            f"[DCI] Phase 5 task {ordinal}/{len(selected_tasks)}: {task_name}; "
            f"bridge_ports={bridge_port}-{bridge_port + args.max_concurrency - 1}",
            flush=True,
        )
        completed = run_command(
            task_command(args, task_name, task_index),
            cwd=repo_root,
            check=False,
        )
        task_exit_code = int(completed.returncode)
        records.append({"task": task_name, "exit_code": task_exit_code})
        if task_exit_code == 1:
            overall_exit_code = 1
            continue
        if task_exit_code != 0:
            return 2, records
    return overall_exit_code, records


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.preflight_only:
            preflight_all_tasks(args)
            print(
                json.dumps(
                    {
                        "schema_version": "dci-phase5-preflight-v1",
                        "tasks": list(TASK_NAMES),
                        "status": "ok",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.backend_manifest is None:
            raise ValueError("--backend-manifest is required unless --preflight-only is used")
        exit_code, records = run_all(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[DCI] Phase 5 configuration/integrity failure: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": "dci-phase5-run-all-summary-v1",
                "tasks": records,
                "exit_code": exit_code,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
