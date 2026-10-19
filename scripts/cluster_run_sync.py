#!/usr/bin/env python3
"""Small, dependency-free synchronization protocol for a multi-node DCI run.

The launcher starts one foreground process on each scheduler node.  Rank zero
creates a fresh state directory, every rank records a heartbeat, and each rank
publishes exactly one terminal marker.  Terminal markers are created with a
same-filesystem hard-link so a stale marker can never be replaced silently.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping


SCHEDULER_ENV = (
    ("SLURM_JOB_ID", "slurm"),
    ("SENSECORE_JOB_ID", "sensecore"),
    ("SENSECORE_PYTORCH_JOB_ID", "sensecore"),
    ("JOB_ID", "generic"),
    ("LSB_JOBID", "lsf"),
    ("PBS_JOBID", "pbs"),
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
TERMINAL_STATUSES = frozenset({"success", "complete_with_sample_failures", "failure", "aborted"})
FAILURE_STATUSES = frozenset({"failure", "aborted"})
DEFAULT_INIT_TIMEOUT_SECONDS = 1800.0
DEFAULT_BARRIER_TIMEOUT_SECONDS = 172800.0
DEFAULT_HEARTBEAT_STALE_SECONDS = 1800.0


class SyncError(RuntimeError):
    """A bounded, user-actionable synchronization failure."""


def detect_scheduler(env: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """Return the first configured scheduler identity without requiring it."""

    source = os.environ if env is None else env
    for variable, scheduler in SCHEDULER_ENV:
        value = str(source.get(variable, "")).strip()
        if value:
            return {"scheduler": scheduler, "job_id": value, "env": variable}
    return {"scheduler": None, "job_id": None, "env": None}


def _safe_scheduler_id(value: str) -> str:
    # Scheduler array IDs commonly contain ``/`` or ``:``.  Keep the identity
    # readable while making it a safe result-protocol path component.
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise SyncError("scheduler job id does not contain a safe run-id component")
    return cleaned[:128]


def validate_run_id(value: str) -> str:
    value = str(value).strip()
    if not RUN_ID_RE.fullmatch(value):
        raise SyncError(
            "run id must be one safe path component matching "
            "[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
        )
    return value


def choose_run_id(
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    now: int | None = None,
    pid: int | None = None,
) -> tuple[str, str]:
    """Choose ``(run_id, source)`` using an explicit ID, then scheduler ID.

    A manual process has no shared identity by itself.  The generated value is
    therefore intentionally unique; multi-rank callers should pass
    ``DCI_RUN_ID`` when they are outside a scheduler allocation.
    """

    source = os.environ if env is None else env
    requested = explicit if explicit is not None else source.get("DCI_RUN_ID")
    if requested:
        return validate_run_id(requested), "DCI_RUN_ID"
    scheduler = detect_scheduler(source)
    if scheduler["job_id"]:
        return validate_run_id(_safe_scheduler_id(str(scheduler["job_id"]))), str(
            scheduler["env"]
        )
    timestamp = int(time.time() if now is None else now)
    process_id = os.getpid() if pid is None else int(pid)
    return validate_run_id(f"manual-{timestamp}-{process_id}"), "generated"


def _state(path: str | os.PathLike[str]) -> Path:
    value = Path(path)
    if value.name in {"", ".", ".."}:
        raise SyncError(f"invalid synchronization state directory: {path}")
    return value


def _positive_int(value: str, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SyncError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise SyncError(f"{name} must be a positive integer")
    return result


def _rank(args: argparse.Namespace) -> int:
    rank = _positive_int(str(args.rank + 1), "rank") - 1
    world_size = _positive_int(str(args.world_size), "world_size")
    if rank >= world_size:
        raise SyncError(f"rank {rank} is outside world size {world_size}")
    return rank


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError(f"invalid synchronization marker: {path}") from exc
    if not isinstance(value, dict):
        raise SyncError(f"synchronization marker is not an object: {path}")
    return value


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a marker and refuse to replace an existing marker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    data = (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            # link() is atomic and fails with EEXIST instead of replacing a
            # marker published by an earlier invocation.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SyncError(f"terminal marker already exists: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    data = (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _group_path(state: Path) -> Path:
    return state / "group.json"


def _load_group(state: Path) -> dict[str, Any]:
    path = _group_path(state)
    if not path.is_file():
        raise SyncError(f"synchronization group is not initialized: {path}")
    group = _read_json(path)
    world_size = group.get("world_size")
    if not isinstance(world_size, int) or world_size < 1:
        raise SyncError(f"invalid world size in synchronization group: {path}")
    return group


def _validate_group(group: Mapping[str, Any], args: argparse.Namespace) -> None:
    rank = _rank(args)
    if group.get("world_size") != int(args.world_size):
        raise SyncError(
            f"world size mismatch: group={group.get('world_size')} requested={args.world_size}"
        )
    requested_run_id = getattr(args, "run_id", None)
    if requested_run_id and group.get("run_id") != requested_run_id:
        raise SyncError(f"run id mismatch: group={group.get('run_id')} requested={requested_run_id}")
    requested_model_key = getattr(args, "model_key", None)
    if requested_model_key and group.get("model_key") != requested_model_key:
        raise SyncError(
            f"model key mismatch: group={group.get('model_key')} requested={requested_model_key}"
        )
    if group.get("rank_count") != int(args.world_size):
        raise SyncError("synchronization group rank count is inconsistent")
    if rank < 0:  # defensive; _rank already validates this
        raise SyncError("rank must be non-negative")


def _wait_for(path: Path, timeout: float, poll_seconds: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SyncError(f"timed out waiting for synchronization group: {path}")
        time.sleep(min(poll_seconds, remaining))


def init_group(args: argparse.Namespace) -> dict[str, Any]:
    state = _state(args.state_dir)
    rank = _rank(args)
    world_size = int(args.world_size)
    run_id = validate_run_id(args.run_id)
    if rank == 0:
        state.parent.mkdir(parents=True, exist_ok=True)
        try:
            state.mkdir()
        except FileExistsError as exc:
            raise SyncError(f"synchronization state already exists; refusing reuse: {state}") from exc
        group = {
            "schema_version": "dci-cluster-sync-v1",
            "run_id": run_id,
            "world_size": world_size,
            "rank_count": world_size,
            "model_key": str(args.model_key),
            "created_at": time.time(),
        }
        _write_once(_group_path(state), group)
    else:
        _wait_for(_group_path(state), float(args.timeout), float(args.poll_seconds))
        group = _load_group(state)
    group = _load_group(state)
    _validate_group(group, args)
    participant = {
        "rank": rank,
        "pid": os.getpid(),
        "tasks": list(args.tasks),
        "joined_at": time.time(),
    }
    participant_path = state / "participants" / f"rank-{rank}.json"
    if participant_path.exists():
        raise SyncError(f"participant marker already exists; refusing rank reuse: {participant_path}")
    _write_once(participant_path, participant)
    return group


def heartbeat(args: argparse.Namespace) -> None:
    state = _state(args.state_dir)
    group = _load_group(state)
    _validate_group(group, args)
    rank = _rank(args)
    _replace_json(
        state / "heartbeats" / f"rank-{rank}.json",
        {"rank": rank, "pid": os.getpid(), "at": time.time(), "state": args.state},
    )


def terminal(args: argparse.Namespace) -> dict[str, Any]:
    state = _state(args.state_dir)
    group = _load_group(state)
    _validate_group(group, args)
    rank = _rank(args)
    status = str(args.status)
    if status not in TERMINAL_STATUSES:
        raise SyncError(f"unknown terminal status: {status}")
    payload: dict[str, Any] = {
        "rank": rank,
        "pid": os.getpid(),
        "status": status,
        "exit_code": int(args.exit_code),
        "reason": str(args.reason),
        "tasks": list(args.tasks),
        "at": time.time(),
    }
    path = state / "terminal" / f"rank-{rank}.json"
    if path.exists():
        existing = _read_json(path)
        # Idempotent cleanup is useful for EXIT plus signal traps, but no
        # invocation is allowed to mutate the first terminal decision.
        if existing.get("rank") != rank:
            raise SyncError(f"terminal marker has the wrong rank: {path}")
        print(json.dumps(existing, ensure_ascii=False, sort_keys=True))
        return existing
    _write_once(path, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _terminal_markers(state: Path, world_size: int) -> dict[int, dict[str, Any]]:
    markers: dict[int, dict[str, Any]] = {}
    directory = state / "terminal"
    for rank in range(world_size):
        path = directory / f"rank-{rank}.json"
        if path.exists():
            marker = _read_json(path)
            if marker.get("rank") != rank or marker.get("status") not in TERMINAL_STATUSES:
                raise SyncError(f"invalid terminal marker: {path}")
            markers[rank] = marker
    return markers


def _status_payload(
    state: Path,
    world_size: int,
    *,
    heartbeat_stale_seconds: float = DEFAULT_HEARTBEAT_STALE_SECONDS,
) -> dict[str, Any]:
    group = _load_group(state)
    markers = _terminal_markers(state, world_size)
    now = time.time()
    group_created_at = group.get("created_at")
    if not isinstance(group_created_at, (int, float)):
        group_created_at = None
    heartbeats: dict[str, dict[str, Any]] = {}
    stale_ranks: list[int] = []
    for rank in range(world_size):
        if rank in markers:
            continue
        heartbeat_path = state / "heartbeats" / f"rank-{rank}.json"
        participant_path = state / "participants" / f"rank-{rank}.json"
        heartbeat = _read_json(heartbeat_path) if heartbeat_path.exists() else None
        participant = _read_json(participant_path) if participant_path.exists() else None
        observed_at = None
        observed_source = None
        if heartbeat is not None and isinstance(heartbeat.get("at"), (int, float)):
            observed_at = float(heartbeat["at"])
            observed_source = "heartbeat"
        elif participant is not None and isinstance(participant.get("joined_at"), (int, float)):
            observed_at = float(participant["joined_at"])
            observed_source = "participant"
        elif group_created_at is not None:
            # A rank that never joined must still fail the bounded startup
            # check.  Otherwise wait_terminal could sleep for its full
            # 48-hour terminal timeout without diagnosing the missing node.
            observed_at = float(group_created_at)
            observed_source = "group_created_at"
        if observed_at is None:
            continue
        age_seconds = max(0.0, now - observed_at)
        stale = age_seconds > heartbeat_stale_seconds
        heartbeats[str(rank)] = {
            "at": observed_at,
            "age_seconds": age_seconds,
            "stale": stale,
            "source": observed_source,
        }
        if stale:
            stale_ranks.append(rank)
    return {
        "complete": len(markers) == world_size,
        "failure": any(marker["status"] in FAILURE_STATUSES for marker in markers.values()),
        "terminal": {str(rank): marker for rank, marker in sorted(markers.items())},
        "missing_ranks": [rank for rank in range(world_size) if rank not in markers],
        "heartbeats": heartbeats,
        "stale_ranks": stale_ranks,
    }


def check_abort(args: argparse.Namespace) -> int:
    state = _state(args.state_dir)
    group = _load_group(state)
    _validate_group(group, args)
    payload = _status_payload(
        state,
        int(args.world_size),
        heartbeat_stale_seconds=float(args.heartbeat_timeout),
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["failure"] or payload["stale_ranks"] else 1


def wait_terminal(args: argparse.Namespace) -> int:
    state = _state(args.state_dir)
    group = _load_group(state)
    _validate_group(group, args)
    world_size = int(args.world_size)
    timeout = float(args.timeout)
    poll_seconds = float(args.poll_seconds)
    heartbeat_timeout = float(args.heartbeat_timeout)
    if timeout <= 0 or poll_seconds <= 0:
        raise SyncError("timeout and poll_seconds must be positive")
    if heartbeat_timeout <= 0:
        raise SyncError("heartbeat_timeout must be positive")
    deadline = time.monotonic() + timeout
    while True:
        payload = _status_payload(
            state,
            world_size,
            heartbeat_stale_seconds=heartbeat_timeout,
        )
        if payload["complete"]:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 1 if payload["failure"] else 0
        if payload["stale_ranks"]:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            raise SyncError(
                "heartbeat stale before terminal barrier; ranks="
                + ",".join(str(rank) for rank in payload["stale_ranks"])
            )
        if time.monotonic() >= deadline:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            raise SyncError(
                "terminal barrier timed out; missing ranks="
                + ",".join(str(rank) for rank in payload["missing_ranks"])
            )
        time.sleep(min(poll_seconds, deadline - time.monotonic()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    scheduler_parser = subparsers.add_parser("scheduler")
    scheduler_parser.add_argument("--run-id")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state-dir", required=True)
    common.add_argument("--world-size", required=True, type=int)
    common.add_argument("--rank", required=True, type=int)
    common.add_argument("--run-id", required=True)
    common.add_argument("--model-key", default="Qwen3.8-27B")
    common.add_argument("--tasks", nargs="*", default=[])
    common.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=DEFAULT_HEARTBEAT_STALE_SECONDS,
    )
    common.add_argument("--poll-seconds", type=float, default=1.0)

    init_parser = subparsers.add_parser("init", parents=[common])
    init_parser.add_argument("--timeout", type=float, default=DEFAULT_INIT_TIMEOUT_SECONDS)
    subparsers.add_parser("heartbeat", parents=[common]).add_argument("--state", default="running")
    terminal_parser = subparsers.add_parser("terminal", parents=[common])
    terminal_parser.add_argument("--status", required=True)
    terminal_parser.add_argument("--exit-code", type=int, default=0)
    terminal_parser.add_argument("--reason", default="")
    subparsers.add_parser("check-abort", parents=[common])
    wait_parser = subparsers.add_parser("wait", parents=[common])
    wait_parser.add_argument("--timeout", type=float, default=DEFAULT_BARRIER_TIMEOUT_SECONDS)
    subparsers.add_parser("status", parents=[common])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "scheduler":
            run_id, source = choose_run_id(args.run_id)
            print(
                json.dumps(
                    {**detect_scheduler(), "run_id": run_id, "run_id_source": source},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "init":
            print(json.dumps(init_group(args), ensure_ascii=False, sort_keys=True))
            return 0
        if args.action == "heartbeat":
            heartbeat(args)
            return 0
        if args.action == "terminal":
            terminal(args)
            return 0
        if args.action == "check-abort":
            return check_abort(args)
        if args.action == "wait":
            return wait_terminal(args)
        if args.action == "status":
            state = _state(args.state_dir)
            group = _load_group(state)
            _validate_group(group, args)
            print(json.dumps(_status_payload(state, int(args.world_size)), sort_keys=True))
            return 0
        raise SyncError(f"unknown action: {args.action}")
    except (OSError, SyncError, ValueError) as exc:
        print(f"[DCI] cluster synchronization error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
