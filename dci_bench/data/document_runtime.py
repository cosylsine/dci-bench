"""Host-side document identity checks and per-query corpus routing."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dci_bench.data.loader import stable_json_sha256


WORKSPACE_PREFETCH_ENV = "DCI_WORKSPACE_PREFETCH"
WORKSPACE_PREFETCH_BATCH_ENV = "DCI_WORKSPACE_PREFETCH_BATCH_SIZE"
WORKSPACE_PREFETCH_MAX_BYTES_ENV = "DCI_WORKSPACE_PREFETCH_MAX_BYTES"
_SHORT_DOCUMENT_ID = re.compile(r"d[0-9]{6,}\Z")


def _parse_bool_env(name: str, value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{name} must be one of 0/1/false/true (got {value!r})"
    )


def _parse_positive_env(name: str, value: str | None, *, default: int | None) -> int | None:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer (got {value!r})") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be zero or greater (got {parsed})")
    return None if parsed == 0 else parsed


@dataclass(frozen=True)
class WorkspacePrefetchConfig:
    """Run-scoped query workspace prefetch settings.

    ``max_bytes=None`` means that the in-flight workspace count is the only
    bound.  A positive byte cap reserves the declared document bytes before a
    copy starts, so a failed reservation cannot partially fill the scratch
    area.  The default is deliberately disabled for compatibility with the
    original per-sample materialization path.
    """

    enabled: bool
    max_concurrency: int
    batch_size: int
    max_bytes: int | None


def workspace_prefetch_config(max_concurrency: int, *, environ: dict[str, str] | None = None) -> WorkspacePrefetchConfig:
    """Resolve explicit prefetch configuration without process-global state."""

    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise ValueError("workspace prefetch max_concurrency must be an integer")
    if max_concurrency < 1:
        raise ValueError("workspace prefetch max_concurrency must be at least 1")
    values = os.environ if environ is None else environ
    enabled = _parse_bool_env(WORKSPACE_PREFETCH_ENV, values.get(WORKSPACE_PREFETCH_ENV))
    raw_batch = values.get(WORKSPACE_PREFETCH_BATCH_ENV)
    if raw_batch is None or not raw_batch.strip():
        batch_size = max_concurrency
    else:
        try:
            batch_size = int(raw_batch)
        except ValueError as exc:
            raise ValueError(
                f"{WORKSPACE_PREFETCH_BATCH_ENV} must be an integer (got {raw_batch!r})"
            ) from exc
    if batch_size < 1 or batch_size > max_concurrency:
        raise ValueError(
            f"{WORKSPACE_PREFETCH_BATCH_ENV} must be between 1 and max_concurrency "
            f"({max_concurrency})"
        )
    max_bytes = _parse_positive_env(
        WORKSPACE_PREFETCH_MAX_BYTES_ENV,
        values.get(WORKSPACE_PREFETCH_MAX_BYTES_ENV),
        default=None,
    )
    return WorkspacePrefetchConfig(enabled, max_concurrency, batch_size, max_bytes)


def audit_document_ids(metadata_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the alias/label bijection before accepting an evaluation input."""
    from dci_bench.data.document_ids import DOCUMENT_ID_POLICY, validate_source_document_ids

    mapping_path = metadata_dir / "source_doc_ids.json"
    originals_path = metadata_dir / "original_qrels.json"
    policy = manifest.get("document_id_policy")
    if policy is None and not mapping_path.exists() and not originals_path.exists():
        return {}
    if policy != DOCUMENT_ID_POLICY:
        raise ValueError("short document metadata must declare its document ID policy")
    mapping = validate_source_document_ids(json.loads(mapping_path.read_text(encoding="utf-8")))
    originals = json.loads(originals_path.read_text(encoding="utf-8"))
    qrels = json.loads((metadata_dir / "qrels.json").read_text(encoding="utf-8"))
    if not isinstance(originals, dict) or not isinstance(qrels, dict):
        raise ValueError("document qrels must be objects")
    for key, value in (("source_doc_ids_hash", mapping), ("original_qrels_hash", originals)):
        if stable_json_sha256(value) != manifest.get(key):
            raise ValueError(f"document identity hash mismatch: {key}")
    inverse = {original: alias for alias, original in mapping.items()}
    for query_id, judgments in qrels.items():
        source = originals.get(query_id)
        if not isinstance(source, dict) or not source or set(source) - inverse.keys():
            raise ValueError(f"missing or unmapped original qrels for {query_id}")
        if judgments != {inverse[original]: score for original, score in source.items()}:
            raise ValueError(f"document ID conversion changed labels for {query_id}")
    return {
        "document_id_policy": policy,
        "source_doc_ids_hash": manifest["source_doc_ids_hash"],
        "original_qrels_hash": manifest["original_qrels_hash"],
    }


def load_pool_for_runtime(
    metadata_root: Path,
    workspace_root: Path,
    task: str,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Verify a sealed pool and bind its tier to the selected runtime profile."""
    marker = metadata_root / task / "candidate_pools.json"
    configured = contract.get("candidate_pool")
    if not marker.exists():
        if configured:
            raise ValueError("document-pool profiles require prepared per-query candidate pools")
        return None
    if not configured:
        raise ValueError("candidate pools require a document-pool-v1 tier profile")
    suite = metadata_root.resolve().parent
    if metadata_root.resolve() != suite / "metadata" or workspace_root.resolve() != suite / "workspaces":
        raise ValueError("candidate-pool metadata and workspaces must belong to the same prepared suite")
    from dci_bench.data.document_pools import load_document_pool_manifest

    # V2 pins small metadata inventories at batch startup. Each selected body
    # is checked after copying into its isolated query workspace, avoiding a
    # scan of every candidate body before the first model request.
    payload = load_document_pool_manifest(
        suite, task=task,
        verify_content_hashes=configured.get('schema_version') != 'document-pool-v2',
    )
    if (payload["pool_size"] != configured["tier"]
            or payload["capacity"] != configured["capacity"]):
        raise ValueError("candidate-pool tier/capacity does not match the selected protocol")
    manifest = json.loads((metadata_root / task / "manifest.json").read_text(encoding="utf-8"))
    audit_document_ids(metadata_root / task, manifest)
    inventory = json.loads((metadata_root / task / "file_inventory.json").read_text(encoding="utf-8"))
    from dci_bench.data.document_pools import shared_pool_corpus
    shared = shared_pool_corpus(suite, task)
    if shared is not None:
        payload["_runtime_corpus"] = str(shared)
    payload["_runtime_documents"] = {row["alias"]: row for row in inventory.values()}
    return payload


class WorkspacePrefetchError(RuntimeError):
    """A bounded prefetch reservation or materialization failed."""


@dataclass(frozen=True)
class _SourceMetadata:
    path: Path
    bytes: int
    sha256: str
    fingerprint: tuple[int, int, int, int, int, int]


class PreparedQueryWorkspace:
    """A materialized query workspace owned by one prefetch lease."""

    def __init__(
        self,
        *,
        query_id: str,
        path: Path,
        reserved_bytes: int,
        prepare_seconds: float,
    ) -> None:
        self.query_id = query_id
        self.path = path
        self.reserved_bytes = reserved_bytes
        self.prepare_seconds = prepare_seconds
        self._released = False

    def cleanup(self) -> None:
        if self._released:
            return
        self._released = True
        if self.path.exists():
            shutil.rmtree(self.path)


def _candidate_record(pool: dict[str, Any], query_id: str) -> tuple[dict[str, Any], list[str]]:
    record = pool.get("pools", {}).get(query_id)
    if not isinstance(record, dict):
        raise ValueError(f"query has no candidate pool: {query_id}")
    aliases = record.get("candidate_doc_ids")
    if not isinstance(aliases, list):
        raise ValueError(f"candidate pool IDs must be a list: {query_id}")
    seen: set[str] = set()
    for alias in aliases:
        if not isinstance(alias, str) or not _SHORT_DOCUMENT_ID.fullmatch(alias):
            raise ValueError("invalid short candidate document ID")
        if alias in seen:
            raise ValueError(f"duplicate candidate document ID: {query_id}/{alias}")
        seen.add(alias)
    return record, aliases


def _runtime_corpus(pool: dict[str, Any], shared_workspace: Path) -> Path:
    configured = pool.get("_runtime_corpus")
    raw = Path(configured) if configured is not None else shared_workspace / "corpus"
    # Candidate pools are allowed to point at a prepared source corpus, but
    # the source itself must be a real directory.  The query workspace below
    # always receives regular copies; it never mounts this directory.
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("runtime corpus must be a regular directory")
    corpus = raw.resolve(strict=True)
    if corpus.is_symlink() or not corpus.is_dir():
        raise ValueError("runtime corpus must be a regular directory")
    return corpus


def _source_fingerprint(path: Path, expected: dict[str, Any]) -> _SourceMetadata:
    try:
        information = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"candidate document is missing: {path.name.removesuffix('.md')}") from exc
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISREG(information.st_mode):
        raise ValueError(
            f"candidate document is not a regular isolated file: {path.name.removesuffix('.md')}"
        )
    if information.st_nlink != 1:
        raise ValueError(
            f"candidate document is not a regular isolated file: {path.name.removesuffix('.md')}"
        )
    expected_bytes = expected.get("bytes")
    expected_sha = expected.get("sha256")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ValueError(f"invalid source byte inventory: {path.name}")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError(f"invalid source hash inventory: {path.name}")
    if information.st_size != expected_bytes:
        raise ValueError(f"candidate document changed since preflight: {path.stem}")
    fingerprint = (
        information.st_dev,
        information.st_ino,
        information.st_size,
        information.st_mtime_ns,
        information.st_ctime_ns,
        information.st_nlink,
    )
    return _SourceMetadata(path, expected_bytes, expected_sha, fingerprint)


class _SourceMetadataCache:
    """Cache source stat metadata only for one runtime object.

    A fresh lstat is still made on each copy so a symlink/hardlink swap cannot
    turn a cached path into a shared corpus mount.  The expected content hash
    is checked on every copied target as the final race-safe integrity check.
    """

    def __init__(self, corpus: Path, inventory: dict[str, Any]) -> None:
        self._corpus = corpus
        self._inventory = inventory
        self._lock = threading.Lock()
        self._cached: dict[str, _SourceMetadata] = {}

    @property
    def corpus(self) -> Path:
        return self._corpus

    def get(self, alias: str) -> _SourceMetadata:
        expected = self._inventory.get(alias)
        if not isinstance(expected, dict):
            raise ValueError(f"candidate document is absent from the runtime inventory: {alias}")
        path = self._corpus / f"{alias}.md"
        current = _source_fingerprint(path, expected)
        with self._lock:
            previous = self._cached.get(alias)
            if previous is not None and previous.fingerprint != current.fingerprint:
                raise ValueError(f"candidate document source fingerprint changed since preflight: {alias}")
            if previous is None:
                self._cached[alias] = current
                return current
            return previous


def _copy_verified_query_workspace(
    shared_workspace: Path,
    query_id: str,
    pool: dict[str, Any],
    *,
    source_cache: _SourceMetadataCache | None = None,
) -> PreparedQueryWorkspace:
    """Synchronously materialize one isolated query corpus with target hashes."""

    started = time.perf_counter()
    _record, aliases = _candidate_record(pool, query_id)
    inventory = pool.get("_runtime_documents")
    if not isinstance(inventory, dict):
        raise ValueError("runtime document inventory is missing")
    cache = source_cache or _SourceMetadataCache(_runtime_corpus(pool, shared_workspace), inventory)
    corpus = cache.corpus
    reserved_bytes = 0
    source_metadata: list[_SourceMetadata] = []
    for alias in aliases:
        metadata = cache.get(alias)
        source_metadata.append(metadata)
        reserved_bytes += metadata.bytes

    temporary = Path(tempfile.mkdtemp(prefix="dci-query-"))
    try:
        destination = temporary / "corpus"
        destination.mkdir()
        for alias, metadata in zip(aliases, source_metadata):
            target = destination / metadata.path.name
            shutil.copyfile(metadata.path, target)
            target_stat = target.lstat()
            if target_stat.st_nlink != 1 or not stat.S_ISREG(target_stat.st_mode) or target.is_symlink():
                raise ValueError(f"materialized candidate is not a regular isolated file: {alias}")
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if target_stat.st_size != metadata.bytes or digest.hexdigest() != metadata.sha256:
                raise ValueError(f"candidate document changed since preflight: {alias}")
        return PreparedQueryWorkspace(
            query_id=query_id,
            path=temporary,
            reserved_bytes=reserved_bytes,
            prepare_seconds=time.perf_counter() - started,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@dataclass
class _PrefetchFuture:
    query_id: str
    reserved_bytes: int
    future: Future[PreparedQueryWorkspace]
    optional: bool


class _WorkspaceAcquireCancelled(Exception):
    """Private signal used by an abandoned acquisition worker."""


class WorkspacePrefetcher:
    """Bounded producer/consumer for query workspaces in one Inspect task.

    The producer is a small thread pool because source copies are blocking
    filesystem work.  Async callers wait through ``anyio.to_thread`` in the
    task solver, and every acquired workspace must be released by that solver.
    At most ``active concurrency + batch_size`` workspaces are reserved.  A
    canceled acquisition only abandons that sample's request; the evaluation
    lifecycle owns the final manager close.
    """

    def __init__(
        self,
        shared_workspace: Path,
        query_ids: list[str] | tuple[str, ...],
        pool: dict[str, Any],
        config: WorkspacePrefetchConfig,
    ) -> None:
        if not config.enabled:
            raise ValueError("WorkspacePrefetcher requires enabled configuration")
        self._shared_workspace = Path(shared_workspace)
        self._query_ids = tuple(query_ids)
        if len(set(self._query_ids)) != len(self._query_ids):
            raise ValueError("workspace prefetch query IDs must be unique")
        self._pool = pool
        self._config = config
        self._source_cache = _SourceMetadataCache(
            _runtime_corpus(pool, self._shared_workspace),
            pool.get("_runtime_documents", {}),
        )
        self._estimated_bytes: dict[str, int] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(config.max_concurrency, config.batch_size)),
            thread_name_prefix="dci-workspace",
        )
        self._condition = threading.Condition()
        self._futures: dict[str, _PrefetchFuture] = {}
        self._active: dict[int, PreparedQueryWorkspace | None] = {}
        self._submitted: set[str] = set()
        self._finished: set[str] = set()
        self._next_index = 0
        self._reserved_count = 0
        self._reserved_bytes = 0
        self._active_token = 0
        self._cancelled = False
        self._closed = False
        self._acquiring = 0
        self._acquire_requests: set[threading.Event] = set()
        self._cleaning = 0

    @property
    def config(self) -> WorkspacePrefetchConfig:
        return self._config

    @property
    def inflight_count(self) -> int:
        with self._condition:
            return self._reserved_count

    @property
    def active_count(self) -> int:
        with self._condition:
            return sum(value is not None for value in self._active.values())

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._futures)

    @property
    def reserved_bytes(self) -> int:
        with self._condition:
            return self._reserved_bytes

    def refresh(self) -> None:
        """Start a fresh evaluation epoch after the previous one was closed."""

        with self._condition:
            if self._acquiring or self._cleaning or self._futures or self._active or self._reserved_count:
                raise WorkspacePrefetchError(
                    "workspace prefetcher cannot refresh while an evaluation is active"
                )
            if self._closed:
                self._executor = ThreadPoolExecutor(
                    max_workers=max(1, min(self._config.max_concurrency, self._config.batch_size)),
                    thread_name_prefix="dci-workspace",
                )
            self._closed = False
            self._cancelled = False
            self._submitted.clear()
            self._finished.clear()
            self._next_index = 0
            self._reserved_count = 0
            self._reserved_bytes = 0
            self._active.clear()
            self._condition.notify_all()

    def _query_bytes(self, query_id: str) -> int:
        if query_id not in self._estimated_bytes:
            _record, aliases = _candidate_record(self._pool, query_id)
            inventory = self._pool.get("_runtime_documents")
            if not isinstance(inventory, dict):
                raise ValueError("runtime document inventory is missing")
            total = 0
            for alias in aliases:
                expected = inventory.get(alias)
                if not isinstance(expected, dict) or not isinstance(expected.get("bytes"), int):
                    raise ValueError(f"invalid source byte inventory: {alias}")
                total += expected["bytes"]
            self._estimated_bytes[query_id] = total
        return self._estimated_bytes[query_id]

    def _cap_diagnostic(self, query_id: str, required: int) -> WorkspacePrefetchError:
        return WorkspacePrefetchError(
            "workspace prefetch disk byte cap exceeded for "
            f"query {query_id!r}: required {required} bytes, "
            f"reserved {self._reserved_bytes} bytes, cap {self._config.max_bytes} bytes; "
            f"increase {WORKSPACE_PREFETCH_MAX_BYTES_ENV} or reduce concurrency"
        )

    def _can_reserve_locked(self, required: int) -> bool:
        cap = self._config.max_bytes
        return cap is None or self._reserved_bytes + required <= cap

    def _submit_locked(self, query_id: str, *, required: bool) -> _PrefetchFuture | None:
        if self._closed or self._cancelled:
            if required:
                raise WorkspacePrefetchError("workspace prefetcher is canceled")
            return None
        required_bytes = self._query_bytes(query_id)
        if self._config.max_bytes is not None and required_bytes > self._config.max_bytes:
            if required:
                raise self._cap_diagnostic(query_id, required_bytes)
            return None
        if not self._can_reserve_locked(required_bytes):
            if required:
                raise self._cap_diagnostic(query_id, required_bytes)
            return None
        future = self._executor.submit(
            _copy_verified_query_workspace,
            self._shared_workspace,
            query_id,
            self._pool,
            source_cache=self._source_cache,
        )
        state = _PrefetchFuture(query_id, required_bytes, future, optional=not required)
        self._futures[query_id] = state
        self._submitted.add(query_id)
        self._reserved_count += 1
        self._reserved_bytes += required_bytes
        future.add_done_callback(self._future_done)
        return state

    def _future_done(self, _future: Future[PreparedQueryWorkspace]) -> None:
        with self._condition:
            self._condition.notify_all()

    def _evict_pending_locked(
        self,
    ) -> tuple[bool, list[tuple[_PrefetchFuture, PreparedQueryWorkspace | None]]]:
        """Evict one optional producer, retaining its reservation until cleanup."""

        for query_id, state in list(self._futures.items())[::-1]:
            if not state.optional:
                continue
            if not state.future.done() and not state.future.cancel():
                continue
            del self._futures[query_id]
            self._submitted.remove(query_id)
            if state.future.cancelled():
                self._reserved_count -= 1
                self._reserved_bytes -= state.reserved_bytes
                self._condition.notify_all()
                return True, []
            try:
                prepared = state.future.result()
            except BaseException:
                self._reserved_count -= 1
                self._reserved_bytes -= state.reserved_bytes
                self._condition.notify_all()
                return True, []
            # The directory still occupies its reservation until cleanup has
            # completed.  The caller performs that cleanup outside the lock.
            return True, [(state, prepared)]
        return False, []

    def _fill_locked(self) -> None:
        target = min(
            len(self._query_ids),
            len(self._active) + self._config.batch_size,
        )
        while self._reserved_count < target and self._next_index < len(self._query_ids):
            query_id = self._query_ids[self._next_index]
            self._next_index += 1
            if query_id in self._submitted or query_id in self._finished:
                continue
            if self._submit_locked(query_id, required=False) is None:
                # A byte cap can leave a perfectly valid later query waiting
                # for a release.  Do not turn that optional prefetch into a
                # failure of the active sample.
                self._next_index -= 1
                break

    def _acquire_sync(
        self,
        query_id: str,
        cancel_event: threading.Event,
    ) -> PreparedQueryWorkspace:
        if query_id not in self._query_ids:
            raise ValueError(f"workspace prefetch query is not in the task selection: {query_id}")
        with self._condition:
            if self._closed or self._cancelled:
                raise WorkspacePrefetchError("workspace prefetcher is canceled")
            self._acquiring += 1
            self._acquire_requests.add(cancel_event)
            self._active_token += 1
            token = self._active_token
            self._active[token] = None  # type: ignore[assignment]
        try:
            while True:
                evicted: list[tuple[_PrefetchFuture, PreparedQueryWorkspace | None]] = []
                evicted_any = False
                with self._condition:
                    if cancel_event.is_set():
                        raise _WorkspaceAcquireCancelled
                    while query_id not in self._futures:
                        if cancel_event.is_set():
                            raise _WorkspaceAcquireCancelled
                        if self._closed or self._cancelled:
                            raise WorkspacePrefetchError("workspace prefetcher is canceled")
                        if query_id in self._finished:
                            raise WorkspacePrefetchError(
                                f"query workspace already released: {query_id}"
                            )
                        required_bytes = self._query_bytes(query_id)
                        if (
                            self._config.max_bytes is not None
                            and required_bytes > self._config.max_bytes
                        ):
                            raise self._cap_diagnostic(query_id, required_bytes)
                        if self._can_reserve_locked(required_bytes):
                            self._submit_locked(query_id, required=True)
                            break
                        evicted_any, evicted = self._evict_pending_locked()
                        if evicted_any:
                            break
                        self._condition.wait(timeout=0.05)
                    if not evicted_any:
                        # Claim the future before waiting for its result.
                        # Cancellation can then only discard this request's
                        # producer; the lease owns its reservation.
                        state = self._futures.pop(query_id)
                if evicted_any:
                    for evicted_state, prepared in evicted:
                        try:
                            if prepared is not None:
                                prepared.cleanup()
                        finally:
                            with self._condition:
                                self._reserved_count -= 1
                                self._reserved_bytes -= evicted_state.reserved_bytes
                                self._condition.notify_all()
                    continue
                break

            try:
                prepared = state.future.result()
            except BaseException:
                with self._condition:
                    self._submitted.discard(query_id)
                    self._reserved_count -= 1
                    self._reserved_bytes -= state.reserved_bytes
                    self._active.pop(token, None)
                    self._condition.notify_all()
                raise
            if cancel_event.is_set() or self._closed or self._cancelled:
                try:
                    prepared.cleanup()
                finally:
                    with self._condition:
                        self._submitted.discard(query_id)
                        self._reserved_count -= 1
                        self._reserved_bytes -= state.reserved_bytes
                        self._active.pop(token, None)
                        self._condition.notify_all()
                raise _WorkspaceAcquireCancelled
            with self._condition:
                self._active[token] = prepared
                self._fill_locked()
                self._condition.notify_all()
            if cancel_event.is_set() or self._closed or self._cancelled:
                self._release_sync(prepared)
                raise _WorkspaceAcquireCancelled
            prepared._prefetch_token = token  # type: ignore[attr-defined]
            return prepared
        except BaseException:
            with self._condition:
                orphan = self._active.pop(token, None)
                if orphan is not None:
                    self._cleaning += 1
                self._condition.notify_all()
            if orphan is not None:
                try:
                    orphan.cleanup()
                finally:
                    with self._condition:
                        self._reserved_count -= 1
                        self._reserved_bytes -= orphan.reserved_bytes
                        self._cleaning -= 1
                        self._condition.notify_all()
            raise
        finally:
            with self._condition:
                self._acquiring -= 1
                self._acquire_requests.discard(cancel_event)
                self._condition.notify_all()

    async def acquire(self, query_id: str) -> PreparedQueryWorkspace:
        import anyio

        cancel_event = threading.Event()
        try:
            return await anyio.to_thread.run_sync(
                self._acquire_sync,
                query_id,
                cancel_event,
                abandon_on_cancel=True,
            )
        except anyio.get_cancelled_exc_class():
            cancel_event.set()
            with self._condition:
                self._condition.notify_all()
            raise

    def _release_sync(self, prepared: PreparedQueryWorkspace) -> None:
        token = getattr(prepared, "_prefetch_token", None)
        with self._condition:
            if token not in self._active:
                return
            self._active.pop(token)
            self._finished.add(prepared.query_id)
            self._cleaning += 1
        try:
            prepared.cleanup()
        except BaseException:
            with self._condition:
                self._reserved_count -= 1
                self._reserved_bytes -= prepared.reserved_bytes
                self._cleaning -= 1
                self._condition.notify_all()
            raise
        with self._condition:
            self._reserved_count -= 1
            self._reserved_bytes -= prepared.reserved_bytes
            self._cleaning -= 1
            self._fill_locked()
            self._condition.notify_all()

    async def release(self, prepared: PreparedQueryWorkspace) -> None:
        import anyio

        await anyio.to_thread.run_sync(self._release_sync, prepared, abandon_on_cancel=False)

    def _cancel_pending_sync(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._cancelled = True
            for cancel_event in self._acquire_requests:
                cancel_event.set()
            self._condition.notify_all()
            while self._acquiring or self._cleaning:
                self._condition.wait(timeout=0.05)
            pending = list(self._futures.values())
            self._futures.clear()
            for state in pending:
                self._submitted.discard(state.query_id)
                if state.future.cancel():
                    self._reserved_count -= 1
                    self._reserved_bytes -= state.reserved_bytes
                elif state.future.cancelled():
                    # A future may have been canceled by the executor between
                    # the first cancel attempt and this accounting pass.  It
                    # has no producer cleanup to await, so release its
                    # reservation exactly once here.
                    self._reserved_count -= 1
                    self._reserved_bytes -= state.reserved_bytes
                else:
                    self._cleaning += 1
            self._condition.notify_all()
        for state in pending:
            if state.future.cancelled():
                continue
            try:
                prepared = state.future.result()
            except BaseException:
                with self._condition:
                    self._reserved_count -= 1
                    self._reserved_bytes -= state.reserved_bytes
                    self._cleaning -= 1
                    self._condition.notify_all()
                continue
            try:
                prepared.cleanup()
            finally:
                with self._condition:
                    self._reserved_count -= 1
                    self._reserved_bytes -= state.reserved_bytes
                    self._cleaning -= 1
                    self._condition.notify_all()

    async def cancel_pending(self) -> None:
        import anyio

        await anyio.to_thread.run_sync(self._cancel_pending_sync, abandon_on_cancel=False)

    def _close_sync(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for cancel_event in self._acquire_requests:
                cancel_event.set()
            self._condition.notify_all()
            while self._acquiring or self._cleaning:
                self._condition.wait(timeout=0.05)
            pending = list(self._futures.values())
            self._futures.clear()
            for state in pending:
                self._submitted.discard(state.query_id)
                state.future.cancel()
            active = [prepared for prepared in self._active.values() if prepared is not None]
            self._active.clear()
            self._condition.notify_all()
        for state in pending:
            if state.future.cancelled():
                with self._condition:
                    self._reserved_count -= 1
                    self._reserved_bytes -= state.reserved_bytes
                continue
            try:
                prepared = state.future.result()
            except BaseException:
                with self._condition:
                    self._reserved_count -= 1
                    self._reserved_bytes -= state.reserved_bytes
                continue
            try:
                prepared.cleanup()
            finally:
                with self._condition:
                    self._reserved_count -= 1
                    self._reserved_bytes -= state.reserved_bytes
        for prepared in active:
            try:
                prepared.cleanup()
            finally:
                with self._condition:
                    self._reserved_count -= 1
                    self._reserved_bytes -= prepared.reserved_bytes
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._condition:
            self._reserved_count = 0
            self._reserved_bytes = 0
            self._condition.notify_all()

    async def close(self) -> None:
        import anyio

        await anyio.to_thread.run_sync(self._close_sync, abandon_on_cancel=False)

    def close_sync(self) -> None:
        """Close the manager from a synchronous evaluation finalizer."""

        self._close_sync()


@contextmanager
def materialize_query_workspace(shared_workspace: Path, query_id: str, pool: dict[str, Any] | None):
    """Create only the executing query's corpus, and remove it on every exit.

    Plain copies keep the existing Pi single-link/no-symlink isolation checks.
    Temporary disk usage is bounded by active queries, rather than dataset size.
    The complete shared corpus is never passed to Pi for a pooled evaluation.
    """
    if pool is None:
        yield shared_workspace
        return
    prepared = _copy_verified_query_workspace(shared_workspace, query_id, pool)
    try:
        yield prepared.path
    finally:
        prepared.cleanup()
