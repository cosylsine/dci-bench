#!/usr/bin/env python3
"""Project one deterministic query from every source into three pool tiers.

The input catalogue and all of its source suites are treated as immutable.
The projection copies only the selected documents and the selected metadata
rows into a new output tree.  Candidate records are read with :mod:`ijson` so
that a large ``candidate_pools.json`` is never expanded in memory.
"""

from __future__ import annotations

import argparse
import copy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
from typing import Any, Iterable, Iterator, Mapping

try:
    import ijson
except ImportError as exc:  # pragma: no cover - exercised by an environment check
    raise RuntimeError(
        "prepare_pool_smoke.py requires ijson; use the dci-bench Python environment"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.document_pools import (  # noqa: E402
    POOL_SCHEMA_VERSION,
    sha256_file,
    stable_json_sha256,
    verify_document_pool_suite,
)


DEFAULT_SEED = 20261026
CAPACITIES = (200, 500, 1000)
CAPACITY_TO_POOL_SIZE = {200: "small", 500: "medium", 1000: "large"}
CAPACITY_TO_PROFILE = {
    200: "document-pool-v1-small",
    500: "document-pool-v1-medium",
    1000: "document-pool-v1-large",
}
PROJECTION_SCHEMA_VERSION = "dci-pool-smoke-projection-20261026-v1"
SHARED_CORPUS_MARKER = "shared_corpus.json"
POOL_METADATA_FILES = (
    "candidate_pools.json",
    "doc_id_files.json",
    "file_inventory.json",
    "manifest.json",
    "original_qrels.json",
    "qrels.json",
    "queries.jsonl",
    "source_doc_ids.json",
    "source_identity.json",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STORAGE_ENCODING_RE = re.compile(
    rb'"storage_encoding"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
)


def _json_compatible(value: Any) -> Any:
    """Normalize ijson's Decimal numbers to stdlib JSON number types."""

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    return value


def _json_load(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {path}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _query_digest(seed: int, task: str, query_id: str) -> str:
    return _sha256_bytes(f"{seed}:{task}:{query_id}".encode("utf-8"))


def _safe_relative_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or os.path.isabs(value) or "\\" in value:
        raise ValueError(f"invalid relative path: {value!r}")
    path = root.joinpath(*Path(value).parts)
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes root: {value!r}") from exc
    return path


def _read_query_rows(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    rows: list[tuple[dict[str, Any], bytes]] = []
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid query row at {path}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"query row is not an object at {path}:{line_number}")
            query_id = row.get("_id", row.get("query_id"))
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"query row has no non-empty ID at {path}:{line_number}")
            if query_id in seen:
                raise ValueError(f"duplicate query ID {query_id!r} in {path}")
            seen.add(query_id)
            rows.append((dict(row), raw if raw.endswith(b"\n") else raw + b"\n"))
    if not rows:
        raise ValueError(f"query file is empty: {path}")
    return rows


def _choose_query(
    rows: Iterable[tuple[dict[str, Any], bytes]], seed: int, task: str
) -> tuple[dict[str, Any], bytes, str]:
    candidates: list[tuple[str, dict[str, Any], bytes]] = []
    for row, raw in rows:
        query_id = row.get("_id", row.get("query_id"))
        candidates.append((_query_digest(seed, task, query_id), row, raw))
    digest, row, raw = min(candidates, key=lambda item: (item[0], item[1].get("_id", item[1].get("query_id"))))
    return row, raw, digest


class _HashingReader:
    """A file wrapper that hashes bytes consumed by ijson and detects encoding."""

    def __init__(self, handle: Any):
        self._handle = handle
        self._digest = hashlib.sha256()
        self._tail = b""
        self.storage_encoding: str | None = None

    def _record(self, data: bytes) -> bytes:
        if data:
            self._digest.update(data)
            if self.storage_encoding is None:
                window = self._tail + data
                match = _STORAGE_ENCODING_RE.search(window)
                if match is not None:
                    try:
                        self.storage_encoding = json.loads(
                            b'"' + match.group(1) + b'"'
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        raise ValueError("candidate_pools.json has invalid storage_encoding")
                self._tail = window[-4096:]
        return data

    def read(self, size: int = -1) -> bytes:
        return self._record(self._handle.read(size))

    def readinto(self, buffer: bytearray) -> int:
        count = self._handle.readinto(buffer)
        if count:
            self._record(bytes(buffer[:count]))
        return count

    def digest(self) -> str:
        return self._digest.hexdigest()


class _ValueBuilder:
    """Build exactly one ijson value from parser events."""

    def __init__(self) -> None:
        self._stack: list[tuple[Any, Any]] = []
        self.result: Any = None
        self.complete = False

    def _attach(self, value: Any) -> None:
        if not self._stack:
            self.result = value
            self.complete = True
            return
        container, pending = self._stack[-1]
        if isinstance(container, dict):
            if pending is None:
                raise ValueError("invalid candidate JSON map event")
            container[pending] = value
            self._stack[-1] = (container, None)
        else:
            container.append(value)

    def feed(self, event: str, value: Any) -> None:
        if self.complete:
            raise ValueError("candidate record has trailing events")
        if event == "start_map":
            container: Any = {}
            if self._stack:
                self._attach(container)
            else:
                self.result = container
            self._stack.append((container, None))
        elif event == "start_array":
            container = []
            if self._stack:
                self._attach(container)
            else:
                self.result = container
            self._stack.append((container, None))
        elif event == "map_key":
            if not self._stack or not isinstance(self._stack[-1][0], dict):
                raise ValueError("invalid candidate JSON map key")
            container, pending = self._stack[-1]
            if pending is not None:
                raise ValueError("candidate JSON map key is repeated")
            self._stack[-1] = (container, value)
        elif event in {"end_map", "end_array"}:
            if not self._stack:
                raise ValueError("invalid candidate JSON container end")
            container, pending = self._stack.pop()
            if event == "end_map" and not isinstance(container, dict):
                raise ValueError("candidate JSON map/array mismatch")
            if event == "end_array" and not isinstance(container, list):
                raise ValueError("candidate JSON array/map mismatch")
            if isinstance(container, dict) and pending is not None:
                raise ValueError("candidate JSON map has a missing value")
            if not self._stack and self.result is container:
                self.complete = True
        else:
            self._attach(value)


def _skip_ijson_value(iterator: Iterator[tuple[str, str, Any]], first: tuple[str, str, Any]) -> None:
    event = first[1]
    if event not in {"start_map", "start_array"}:
        return
    depth = 1
    while depth:
        try:
            _, event, _ = next(iterator)
        except StopIteration as exc:
            raise ValueError("candidate JSON ended while skipping a pool record") from exc
        if event in {"start_map", "start_array"}:
            depth += 1
        elif event in {"end_map", "end_array"}:
            depth -= 1


def _extract_candidate_record(path: Path, query_id: str) -> tuple[dict[str, Any], str, str | None]:
    """Extract one ``pools[query_id]`` record while hashing the whole file."""

    with path.open("rb") as raw_handle:
        reader = _HashingReader(raw_handle)
        iterator = iter(ijson.parse(reader))
        try:
            _, event, _ = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"candidate pool file is empty: {path}") from exc
        if event != "start_map":
            raise ValueError(f"candidate pool root is not an object: {path}")
        found: dict[str, Any] | None = None
        while True:
            try:
                _, event, value = next(iterator)
            except StopIteration:
                break
            if event == "end_map":
                break
            if event != "map_key":
                raise ValueError(f"invalid candidate pool root event in {path}")
            key = value
            try:
                first = next(iterator)
            except StopIteration as exc:
                raise ValueError(f"candidate pool key {key!r} has no value") from exc
            if key != "pools":
                _skip_ijson_value(iterator, first)
                continue
            if first[1] != "start_map":
                raise ValueError("candidate pool pools field is not an object")
            while True:
                try:
                    _, event, value = next(iterator)
                except StopIteration as exc:
                    raise ValueError("candidate pool pools object ended unexpectedly") from exc
                if event == "end_map":
                    break
                if event != "map_key":
                    raise ValueError("invalid candidate pool query key")
                current_query_id = value
                try:
                    first_record_event = next(iterator)
                except StopIteration as exc:
                    raise ValueError("candidate pool record has no value") from exc
                if current_query_id != query_id:
                    _skip_ijson_value(iterator, first_record_event)
                    continue
                builder = _ValueBuilder()
                builder.feed(first_record_event[1], first_record_event[2])
                while not builder.complete:
                    try:
                        _, record_event, record_value = next(iterator)
                    except StopIteration as exc:
                        raise ValueError("candidate pool selected record is truncated") from exc
                    builder.feed(record_event, record_value)
                if not isinstance(builder.result, dict):
                    raise ValueError("candidate pool selected record is not an object")
                found = _json_compatible(builder.result)
                # The remaining bytes are drained below for the completion hash;
                # no other candidate record is expanded.
                break
            break
        while reader.read(4 * 1024 * 1024):
            pass
        digest = reader.digest()
        if found is None:
            raise ValueError(f"candidate pool query is missing: {query_id!r} in {path}")
        return found, digest, reader.storage_encoding


def _normalized_candidate_record(record: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    """Mirror document_pools._load_json's compact-record expansion."""

    result = copy.deepcopy(dict(record))
    aliases = result.get("candidate_doc_ids")
    if not isinstance(aliases, list):
        raise ValueError("candidate record candidate_doc_ids is not a list")
    if "candidate_original_doc_ids" not in result:
        try:
            result["candidate_original_doc_ids"] = [mapping[alias] for alias in aliases]
        except KeyError as exc:
            raise ValueError(f"candidate record references unknown alias: {exc.args[0]!r}") from exc
    audit = result.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("candidate record audit is not an object")
    if "selected_ids" not in audit:
        audit["selected_ids"] = list(result["candidate_original_doc_ids"])
    if "selected_scores" not in audit:
        audit["selected_scores"] = result.get("candidate_scores")
    return result


def _metadata_hashes(completion: Mapping[str, Any], *, label: str) -> Mapping[str, str]:
    hashes = completion.get("metadata_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError(f"{label} completion has no metadata_hashes")
    for name, digest in hashes.items():
        if not isinstance(name, str) or not name or Path(name).is_absolute() or "\\" in name:
            raise ValueError(f"{label} completion has an unsafe metadata name: {name!r}")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"{label} completion has an invalid metadata digest: {name!r}")
    return hashes  # type: ignore[return-value]


def _verify_metadata_completion(
    metadata_dir: Path,
    completion: Mapping[str, Any],
    *,
    label: str,
    skip: set[str] | None = None,
) -> None:
    hashes = _metadata_hashes(completion, label=label)
    skip = skip or set()
    for name, expected in hashes.items():
        path = _safe_relative_path(metadata_dir, name)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} metadata file is missing or linked: {name}")
        if name not in skip and sha256_file(path) != expected:
            raise ValueError(f"{label} metadata SHA-256 mismatch: {name}")


def _load_source_export(
    entry: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    task = entry.get("task")
    export_value = entry.get("export")
    if not isinstance(task, str) or not task:
        raise ValueError("catalog source has an invalid task")
    if not isinstance(export_value, str) or not export_value:
        raise ValueError(f"catalog source {task!r} has no export path")
    export = Path(export_value).resolve()
    metadata_dir = export / "metadata" / task
    completion_path = metadata_dir / "_COMPLETE.json"
    completion = _json_load(completion_path, label=f"source export completion for {task}")
    if not isinstance(completion, Mapping):
        raise ValueError(f"source export completion is not an object: {completion_path}")
    _verify_metadata_completion(metadata_dir, completion, label=f"source export {task}")
    manifest = _json_load(metadata_dir / "manifest.json", label=f"source manifest for {task}")
    if not isinstance(manifest, Mapping) or manifest.get("task") != task:
        raise ValueError(f"source export manifest task mismatch: {task}")
    query_rows = _read_query_rows(metadata_dir / "queries.jsonl")
    query_row, query_raw, query_digest = _choose_query(query_rows, seed, task)
    return {
        "task": task,
        "export": export,
        "metadata_dir": metadata_dir,
        "completion": completion,
        "manifest": dict(manifest),
        "query_row": query_row,
        "query_raw": query_raw,
        "query_digest": query_digest,
        "query_count": len(query_rows),
        "source_metadata_complete_sha256": sha256_file(completion_path),
    }


def _validate_shared_marker(pool_suite: Path, task: str, completion: Mapping[str, Any]) -> Path:
    marker_path = pool_suite / SHARED_CORPUS_MARKER
    marker = _json_load(marker_path, label=f"shared corpus marker for {task}")
    expected = {
        "schema_version": "dci-shared-pool-corpus-v1",
        "task": task,
        "corpus": f"source/workspaces/{task}/corpus",
    }
    if marker != expected:
        raise ValueError(f"source shared corpus marker mismatch: {marker_path}")
    expected_hash = completion.get("shared_corpus_sha256")
    if not isinstance(expected_hash, str) or sha256_file(marker_path) != expected_hash:
        raise ValueError(f"source shared corpus marker SHA-256 mismatch: {marker_path}")
    corpus = (pool_suite.parent.parent / "source" / "workspaces" / task / "corpus").resolve()
    if corpus.is_symlink() or not corpus.is_dir():
        raise ValueError(f"source shared corpus is missing or linked: {corpus}")
    return corpus


def _read_source_queries(path: Path) -> dict[str, dict[str, Any]]:
    return {row.get("_id", row.get("query_id")): row for row, _ in _read_query_rows(path)}


def _validate_pool_inventory(
    inventory: Mapping[str, Any],
    *,
    task: str,
    mapping: Mapping[str, str],
    doc_id_files: Mapping[str, str],
    selected_aliases: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    prefix = f"workspaces/{task}/corpus/"
    for relative, row in inventory.items():
        if not isinstance(relative, str) or not relative.startswith(prefix):
            raise ValueError(f"source inventory path is invalid: {relative!r}")
        if not isinstance(row, Mapping):
            raise ValueError(f"source inventory row is not an object: {relative!r}")
        alias = row.get("alias")
        if not isinstance(alias, str) or alias not in mapping or alias not in doc_id_files:
            raise ValueError(f"source inventory alias is invalid: {relative!r}")
        filename = relative[len(prefix):]
        if filename != doc_id_files[alias] or row.get("original_id") != mapping[alias]:
            raise ValueError(f"source inventory mapping mismatch: {relative!r}")
        if not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
            raise ValueError(f"source inventory byte count is invalid: {relative!r}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"source inventory SHA-256 is invalid: {relative!r}")
        result[alias] = dict(row)
    missing = selected_aliases - set(result)
    if missing:
        raise ValueError(f"source inventory is missing selected aliases: {sorted(missing)[:5]}")
    return {alias: result[alias] for alias in selected_aliases}


def _verify_migrated_completion(pool_suite, task, capacity, expected_sha, current):
    """The consolidated catalog retains the pre-migration completion hash.

    Authenticate that historical seal, constrain changes to the documented
    compact encoding/path migration, then verify every current metadata hash.
    No historical file is rewritten and no whole-corpus verification is claimed.
    """
    archive = pool_suite.parent.parent / 'provenance' / f'pool-{capacity}-controls.tar.gz'
    with tarfile.open(archive, 'r:gz') as handle:
        old_bytes = handle.extractfile(f'{task}/_COMPLETE.json').read()
        old_manifest_bytes = handle.extractfile(f'{task}/manifest.json').read()
    if _sha256_bytes(old_bytes) != expected_sha:
        raise ValueError('Historical pool completion does not match catalog')
    old = json.loads(old_bytes)
    if _sha256_bytes(old_manifest_bytes) != old['metadata_hashes']['manifest.json']:
        raise ValueError('Historical manifest hash mismatch')
    exempt = {'metadata_hashes', 'shared_corpus_sha256'}
    if {k: v for k, v in old.items() if k not in exempt} != {k: v for k, v in current.items() if k not in exempt}:
        raise ValueError('Pool completion changed beyond documented migration')
    if set(old['metadata_hashes']) != set(current['metadata_hashes']):
        raise ValueError('Migrated metadata inventory changed')
    for name, digest in old['metadata_hashes'].items():
        if name not in {'candidate_pools.json', 'manifest.json'} and current['metadata_hashes'][name] != digest:
            raise ValueError(f'Migration changed protected metadata: {name}')
    prior_manifest = json.loads(old_manifest_bytes)
    current_manifest = _json_load(pool_suite / 'metadata' / task / 'manifest.json', label='migrated manifest')
    prior_manifest['pool_size'] = CAPACITY_TO_POOL_SIZE[capacity]
    prior_manifest['rankings_path'] = str(pool_suite.parent.parent / 'ranking.json.gz')
    if prior_manifest != current_manifest:
        raise ValueError('Pool manifest changed beyond tier/path migration')
    return {'legacy_completion_sha256': expected_sha, 'archive': str(archive),
            'archive_sha256': sha256_file(archive), 'migration_fields_verified': True,
            'all_query_record_equivalence_recomputed': False}


def _validate_source_pool(
    source: Mapping[str, Any],
    pool_entry: Mapping[str, Any],
    *,
    task: str,
    capacity: int,
    selected_query: Mapping[str, Any],
    selected_query_raw: bytes,
) -> dict[str, Any]:
    path_value = pool_entry.get("path")
    expected_completion_sha = pool_entry.get("completion_sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"catalog pool path is invalid for {task}/{capacity}")
    pool_suite = Path(path_value).resolve()
    completion_path = pool_suite / "metadata" / task / "_COMPLETE.json"
    if not completion_path.is_file():
        raise ValueError(f"source pool completion is missing: {completion_path}")
    actual_completion_sha = sha256_file(completion_path)
    completion = _json_load(completion_path, label=f"source pool completion {task}/{capacity}")
    migration = None
    if actual_completion_sha != expected_completion_sha:
        migration = _verify_migrated_completion(pool_suite, task, capacity, expected_completion_sha, completion)
    if not isinstance(completion, Mapping) or completion.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError(f"source pool completion schema mismatch: {completion_path}")
    metadata_dir = completion_path.parent
    candidate_path = metadata_dir / "candidate_pools.json"
    _verify_metadata_completion(
        metadata_dir,
        completion,
        label=f"source pool {task}/{capacity}",
        skip={"candidate_pools.json"},
    )
    _validate_shared_marker(pool_suite, task, completion)
    manifest = _json_load(metadata_dir / "manifest.json", label=f"source pool manifest {task}/{capacity}")
    source_identity = _json_load(metadata_dir / "source_identity.json", label=f"source identity {task}/{capacity}")
    mapping = _json_load(metadata_dir / "source_doc_ids.json", label=f"source IDs {task}/{capacity}")
    doc_id_files = _json_load(metadata_dir / "doc_id_files.json", label=f"source filenames {task}/{capacity}")
    if not isinstance(manifest, Mapping) or manifest.get("task") != task or manifest.get("capacity") != capacity:
        raise ValueError(f"source pool manifest task/capacity mismatch: {task}/{capacity}")
    if not isinstance(source_identity, Mapping) or not isinstance(mapping, Mapping) or not isinstance(doc_id_files, Mapping):
        raise ValueError(f"source pool identity/mapping is invalid: {task}/{capacity}")
    mapping = {str(k): str(v) for k, v in mapping.items()}
    doc_id_files = {str(k): str(v) for k, v in doc_id_files.items()}
    if set(mapping) != set(doc_id_files) or not mapping:
        raise ValueError(f"source pool mapping/file mapping is not complete: {task}/{capacity}")
    source_queries = _read_source_queries(metadata_dir / "queries.jsonl")
    query_id = selected_query.get("_id", selected_query.get("query_id"))
    if query_id not in source_queries or source_queries[query_id] != dict(selected_query):
        raise ValueError(f"source pool query does not pair with export query: {task}/{capacity}")
    # Retain the source qrels values and aliases exactly; the output filter is
    # applied only after all source completion checks have passed.
    original_qrels = _json_load(metadata_dir / "original_qrels.json", label=f"source original qrels {task}/{capacity}")
    qrels = _json_load(metadata_dir / "qrels.json", label=f"source canonical qrels {task}/{capacity}")
    if not isinstance(original_qrels, Mapping) or not isinstance(qrels, Mapping):
        raise ValueError(f"source qrels are not mappings: {task}/{capacity}")
    if query_id not in original_qrels or query_id not in qrels:
        raise ValueError(f"source qrels do not contain selected query: {task}/{capacity}")
    record, candidate_sha, storage_encoding = _extract_candidate_record(candidate_path, query_id)
    expected_candidate_sha = _metadata_hashes(completion, label=f"source pool {task}/{capacity}").get(
        "candidate_pools.json"
    )
    if candidate_sha != expected_candidate_sha:
        raise ValueError(f"source candidate pool SHA-256 mismatch: {candidate_path}")
    if record.get("query_id") != query_id:
        raise ValueError(f"source candidate record query mismatch: {task}/{capacity}")
    aliases = record.get("candidate_doc_ids")
    if not isinstance(aliases, list) or len(aliases) != min(capacity, len(doc_id_files)):
        raise ValueError(f"source candidate record capacity mismatch: {task}/{capacity}")
    if len(set(aliases)) != len(aliases) or any(alias not in doc_id_files for alias in aliases):
        raise ValueError(f"source candidate record has unknown/duplicate aliases: {task}/{capacity}")
    normalized_record = _normalized_candidate_record(record, mapping)
    originals = normalized_record.get("candidate_original_doc_ids")
    if not isinstance(originals, list) or len(originals) != len(aliases):
        raise ValueError(f"source candidate original IDs are invalid: {task}/{capacity}")
    audit = normalized_record.get("audit")
    if not isinstance(audit, Mapping) or audit.get("query_id") != query_id:
        raise ValueError(f"source candidate audit is invalid: {task}/{capacity}")
    if audit.get("capacity") != capacity or audit.get("source_size") != len(doc_id_files):
        raise ValueError(f"source candidate audit source size/capacity mismatch: {task}/{capacity}")
    positive_aliases = {
        alias
        for alias, score in qrels[query_id].items()
        if alias in mapping and float(score) > 0
    }
    if not positive_aliases.issubset(set(aliases)):
        raise ValueError(f"source candidate dropped a positive qrel: {task}/{capacity}")
    inventory = _json_load(metadata_dir / "file_inventory.json", label=f"source inventory {task}/{capacity}")
    if not isinstance(inventory, Mapping):
        raise ValueError(f"source inventory is not an object: {task}/{capacity}")
    inventory_rows = _validate_pool_inventory(
        inventory,
        task=task,
        mapping=mapping,
        doc_id_files=doc_id_files,
        selected_aliases=set(aliases),
    )
    source_export_metadata = source["metadata_dir"]
    export_mapping = _json_load(source_export_metadata / "source_doc_ids.json", label=f"export source IDs {task}")
    export_doc_id_files = _json_load(source_export_metadata / "doc_id_files.json", label=f"export filenames {task}")
    if mapping != export_mapping or doc_id_files != export_doc_id_files:
        raise ValueError(f"source pool mapping differs from export metadata: {task}/{capacity}")
    export_query = source["query_row"]
    if export_query != selected_query:
        raise ValueError(f"selected query changed while preparing {task}")
    if source_queries[query_id] != json.loads(selected_query_raw.decode("utf-8")):
        raise ValueError(f"selected query bytes do not parse identically: {task}/{capacity}")
    return {
        "capacity": capacity,
        "pool_suite": pool_suite,
        "metadata_dir": metadata_dir,
        "completion": dict(completion),
        "completion_sha256": actual_completion_sha,
        "migration": migration,
        "manifest": dict(manifest),
        "source_identity": dict(source_identity),
        "mapping": mapping,
        "doc_id_files": doc_id_files,
        "query_id": query_id,
        "query_row": dict(selected_query),
        "query_raw": selected_query_raw,
        "original_qrels": dict(original_qrels),
        "qrels": dict(qrels),
        "record": record,
        "normalized_record": normalized_record,
        "record_sha256": stable_json_sha256(normalized_record),
        "candidate_sha256": candidate_sha,
        "storage_encoding": storage_encoding,
        "inventory_rows": inventory_rows,
    }


def _copy_verified(source: Path, destination: Path, expected_sha: str, expected_bytes: int) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"selected source document is missing or linked: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        for chunk in iter(lambda: source_handle.read(4 * 1024 * 1024), b""):
            destination_handle.write(chunk)
    if destination.stat().st_nlink != 1:
        raise ValueError(f"selected document copy size/link mismatch: {destination}")
    actual_sha = sha256_file(destination)
    if destination.stat().st_size != expected_bytes or actual_sha != expected_sha:
        raise ValueError(f"selected document copy SHA-256 mismatch: {destination}")


def _filter_qrels(payload: Mapping[str, Any], query_id: str) -> dict[str, Any]:
    if query_id not in payload:
        raise ValueError(f"qrels are missing selected query: {query_id}")
    return {query_id: copy.deepcopy(payload[query_id])}


def _build_pool_projection(
    destination: Path,
    source: Mapping[str, Any],
    pool: Mapping[str, Any],
    *,
    task: str,
    capacity: int,
    selected_aliases: set[str],
) -> dict[str, Any]:
    metadata_dir = destination / "metadata" / task
    workspace_dir = destination / "workspaces" / task
    metadata_dir.mkdir(parents=True, exist_ok=False)
    workspace_dir.mkdir(parents=True, exist_ok=False)
    source_metadata = pool["metadata_dir"]
    # Mapping files are intentionally copied byte-for-byte and remain complete
    # so audit.source_size continues to describe the immutable source corpus.
    for name in ("source_doc_ids.json", "doc_id_files.json", "source_identity.json"):
        shutil.copyfile(source_metadata / name, metadata_dir / name)
    query_id = pool["query_id"]
    (metadata_dir / "queries.jsonl").write_bytes(pool["query_raw"])
    _write_json(metadata_dir / "original_qrels.json", _filter_qrels(pool["original_qrels"], query_id))
    _write_json(metadata_dir / "qrels.json", _filter_qrels(pool["qrels"], query_id))
    inventory_rows = pool["inventory_rows"]
    output_inventory: dict[str, Any] = {}
    for alias in sorted(selected_aliases):
        row = inventory_rows[alias]
        relative = f"workspaces/{task}/corpus/{pool['doc_id_files'][alias]}"
        output_inventory[relative] = copy.deepcopy(row)
    _write_json(metadata_dir / "file_inventory.json", output_inventory)

    source_manifest = dict(pool["manifest"])
    source_manifest["num_queries"] = 1
    # Keep corpus_size tied to the complete mapping: it is the source size
    # recorded in every retained candidate audit.  The bounded workspace size
    # is represented by file_inventory and completion.workspace_hashes.
    source_manifest["corpus_size"] = len(pool["mapping"])
    source_manifest["qrels_hash"] = stable_json_sha256(_filter_qrels(pool["qrels"], query_id))
    source_manifest["original_qrels_hash"] = stable_json_sha256(
        _filter_qrels(pool["original_qrels"], query_id)
    )
    source_manifest["queries_hash"] = stable_json_sha256([pool["query_row"]])
    if "num_qrels" in source_manifest:
        source_manifest["num_qrels"] = len(pool["original_qrels"][query_id])
    source_manifest["verification_policy"] = "content-hashes-verified"
    source_manifest["content_hashes_verified"] = True
    _write_json(metadata_dir / "manifest.json", source_manifest)

    candidate_payload: dict[str, Any] = {
        "capacity": capacity,
        "corpus_size": len(pool["mapping"]),
        "num_queries": 1,
        "pool_size": source_manifest.get("pool_size", CAPACITY_TO_POOL_SIZE[capacity]),
        "pools": {query_id: copy.deepcopy(pool["record"])},
        "schema_version": source_manifest.get("schema_version", POOL_SCHEMA_VERSION),
        "seed": source_manifest.get("seed"),
        "shared_corpus": "corpus",
        "source_identity": copy.deepcopy(pool["source_identity"]),
        "source_release": {
            "source_manifest_sha256": source_manifest.get("source_manifest_sha256"),
            "source_inventory_sha256": source_manifest.get("source_inventory_sha256"),
        },
        "task": task,
    }
    if pool["storage_encoding"] is not None:
        candidate_payload["storage_encoding"] = pool["storage_encoding"]
    _write_json(metadata_dir / "candidate_pools.json", candidate_payload)

    metadata_hashes = {
        name: sha256_file(metadata_dir / name) for name in POOL_METADATA_FILES
    }
    workspace_hashes = {
        relative: row["sha256"] for relative, row in sorted(output_inventory.items())
    }
    marker = {
        "schema_version": "dci-shared-pool-corpus-v1",
        "task": task,
        "corpus": f"source/workspaces/{task}/corpus",
    }
    # The marker is written by the caller after the task source corpus exists;
    # writing it here keeps every tier's completion anchor independent.
    _write_json(destination / SHARED_CORPUS_MARKER, marker)
    completion = {
        "content_hashes_verified": True,
        "file_inventory_sha256": sha256_file(metadata_dir / "file_inventory.json"),
        "metadata_hashes": metadata_hashes,
        "schema_version": POOL_SCHEMA_VERSION,
        "shared_corpus_sha256": sha256_file(destination / SHARED_CORPUS_MARKER),
        "task": task,
        "verification_policy": "content-hashes-verified",
        "workspace_hashes": workspace_hashes,
    }
    _write_json(metadata_dir / "_COMPLETE.json", completion)
    verify_document_pool_suite(destination, task=task, verify_content_hashes=True)
    return {
        "suite": destination.resolve(),
        "selected_alias_count": len(selected_aliases),
        "output_completion_sha256": sha256_file(metadata_dir / "_COMPLETE.json"),
        "output_candidate_sha256": sha256_file(metadata_dir / "candidate_pools.json"),
    }


def _ensure_output_scope(output: Path, catalog: Path, sources: Iterable[Mapping[str, Any]]) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output_resolved = output.resolve(strict=False)
    protected = [catalog.resolve(strict=False)]
    for entry in sources:
        for key in ("export",):
            value = entry.get(key)
            if isinstance(value, str):
                protected.append(Path(value).resolve(strict=False))
        for pool in entry.get("pools", []):
            if isinstance(pool, Mapping) and isinstance(pool.get("path"), str):
                protected.append(Path(pool["path"]).resolve(strict=False))
    for source in protected:
        if output_resolved == source or output_resolved in source.parents or source in output_resolved.parents:
            raise ValueError(f"output overlaps immutable source: {output} / {source}")


def prepare_pool_smoke(
    catalog: Path,
    output: Path,
    *,
    seed: int = DEFAULT_SEED,
    expected_sources: int | None = None,
) -> dict[str, Any]:
    """Create the projection and return its root manifest summary.

    ``expected_sources`` is intentionally an internal/testing hook.  The
    default enforces the production 53-source catalogue; fixtures may pass 1
    without changing the production CLI contract.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    catalog = Path(catalog).resolve()
    output = Path(output)
    catalog_payload = _json_load(catalog, label="catalog")
    if not isinstance(catalog_payload, Mapping) or not isinstance(catalog_payload.get("sources"), list):
        raise ValueError("catalog must contain a sources list")
    sources = list(catalog_payload["sources"])
    if expected_sources is None:
        expected_sources = 53
    if isinstance(expected_sources, bool) or not isinstance(expected_sources, int) or expected_sources <= 0:
        raise ValueError("expected_sources must be a positive integer")
    if len(sources) != expected_sources or catalog_payload.get("source_count") != expected_sources:
        raise ValueError(
            f"catalog source count mismatch: expected {expected_sources}, got {len(sources)}"
        )
    expected_pool_count = expected_sources * 3
    if catalog_payload.get("pool_count") != expected_pool_count:
        raise ValueError(f"catalog pool count mismatch: expected {expected_pool_count}")
    catalog_complete = catalog.parent / "_COMPLETE.json"
    catalog_completion = _json_load(catalog_complete, label="catalog completion")
    catalog_sha = sha256_file(catalog)
    if not isinstance(catalog_completion, Mapping) or catalog_completion.get("catalog_sha256") != catalog_sha:
        raise ValueError("catalog completion SHA-256 mismatch")
    _ensure_output_scope(output, catalog, sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False, exist_ok=False)

    plan_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    try:
        source_results: list[dict[str, Any]] = []
        seen_tasks: set[str] = set()
        for source_index, entry in enumerate(sources, 1):
            if not isinstance(entry, Mapping):
                raise ValueError("catalog source entry is not an object")
            source = entry.get("source")
            task = entry.get("task")
            if not isinstance(source, str) or not source or not isinstance(task, str) or not task:
                raise ValueError("catalog source/source task is invalid")
            if task in seen_tasks:
                raise ValueError(f"duplicate catalog task: {task}")
            seen_tasks.add(task)
            export_info = _load_source_export(entry, seed=seed)
            query_row = export_info["query_row"]
            query_raw = export_info["query_raw"]
            query_id = query_row.get("_id", query_row.get("query_id"))
            pools_by_capacity: dict[int, dict[str, Any]] = {}
            pools = entry.get("pools")
            if not isinstance(pools, list) or len(pools) != 3:
                raise ValueError(f"catalog source {task} must contain exactly 3 pools")
            for pool_entry in pools:
                if not isinstance(pool_entry, Mapping):
                    raise ValueError(f"catalog pool entry is not an object: {task}")
                capacity = pool_entry.get("capacity")
                if capacity not in CAPACITIES or capacity in pools_by_capacity:
                    raise ValueError(f"catalog source {task} has invalid/duplicate capacity")
                pools_by_capacity[capacity] = _validate_source_pool(
                    export_info,
                    pool_entry,
                    task=task,
                    capacity=capacity,
                    selected_query=query_row,
                    selected_query_raw=query_raw,
                )
            # The same source mapping and query are paired across all tiers.
            base_mapping = pools_by_capacity[200]["mapping"]
            base_query = pools_by_capacity[200]["query_row"]
            for capacity, pool in pools_by_capacity.items():
                if pool["mapping"] != base_mapping or pool["query_row"] != base_query:
                    raise ValueError(f"source tier pairing mismatch: {task}/{capacity}")
            selected_union = set().union(*(set(pool["record"]["candidate_doc_ids"]) for pool in pools_by_capacity.values()))
            source_results.append({
                "source": source,
                "task": task,
                "export_info": export_info,
                "pools": pools_by_capacity,
                "selected_union": selected_union,
            })
            print(f"[pool-smoke] verified source metadata {source_index}/{len(sources)}: {task}", file=sys.stderr, flush=True)

        for source_index, result in enumerate(source_results, 1):
            task = result["task"]
            pools = result["pools"]
            task_root = output / "tasks" / task
            source_workspace = task_root / "source" / "workspaces" / task / "corpus"
            source_workspace.mkdir(parents=True, exist_ok=False)
            mapping = pools[200]["mapping"]
            doc_id_files = pools[200]["doc_id_files"]
            for alias in sorted(result["selected_union"]):
                rows = [pool["inventory_rows"].get(alias) for pool in pools.values()]
                rows = [row for row in rows if row is not None]
                if not rows:
                    raise ValueError(f"no source inventory row for selected alias: {task}/{alias}")
                first = rows[0]
                if any(row["sha256"] != first["sha256"] or row["bytes"] != first["bytes"] for row in rows[1:]):
                    raise ValueError(f"tier inventory hash disagreement: {task}/{alias}")
                source_path = result["export_info"]["export"] / "workspaces" / task / "corpus" / doc_id_files[alias]
                _copy_verified(source_path, source_workspace / doc_id_files[alias], first["sha256"], first["bytes"])
            for capacity in CAPACITIES:
                pool = pools[capacity]
                destination = task_root / "pools" / str(capacity)
                projection = _build_pool_projection(
                    destination,
                    result["export_info"],
                    pool,
                    task=task,
                    capacity=capacity,
                    selected_aliases=set(pool["record"]["candidate_doc_ids"]),
                )
                plan_row = {
                    "task": task,
                    "source": result["source"],
                    "query_id": pool["query_id"],
                    "capacity": capacity,
                    "profile": CAPACITY_TO_PROFILE[capacity],
                    "suite": str(projection["suite"]),
                    "source_suite": str(pool["pool_suite"]),
                    "parent_completion_sha256": pool["completion_sha256"],
                    "pool_completion_sha256": projection["output_completion_sha256"],
                    "selected_record_sha256": pool["record_sha256"],
                }
                plan_rows.append(plan_row)
                audit_rows.append({
                    **plan_row,
                    "source_record_sha256": stable_json_sha256(pool["record"]),
                    "normalized_record_sha256": pool["record_sha256"],
                    "projected_record_sha256": pool["record_sha256"],
                    "normalized_summary_equal": True,
                    "candidate_file_sha256": pool["candidate_sha256"],
                    "selected_alias_count": len(pool["record"]["candidate_doc_ids"]),
                    "selected_union_alias_count": len(result["selected_union"]),
                    "source_metadata_complete_sha256": result["export_info"]["source_metadata_complete_sha256"],
                    "parent_migration": pool["migration"],
                    "original_full_content_hashes_verified": False,
                })
            print(f"[pool-smoke] published selected pools {source_index}/{len(sources)}: {task}", file=sys.stderr, flush=True)

        # Original IDs are task-local and may legitimately repeat across sources.
        unique_queries = {(row["task"], row["query_id"]) for row in plan_rows}
        if len(unique_queries) != len(source_results):
            raise ValueError("selected query IDs are not unique across sources")
        plan = {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "sources": len(source_results),
            "evaluations": len(plan_rows),
            "unique_queries": len(unique_queries),
            "seed": seed,
            "catalog_sha256": catalog_sha,
            "original_full_content_hashes_verified": False,
            "rows": sorted(plan_rows, key=lambda row: (row["task"], row["capacity"])),
        }
        audit = {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "catalog_sha256": catalog_sha,
            "seed": seed,
            "sources": len(source_results),
            "evaluations": len(plan_rows),
            "unique_queries": len(unique_queries),
            "original_full_content_hashes_verified": False,
            "rows": sorted(audit_rows, key=lambda row: (row["task"], row["capacity"])),
        }
        _write_json(output / "plan.json", plan)
        _write_json(output / "projection-audit.json", audit)
        completion = {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "status": "complete",
            "seed": seed,
            "catalog_sha256": catalog_sha,
            "sources": len(source_results),
            "evaluations": len(plan_rows),
            "unique_queries": len(unique_queries),
            "plan_sha256": sha256_file(output / "plan.json"),
            "projection_audit_sha256": sha256_file(output / "projection-audit.json"),
            "content_hashes_verified": True,
            "original_full_content_hashes_verified": False,
        }
        _write_json(output / "_COMPLETE.json", completion)
        return completion
    except BaseException as exc:
        try:
            _write_json(
                output / "_FAILED.json",
                {
                    "schema_version": PROJECTION_SCHEMA_VERSION,
                    "status": "failed",
                    "seed": seed,
                    "catalog": str(catalog),
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
        except BaseException:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        result = prepare_pool_smoke(args.catalog, args.output, seed=args.seed)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"[pool-smoke] preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


# Small import-friendly aliases for callers that refer to the operation as a
# projection rather than a smoke preparation.  They share the exact same
# bounded implementation and test-only expected_sources hook.
prepare_projection = prepare_pool_smoke
prepare_pool_smoke_projection = prepare_pool_smoke


if __name__ == "__main__":
    raise SystemExit(main())
