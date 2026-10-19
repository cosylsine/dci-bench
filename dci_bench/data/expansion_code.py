"""Immutable ARB file-view export for the independent host-side code scorer.

This adapter preserves the released ``kind=file`` text verbatim. ARB's producer
normalizes/strips/truncates that text; it is NOT a complete source checkout.
Source-line and function annotations therefore remain host-only raw evidence,
and do not become scoreable span/function labels. No document Pi settings change.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import tempfile
from typing import Any, Iterator

from dci_bench.protocol.code import CODE_LABELS_VERSION, CODE_TOKENIZER, code_protocol_contract
from dci_bench.scoring.code import file_metadata, load_json, validate_reference

ARB_DATASET = "eyuansu71/agent_retrieval_bench"
ARB_REVISION = "5901e1ee3aff048290db72edf9c63bc498b79ea3"
DEFAULT_RELEASES = ("v2_code2test", "v2_comment2context", "v2_trace2code")
RELEASES = {
    "v2_code2test": (106, "387cdfb25835b176ba27707d43dc31cae6b31514231923af237ad04deef19c0b"),
    "v2_comment2context": (80, "be68161844d63c64e687e0c6da1d52434160fafcaf8932a5767fbe64b643b3f5"),
    "v2_trace2code": (101, "19b252e8cfff42107fedc74005dbb6972f2970af33651ce0c1571546819e41c4"),
}
SCOPE = "arb_file_view"
MAX_FILE_BYTES = 1_000_000
MAX_JSONL_BYTES = 16_000_000
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class CodeExpansionError(ValueError):
    """An archive or reference cannot be faithfully exported."""


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _safe_path(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(p in ("", ".", "..") for p in value.split("/"))
            or re.match(r"^[A-Za-z]:", value)):
        raise CodeExpansionError(f"unsafe repository/archive path: {value!r}")
    return value


def _member_path(member: tarfile.TarInfo) -> str:
    name = member.name
    if member.isdir() and name in (".", "./"):
        return ""
    while name.startswith("./"):
        name = name[2:]
    if member.isdir():
        name = name.rstrip("/")
        if not name:
            return ""
    _safe_path(name)
    if not (member.isfile() or member.isdir()) or member.issparse():
        raise CodeExpansionError(f"unsupported archive entry (links forbidden): {member.name!r}")
    return name


@contextmanager
def _archive(path: Path) -> Iterator[tarfile.TarFile]:
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("Reading ARB tar.zst needs the zstandard Python package") from exc
    with path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as archive:
            yield archive


def _rows(stream: Any) -> Iterator[dict[str, Any]]:
    # Bound individual records, while preserving the text value inside JSON.
    while True:
        line = stream.readline(MAX_JSONL_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_JSONL_BYTES:
            raise CodeExpansionError("JSONL record exceeds safety limit")
        if not line.strip():
            continue
        row = load_json(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise CodeExpansionError("JSONL rows must be objects")
        yield row


def _snapshot_key(repo: str, commit: str) -> str:
    if not isinstance(repo, str) or not repo.strip() or not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise CodeExpansionError("snapshot requires repository and full lowercase commit")
    return _hash([repo, commit])


def _paths(values: Any) -> list[str]:
    if not isinstance(values, list):
        raise CodeExpansionError("gold paths must be a list")
    result = []
    for value in values:
        path = value if isinstance(value, str) else value.get("path") if isinstance(value, dict) else None
        _safe_path(path)
        if path not in result:
            result.append(path)
    return result


def _gold(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") or {}
    if not isinstance(gold, dict) or gold.get("no_gold") is True:
        raise CodeExpansionError("three positive ARB subsets must have positive gold")
    if gold.get("files"):
        return _paths(gold["files"])
    task = sample.get("task_type")
    if task == "code2test":
        return _paths(gold.get("related_tests") or [])
    if task == "comment2context":
        values = gold.get("must_context_files") or gold.get("context_files")
        if values:
            return _paths(values)
    return _paths(gold.get("root_cause_files") or gold.get("related_tests") or [])


def _span_reason(sample: dict[str, Any], files: dict[str, Any], gold: list[str]) -> str:
    spans = sample.get("gold_spans", (sample.get("gold") or {}).get("gold_spans"))
    if not spans:
        return "source_spans_absent"
    if not isinstance(spans, list):
        return "source_spans_malformed"
    for span in spans:
        if not isinstance(span, dict):
            return "source_spans_malformed"
        path, start, end = span.get("path"), span.get("start_line"), span.get("end_line")
        if path not in gold or type(start) is not int or type(end) is not int or not 1 <= start <= end:
            return "source_spans_malformed"
        if end > files[path]["line_count"]:
            return "source_spans_exceed_released_file_view"
    return "source_line_correspondence_unverified"


class _NativeWriter:
    """Bounded parallel filesystem writes; hashing and ordering stay on caller."""

    def __init__(self, workers: int):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arb-write")
        self.pending = deque()
        self.limit = workers * 4

    @staticmethod
    def _write_bytes(destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)

    def write(self, destination: Path, content: bytes) -> None:
        self.pending.append(self.executor.submit(self._write_bytes, destination, content))
        if len(self.pending) >= self.limit:
            self.pending.popleft().result()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        failed = exc_type is not None
        try:
            if not failed:
                while self.pending:
                    self.pending.popleft().result()
        except BaseException:
            failed = True
            raise
        finally:
            # All running writes finish before build_arb can clean its staging tree.
            self.executor.shutdown(wait=True, cancel_futures=failed)


def _export_chunks(stream: Any, stage: Path, snapshots: dict[str, Any], release: str,
                   member: str, writer: _NativeWriter) -> None:
    files, bounds = {}, {}
    repo = commit = key = None
    for row in _rows(stream):
        row_repo, row_commit = row.get("repo"), row.get("base_commit")
        row_key = _snapshot_key(row_repo, row_commit)
        if key is None:
            repo, commit, key = row_repo, row_commit, row_key
        if row_key != key:
            raise CodeExpansionError(f"mixed snapshots in {member}")
        if row.get("kind") != "file":
            continue
        path = _safe_path(row.get("path"))
        text = row.get("text")
        if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_FILE_BYTES:
            raise CodeExpansionError(f"file view missing text or exceeds 1 MB: {path}")
        if path in files:
            raise CodeExpansionError(f"duplicate file row: {path}")
        files[path] = file_metadata(text)
        bounds[path] = {"start_line": row.get("start_line"), "end_line": row.get("end_line"),
                        "truncation_marker": text.endswith("...[truncated]"),
                        "characters": len(text)}
        if key not in snapshots:
            destination = stage / "snapshots" / key / "corpus" / path
            writer.write(destination, text.encode("utf-8"))
    if not key or not files:
        raise CodeExpansionError(f"no canonical file records in {member}")
    digest = _hash(files)
    if key in snapshots:
        if snapshots[key]["corpus_sha256"] != digest:
            raise CodeExpansionError(f"conflicting corpus snapshots: {repo}@{commit}")
        snapshots[key]["source_releases"].append(release)
        return
    snapshot = {"snapshot_id": key, "repository": repo, "base_commit": commit,
                "scope": SCOPE, "corpus_sha256": digest, "files": files,
                "source_releases": [release], "source_member": member,
                "source_line_correspondence_verified": False,
                "upstream_file_view": bounds,
                "truncated_file_count": sum(v["truncation_marker"] for v in bounds.values())}
    snapshots[key] = snapshot


def load_reference(label: dict[str, Any], snapshots_root: Path) -> dict[str, Any]:
    """Expand one compact host label into a validated code-scoring reference."""
    key = label["snapshot_id"]
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
        raise CodeExpansionError("invalid snapshot ID")
    snapshot = load_json((Path(snapshots_root) / key / "manifest.json").read_text(encoding="utf-8"))
    if (snapshot["snapshot_id"] != key or _snapshot_key(snapshot["repository"], snapshot["base_commit"]) != key
            or snapshot["corpus_sha256"] != _hash(snapshot["files"])
            or label["corpus_sha256"] != snapshot["corpus_sha256"]):
        raise CodeExpansionError("snapshot identity/hash mismatch")
    reference = {"repository": snapshot["repository"], "base_commit": snapshot["base_commit"],
                 "task": label["task"], "files": snapshot["files"],
                 "gold_files": label["gold_files"], "tokenizer": CODE_TOKENIZER}
    return validate_reference(reference)


def _regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CodeExpansionError(f"artifact must be a regular unlinked file: {path}")


class _VerificationPool:
    """Bound independent file stat/read checks without changing audit ordering."""

    def __init__(self, workers: int):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arb-verify")
        self.pending = deque()
        self.limit = workers * 4

    @staticmethod
    def _check(path: Path, metadata: dict[str, Any] | None, view: dict[str, Any] | None) -> None:
        _regular(path)
        if metadata is None:
            return
        content = path.read_bytes()
        if len(content) > MAX_FILE_BYTES:
            raise CodeExpansionError(f"candidate exceeds file size limit: {path}")
        text = content.decode("utf-8")
        if file_metadata(text) != metadata:
            raise CodeExpansionError(f"corpus bytes/metadata mismatch: {path}")
        if len(text) != view["characters"] or text.endswith("...[truncated]") != view["truncation_marker"]:
            raise CodeExpansionError(f"file-view disclosure mismatch: {path}")

    def check(self, path: Path, metadata: dict[str, Any] | None = None,
              view: dict[str, Any] | None = None) -> None:
        self.pending.append(self.executor.submit(self._check, path, metadata, view))
        if len(self.pending) >= self.limit:
            self.pending.popleft().result()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        failed = exc_type is not None
        try:
            if not failed:
                while self.pending:
                    self.pending.popleft().result()
        except BaseException:
            failed = True
            raise
        finally:
            self.executor.shutdown(wait=True, cancel_futures=failed)


def _tree_files(root: Path, *, io_workers: int = 32) -> set[str]:
    """List an artifact tree while rejecting symbolic/hard links and devices."""
    if root.is_symlink() or not root.is_dir():
        raise CodeExpansionError(f"artifact root must be a real directory: {root}")
    files = set()
    with _VerificationPool(io_workers) as checker:
        for directory, subdirs, names in os.walk(root, followlinks=False):
            for name in subdirs:
                path = Path(directory) / name
                if not stat.S_ISDIR(path.lstat().st_mode):
                    raise CodeExpansionError(f"linked/non-directory artifact path: {path}")
            for name in names:
                path = Path(directory) / name
                checker.check(path)
                files.add(path.relative_to(root).as_posix())
    return files


def verify_arb(output: Path, verify_corpus: bool = True, *, io_workers: int = 32) -> dict[str, Any]:
    """Audit the completion/hash chain, candidate bytes, and query/reference closure.

    ``verify_corpus=False`` skips content reads only: file path closure and
    symlink/hardlink checks still run. It must not be reported as byte validation.
    """
    output = Path(output)
    if type(io_workers) is not int or not 1 <= io_workers <= 256:
        raise CodeExpansionError("io_workers must be an integer from 1 through 256")
    actual_paths = _tree_files(output, io_workers=io_workers)
    for name in ("_COMPLETE.json", "manifest.json"):
        _regular(output / name)
    complete = load_json((output / "_COMPLETE.json").read_text(encoding="utf-8"))
    if complete["manifest_sha256"] != _sha(output / "manifest.json"):
        raise CodeExpansionError("completion manifest hash mismatch")
    manifest = load_json((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["schema_version"] != "dci-arb-file-view-v1" or manifest["scope"] != SCOPE
            or manifest["dataset"] != ARB_DATASET or manifest["revision"] != ARB_REVISION
            or manifest["source_complete"] is not False
            or manifest["source_line_correspondence_verified"] is not False):
        raise CodeExpansionError("unexpected file-view export identity")
    artifacts = {"queries.json", "host/labels.json", "host/raw_samples.json", "protocol.json"}
    if set(manifest["artifact_hashes"]) != artifacts:
        raise CodeExpansionError("artifact hash catalog differs from expected layout")
    for name in artifacts:
        if _sha(output / name) != manifest["artifact_hashes"][name]:
            raise CodeExpansionError(f"artifact hash mismatch: {name}")
    if load_json((output / "protocol.json").read_text(encoding="utf-8")) != code_protocol_contract():
        raise CodeExpansionError("exported protocol differs from current scoring contract")
    snapshot_hashes = manifest["snapshot_hashes"]
    if set(snapshot_hashes) != set(manifest["snapshot_manifest_sha256"]):
        raise CodeExpansionError("snapshot catalogs disagree")
    expected_paths = artifacts | {"manifest.json", "_COMPLETE.json"}
    snapshots = {}
    file_count = truncated_count = 0
    with _VerificationPool(io_workers) as checker:
        for key, digest in snapshot_hashes.items():
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                raise CodeExpansionError("invalid snapshot catalog key")
            relative = f"snapshots/{key}/manifest.json"
            if _sha(output / relative) != manifest["snapshot_manifest_sha256"][key]:
                raise CodeExpansionError(f"snapshot manifest hash mismatch: {key}")
            expected_paths.add(relative)
            snapshot = load_json((output / relative).read_text(encoding="utf-8"))
            if (snapshot["snapshot_id"] != key
                    or _snapshot_key(snapshot["repository"], snapshot["base_commit"]) != key
                    or snapshot["corpus_sha256"] != digest or _hash(snapshot["files"]) != digest
                    or snapshot["scope"] != SCOPE
                    or snapshot["source_line_correspondence_verified"] is not False):
                raise CodeExpansionError(f"snapshot identity/hash mismatch: {key}")
            if set(snapshot["files"]) != set(snapshot["upstream_file_view"]):
                raise CodeExpansionError("source file-view metadata differs from candidate paths")
            # Validate every snapshot manifest, including unused released snapshots.
            validate_reference({"repository": snapshot["repository"], "base_commit": snapshot["base_commit"],
                                "task": "manifest_validation", "files": snapshot["files"],
                                "gold_files": [next(iter(snapshot["files"]), "")], "tokenizer": CODE_TOKENIZER})
            truncated = 0
            for path, metadata in snapshot["files"].items():
                _safe_path(path)
                candidate = f"snapshots/{key}/corpus/{path}"
                expected_paths.add(candidate)
                view = snapshot["upstream_file_view"][path]
                truncated += bool(view["truncation_marker"])
                if verify_corpus:
                    checker.check(output / candidate, metadata, view)
            if truncated != snapshot["truncated_file_count"]:
                raise CodeExpansionError("snapshot truncation count mismatch")
            truncated_count += truncated
            file_count += len(snapshot["files"])
            snapshots[key] = snapshot
    if actual_paths != expected_paths:
        raise CodeExpansionError(f"artifact/corpus path set mismatch: missing={sorted(expected_paths-actual_paths)[:3]}, "
                                 f"unexpected={sorted(actual_paths-expected_paths)[:3]}")
    compact = load_json((output / "host/labels.json").read_text(encoding="utf-8"))
    raw = load_json((output / "host/raw_samples.json").read_text(encoding="utf-8"))
    queries = load_json((output / "queries.json").read_text(encoding="utf-8"))
    if compact["schema_version"] != "dci-code-compact-labels-v1":
        raise CodeExpansionError("unknown compact labels schema")
    labels = compact["queries"]
    if not isinstance(queries, list) or any(not isinstance(q, dict) for q in queries):
        raise CodeExpansionError("queries must be a list of objects")
    ids = [q["query_id"] for q in queries]
    if len(ids) != len(set(ids)) or set(ids) != set(labels) or set(ids) != set(raw):
        raise CodeExpansionError("query/raw/reference IDs do not close")
    reasons = {}
    for query in queries:
        query_id = query["query_id"]
        label, sample = labels[query_id], raw[query_id]
        key = label["snapshot_id"]
        if key not in snapshots or sample["id"] != query_id:
            raise CodeExpansionError("query refers to an unknown snapshot or source ID")
        ref = load_reference(label, output / "snapshots")
        if (key != _snapshot_key(sample["repo"], sample["base_commit"])
                or ref["gold_files"] != _gold(sample) or ref["task"] != sample["task_type"]
                or query != {"query_id": query_id, "snapshot_id": key, "task": sample["task_type"],
                             "query": json.dumps(sample.get("query") or {}, ensure_ascii=False, sort_keys=True)}):
            raise CodeExpansionError("query/reference differs from preserved source sample")
        reason = _span_reason(sample, ref["files"], ref["gold_files"])
        if label["span_status"] != {"available": False, "reason": reason}:
            raise CodeExpansionError("optional span disclosure differs from source")
        reasons[reason] = reasons.get(reason, 0) + 1
    if (len(labels) != manifest["samples"] or len(labels) != complete["samples"]
            or len(snapshots) != manifest["snapshots"] or len(snapshots) != complete["snapshots"]
            or file_count != manifest["files"] or truncated_count != manifest["truncated_files"]
            or reasons != manifest["optional_span_status"]):
        raise CodeExpansionError("completion/manifest counts disagree")
    return {"verified": True, "corpus_bytes_verified": verify_corpus,
            "samples": len(labels), "snapshots": len(snapshots), "files": file_count,
            "truncated_files": truncated_count, "scope": SCOPE}


def export_scoring_labels(output: Path, destination: Path, *, io_workers: int = 32) -> None:
    """Materialize scorer CLI labels on demand; destination must be new."""
    output, destination = Path(output), Path(destination)
    if destination.resolve().is_relative_to(output.resolve()):
        raise CodeExpansionError("scoring labels must be written outside the immutable export")
    if not (output / "_COMPLETE.json").is_file():
        raise CodeExpansionError("cannot score an incomplete export")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to alter existing scoring labels: {destination}")
    verify_arb(output, io_workers=io_workers)
    labels = load_json((output / "host" / "labels.json").read_text(encoding="utf-8"))
    # Stream one expanded reference at a time to avoid holding repeated manifests.
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write('{"schema_version":' + json.dumps(CODE_LABELS_VERSION) + ',"queries":{')
            for index, (query_id, label) in enumerate(labels["queries"].items()):
                if index:
                    stream.write(",")
                stream.write(json.dumps(query_id) + ":")
                json.dump(load_reference(label, output / "snapshots"), stream, ensure_ascii=False, sort_keys=True)
            stream.write("}}\n")
        # Atomic publication without replacing an existing destination.
        os.link(temporary, destination)
    finally:
        temporary.unlink()


def build_arb(raw: Path, output: Path, revision: str = ARB_REVISION,
              releases: tuple[str, ...] = DEFAULT_RELEASES, *, io_workers: int = 32) -> dict[str, Any]:
    """Verify official archives and atomically export immutable native file views."""
    raw, output = Path(raw), Path(output)
    if type(io_workers) is not int or not 1 <= io_workers <= 256:
        raise CodeExpansionError("io_workers must be an integer from 1 through 256")
    if revision != ARB_REVISION:
        raise CodeExpansionError("revision must match the audited official archive catalog")
    if not releases or len(set(releases)) != len(releases) or set(releases) - RELEASES.keys():
        raise CodeExpansionError("choose unique audited positive ARB release IDs")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to alter existing export: {output}")
    sources = []
    for release in releases:
        relative = f"releases/{release}/agent_retrieval_bench_{release}.tar.zst"
        archive = raw / relative
        actual = _sha(archive)
        if actual != RELEASES[release][1]:
            raise CodeExpansionError(f"official archive SHA256 mismatch: {relative}")
        sources.append({"release": release, "file": relative, "sha256": actual,
                        "bytes": archive.stat().st_size})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        snapshots, samples = {}, {}
        with _NativeWriter(io_workers) as writer:
            for source in sources:
                release = source["release"]
                count = 0
                seen_members = set()
                with _archive(raw / source["file"]) as archive:
                    for member in archive:
                        name = _member_path(member)
                        if not name or member.isdir():
                            continue
                        if name in seen_members:
                            raise CodeExpansionError(f"duplicate archive member: {name}")
                        seen_members.add(name)
                        # Public bundles may include a leading data/ directory.
                        logical = name[5:] if name.startswith("data/") else name
                        sample_path = f"benchmark/{release}/samples.jsonl"
                        is_chunks = logical.startswith(f"corpus/{release}/") and logical.endswith(".chunks.jsonl")
                        if logical != sample_path and not is_chunks:
                            continue
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise CodeExpansionError(f"missing archive stream: {name}")
                        with stream:
                            if is_chunks:
                                _export_chunks(stream, stage, snapshots, release, name, writer)
                            else:
                                for sample in _rows(stream):
                                    query_id = sample.get("id")
                                    if not isinstance(query_id, str) or not query_id or query_id in samples:
                                        raise CodeExpansionError("missing or duplicate sample ID")
                                    if sample.get("task_type") != release.removeprefix("v2_"):
                                        raise CodeExpansionError("sample task does not match release")
                                    samples[query_id] = sample
                                    count += 1
                if count != RELEASES[release][0]:
                    raise CodeExpansionError(f"{release}: expected {RELEASES[release][0]} samples, found {count}")
        for key, snapshot in snapshots.items():
            _write(stage / "snapshots" / key / "manifest.json", snapshot)
        compact, queries, reasons = {}, [], {}
        for query_id, sample in sorted(samples.items()):
            key = _snapshot_key(sample.get("repo"), sample.get("base_commit"))
            if key not in snapshots:
                raise CodeExpansionError(f"sample missing snapshot: {query_id}")
            snapshot = snapshots[key]
            gold = _gold(sample)
            label = {"snapshot_id": key, "corpus_sha256": snapshot["corpus_sha256"],
                     "task": sample["task_type"], "gold_files": gold}
            # Validate file labels before considering optional source annotations.
            load_reference(label, stage / "snapshots")
            reason = _span_reason(sample, snapshot["files"], gold)
            label["span_status"] = {"available": False, "reason": reason}
            reasons[reason] = reasons.get(reason, 0) + 1
            compact[query_id] = label
            queries.append({"query_id": query_id, "snapshot_id": key, "task": sample["task_type"],
                            "query": json.dumps(sample.get("query") or {}, ensure_ascii=False, sort_keys=True)})
        _write(stage / "host" / "labels.json", {"schema_version": "dci-code-compact-labels-v1", "queries": compact})
        _write(stage / "host" / "raw_samples.json", samples)
        _write(stage / "queries.json", queries)
        _write(stage / "protocol.json", code_protocol_contract())
        manifest = {"schema_version": "dci-arb-file-view-v1", "dataset": ARB_DATASET,
                    "revision": revision, "scope": SCOPE, "source_archives": sources,
                    "samples": len(samples), "snapshots": len(snapshots),
                    "files": sum(len(v["files"]) for v in snapshots.values()),
                    "truncated_files": sum(v["truncated_file_count"] for v in snapshots.values()),
                    "snapshot_hashes": {k: v["corpus_sha256"] for k, v in sorted(snapshots.items())},
                    "snapshot_manifest_sha256": {k: _sha(stage / "snapshots" / k / "manifest.json")
                                                 for k in sorted(snapshots)},
                    "optional_span_status": reasons, "max_file_bytes": MAX_FILE_BYTES,
                    "source_complete": False, "source_line_correspondence_verified": False,
                    "candidate_policy": "all released kind=file records per repository/base_commit; no task filtering",
                    "text_policy": "released text encoded UTF-8 verbatim; upstream strips/normalizes/truncates source",
                    "runtime_status": "host-side artifacts only; no code agent runtime configured",
                    "licenses": "benchmark metadata MIT; corpus retains each upstream repository license",
                    "host_only": ["host/labels.json", "host/raw_samples.json"],
                    "artifact_hashes": {p: _sha(stage / p) for p in
                                        ("queries.json", "host/labels.json", "host/raw_samples.json", "protocol.json")}}
        _write(stage / "manifest.json", manifest)
        # Only publish a completed directory after every reference has validated.
        _write(stage / "_COMPLETE.json", {"manifest_sha256": _sha(stage / "manifest.json"),
                                          "samples": len(samples), "snapshots": len(snapshots)})
        verify_arb(stage, io_workers=io_workers)
        if os.path.lexists(output):
            raise FileExistsError(f"export appeared during build: {output}")
        os.rename(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-arb")
    build.add_argument("--raw", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--revision", default=ARB_REVISION)
    build.add_argument("--io-workers", type=int, default=32, help="bounded native-file writer threads (default: 32)")
    build.add_argument("--releases", nargs="+", choices=tuple(RELEASES), default=DEFAULT_RELEASES)
    expand = commands.add_parser("export-labels")
    expand.add_argument("--output", type=Path, required=True)
    expand.add_argument("--destination", type=Path, required=True)
    expand.add_argument("--io-workers", type=int, default=32)
    verify = commands.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--skip-corpus-bytes", action="store_true")
    verify.add_argument("--io-workers", type=int, default=32)
    args = parser.parse_args()
    if args.command == "build-arb":
        result = build_arb(args.raw, args.output, args.revision, tuple(args.releases), io_workers=args.io_workers)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "export-labels":
        export_scoring_labels(args.output, args.destination, io_workers=args.io_workers)
    else:
        print(json.dumps(verify_arb(args.output, not args.skip_corpus_bytes, io_workers=args.io_workers), sort_keys=True))


if __name__ == "__main__":
    main()
