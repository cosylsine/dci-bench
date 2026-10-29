"""Pinned source connectors and the canonical DCI data bundle.

The connector boundary is deliberately small.  A connector first materialises
an immutable raw snapshot through :func:`fetch_source`, then parses that
snapshot through :func:`load_source`.  Parsing never contacts the network, so a
bundle can be audited and reproduced after the source has disappeared.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .download import (
    DEFAULT_MAX_ARCHIVE_BYTES,
    DEFAULT_MAX_ARCHIVE_FILES,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_COMPRESSION_RATIO,
    DEFAULT_MAX_MEMBER_BYTES,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    ArchiveSecurityError,
    DownloadError,
    IntegrityError,
    SnapshotConflictError,
    canonical_json,
    copy_file,
    download_url,
    file_digest,
    nonblocking_file_lock,
    redact_secrets,
    safe_extract_archive,
    safe_relative_path,
    sha256_json,
    verify_file_inventory,
)


SCHEMA_VERSION = "dci-data-bundle-v1"
INPUT_KINDS = frozenset({"retrieval", "evidence_qa", "corpus"})
CONNECTOR_NAMES = (
    "beir",
    "edgar",
    "eurlex",
    "generic_jsonl",
    "github_markdown",
    "pmc_jats",
    "qasper",
    "squad",
    "wixqa",
    "worksurface",
)
_ROLE_NAMES = ("documents", "queries", "qrels", "evidence", "judgments")
_FLOATING_REVISIONS = {"", "latest", "main", "master", "default", "head", "tip"}
_FILE_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".jsonl",
    ".gz",
    ".tsv",
}


class ConnectorError(RuntimeError):
    """A source cannot be safely fetched or interpreted."""


class UnsupportedSourceError(ConnectorError):
    """A connector option describes a capability this compact adapter lacks."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _spec_fingerprint(spec: Mapping[str, Any]) -> str:
    return sha256_json(_jsonable(spec))


def _require_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a mapping")
    result = _jsonable(dict(spec))
    missing = [key for key in ("id", "connector", "input_kind") if not result.get(key)]
    if missing:
        raise ValueError(f"source spec missing required fields: {', '.join(missing)}")
    connector = str(result["connector"])
    if connector not in CONNECTOR_NAMES:
        raise ValueError(f"unknown source connector {connector!r}; available: {list_connectors()}")
    if str(result["input_kind"]) not in INPUT_KINDS:
        raise ValueError(f"input_kind must be one of {sorted(INPUT_KINDS)}")
    options = result.get("options")
    if options is None:
        result["options"] = {}
    elif not isinstance(options, Mapping):
        raise ValueError("spec.options must be an object")
    else:
        result["options"] = dict(options)
    return result


def _safe_component(value: Any) -> str:
    text = str(value)
    if not text:
        raise ValueError("source id must not be empty")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not text:
        text = "source"
    return text[:120]


def _is_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return urllib.parse.urlsplit(value).scheme in {"http", "https"}


def _as_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.startswith("file://"):
        parsed = urllib.parse.urlsplit(value)
        return Path(urllib.request.url2pathname(parsed.path))
    return Path(os.fspath(value))


def _local_or_url(value: Any, *, base: Path | None = None) -> str | Path:
    if _is_url(value):
        return str(value)
    path = _as_path(value)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _normalise_role(role: Any, name: str = "") -> str | None:
    key = str(role or "").lower().replace("-", "_")
    if key in _ROLE_NAMES:
        return key
    lower = name.lower()
    if any(token in lower for token in ("qrel", "label", "relevance")):
        return "qrels"
    if "evidence" in lower or "annotation" in lower:
        return "evidence"
    if "query" in lower or "queries" in lower or lower.startswith("questions"):
        return "queries"
    if any(token in lower for token in ("corpus", "document", "docs", "passage")):
        return "documents"
    if "judgment" in lower or "judgement" in lower:
        return "judgments"
    return None


def _default_name(role: str, source: Any, index: int = 0) -> str:
    suffix = Path(str(source)).name or f"resource-{index}"
    if role == "documents":
        return "documents.jsonl" if not suffix else suffix
    if role == "queries":
        return "queries.jsonl" if not suffix else suffix
    if role == "qrels":
        return suffix if suffix else "qrels.json"
    if role == "evidence":
        return suffix if suffix else "evidence.jsonl"
    if role == "judgments":
        return suffix if suffix else "judgments.jsonl"
    return suffix


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".gz" or path.name.lower().endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = handle.read()
    else:
        data = path.read_text(encoding="utf-8")
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        return rows


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    data = _read_json_or_jsonl(path)
    if isinstance(data, Mapping):
        # A mapping can be a qrels object, but a document/query JSON file must
        # still expose a row.  Accept common wrapper keys for convenience.
        for key in ("documents", "corpus", "queries", "questions", "rows", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            # A one-line JSONL file is a valid single-row source.
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array or JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(data, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} in {path} is not an object")
        rows.append(dict(row))
    return rows


def _id_from_row(row: Mapping[str, Any], *, kind: str, index: int, path: Path) -> str:
    keys = {
        "document": ("_id", "id", "doc_id", "document_id", "corpus_id"),
        "query": ("_id", "id", "query_id", "qid"),
        "evidence": ("_id", "id", "evidence_id"),
    }[kind]
    for key in keys:
        if key in row:
            value = row[key]
            if value is None:
                return ""
            return str(value)
    raise ValueError(f"row {index} in {path} has no {kind} ID")


def _metadata(row: Mapping[str, Any], reserved: set[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    supplied = row.get("metadata")
    if isinstance(supplied, Mapping):
        metadata.update(_jsonable(dict(supplied)))
    for key, value in row.items():
        if key not in reserved and key != "metadata":
            metadata[str(key)] = _jsonable(value)
    return metadata


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_text(item) for item in value)
    return str(value)


def _canonical_documents(rows: Iterable[Mapping[str, Any]], path: Path, language: str | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        document_id = _id_from_row(row, kind="document", index=index, path=path)
        if document_id in seen:
            raise ValueError(f"duplicate document ID {document_id!r} in {path}")
        seen.add(document_id)
        reserved = {
            "_id",
            "id",
            "doc_id",
            "document_id",
            "corpus_id",
            "title",
            "text",
            "content",
            "body",
            "language",
            "metadata",
        }
        output.append(
            {
                "_id": document_id,
                "title": _text(row.get("title", "")),
                "text": _text(row.get("text", row.get("content", row.get("body", "")))),
                "language": str(row.get("language") or language or ""),
                "metadata": _metadata(row, reserved),
            }
        )
    return output


def _canonical_queries(rows: Iterable[Mapping[str, Any]], path: Path, language: str | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        query_id = _id_from_row(row, kind="query", index=index, path=path)
        if query_id in seen:
            raise ValueError(f"duplicate query ID {query_id!r} in {path}")
        seen.add(query_id)
        reserved = {"_id", "id", "query_id", "qid", "text", "question", "language", "metadata"}
        output.append(
            {
                "_id": query_id,
                "text": _text(row.get("text", row.get("question", ""))),
                "language": str(row.get("language") or language or ""),
                "metadata": _metadata(row, reserved),
            }
        )
    return output


def _number(value: Any, *, where: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"qrel score at {where} is not numeric: {value!r}") from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"qrel score at {where} is not finite")
    return number


def _put_qrel(qrels: dict[str, dict[str, float]], query_id: Any, document_id: Any, score: Any, where: str) -> None:
    qid = str(query_id)
    did = str(document_id)
    per_query = qrels.setdefault(qid, {})
    if did in per_query:
        raise ValueError(f"duplicate qrel pair at {where}: {qid!r}, {did!r}")
    per_query[did] = _number(score, where=where)


def _parse_qrels(path: Path) -> dict[str, dict[str, float]]:
    if path.suffix.lower() in {".json", ".jsonl", ".gz"} or path.name.endswith(".json.gz"):
        data = _read_json_or_jsonl(path)
        if isinstance(data, Mapping):
            result: dict[str, dict[str, float]] = {}
            for query_id, docs in data.items():
                if not isinstance(docs, Mapping):
                    raise ValueError(f"qrels for {query_id!r} must be an object")
                for document_id, score in docs.items():
                    _put_qrel(result, query_id, document_id, score, str(path))
            return result
        if isinstance(data, list):
            result = {}
            for index, row in enumerate(data, 1):
                if not isinstance(row, Mapping):
                    raise ValueError(f"qrels row {index} in {path} is not an object")
                query_id = row.get("query_id", row.get("qid", row.get("query-id")))
                document_id = row.get("doc_id", row.get("corpus_id", row.get("corpus-id")))
                _put_qrel(result, query_id, document_id, row.get("score", row.get("relevance", 1)), f"{path}:{index}")
            return result
        raise ValueError(f"unsupported qrels JSON shape: {path}")
    result: dict[str, dict[str, float]] = {}
    raw = path.read_text(encoding="utf-8")
    for index, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) == 1:
            fields = line.split()
        if index == 1 and fields and fields[0].lower() in {"query-id", "query_id", "qid"}:
            continue
        if len(fields) == 3:
            query_id, document_id, score = fields
        elif len(fields) >= 4:
            # TREC/BEIR form: query-id, iteration, corpus-id, score.
            query_id, document_id, score = fields[0], fields[-2], fields[-1]
        else:
            raise ValueError(f"qrels row {index} in {path} has fewer than three fields")
        _put_qrel(result, query_id, document_id, score, f"{path}:{index}")
    return result


def _canonical_evidence(path: Path) -> list[dict[str, Any]]:
    rows = _iter_jsonl(path)
    evidence: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        query_id = row.get("query_id", row.get("qid", row.get("query-id")))
        document_id = row.get("doc_id", row.get("document_id", row.get("docid", row.get("corpus-id"))))
        if query_id is None or document_id is None:
            raise ValueError(f"evidence row {index} in {path} has no query/document ID")
        result: dict[str, Any] = {
            "query_id": str(query_id),
            "doc_id": str(document_id),
        }
        for key in ("quote", "start", "end"):
            if key in row:
                result[key] = _jsonable(row[key])
        for key, value in row.items():
            if key not in {"query_id", "qid", "query-id", "doc_id", "document_id", "docid", "corpus-id", "quote", "start", "end"}:
                result[key] = _jsonable(value)
        evidence.append(result)
    return evidence


def _find_all_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and not path.name.endswith(".partial")
    )


@dataclass
class _FetchState:
    spec: dict[str, Any]
    root: Path
    actual_urls: list[str] = field(default_factory=list)
    layout: dict[str, list[str]] = field(default_factory=dict)

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.spec.get("options") or {})

    def _target(self, relative: str) -> tuple[Path, str]:
        safe = safe_relative_path(relative)
        destination = (self.root / safe).resolve()
        try:
            destination.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ConnectorError(f"source path escapes snapshot: {relative!r}") from exc
        return destination, safe

    def add(self, source: str | Path, relative: str, *, role: str | None = None, expected_sha256: str | None = None, expected_bytes: int | None = None) -> Path:
        destination, safe = self._target(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        limits = self.options
        max_bytes = int(limits.get("max_bytes", DEFAULT_MAX_BYTES))
        if _is_url(source):
            # Run URL values through the URL-aware redactor.  A bare string has
            # no key name, so ``redact_secrets(url)`` alone cannot know that a
            # query parameter such as ``?token=...`` is sensitive.
            redacted = redact_secrets({"url": str(source)})["url"]
            self.actual_urls.append(redacted)
            download_url(
                str(source),
                destination,
                timeout=float(limits.get("timeout", DEFAULT_TIMEOUT)),
                retries=int(limits.get("retries", DEFAULT_RETRIES)),
                max_bytes=max_bytes,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
                headers=limits.get("headers") if isinstance(limits.get("headers"), Mapping) else None,
                overwrite=True,
            )
        else:
            copy_file(
                _as_path(source),
                destination,
                max_bytes=max_bytes,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
                overwrite=True,
            )
        if role:
            self.layout.setdefault(role, []).append(safe)
        return destination

    def add_tree(self, source: Path, prefix: str = "data") -> list[Path]:
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        copied: list[Path] = []
        for path in sorted(source.rglob("*")):
            if path.is_dir():
                if path.is_symlink():
                    raise ConnectorError(f"symlink directory is not allowed: {path}")
                continue
            if not path.is_file():
                raise ConnectorError(f"special source path is not allowed: {path}")
            relative = path.relative_to(source).as_posix()
            role = _normalise_role(None, relative)
            copied.append(self.add(path, f"{prefix}/{relative}", role=role))
        if not copied:
            raise ConnectorError(f"source directory is empty: {source}")
        return copied

    def add_role_resource(self, role: str | None, source: Any, *, relative: str | None = None, base: Path | None = None, expected_sha256: str | None = None, expected_bytes: int | None = None, index: int = 0) -> Path:
        source_value = _local_or_url(source, base=base)
        if relative is None:
            relative = f"data/{_default_name(role or _normalise_role(None, str(source)) or 'resource', source, index)}"
        if not relative.startswith("data/") and not relative.startswith("beir/") and not relative.startswith("source/") and not relative.startswith("github/"):
            relative = f"data/{relative}"
        return self.add(source_value, relative, role=role or _normalise_role(None, str(relative)), expected_sha256=expected_sha256, expected_bytes=expected_bytes)


def _resource_fields(role: str | None, value: Any, *, index: int = 0) -> tuple[str | None, Any, str | None, str | None, int | None]:
    if isinstance(value, Mapping):
        role = str(value.get("role", role)) if value.get("role", role) is not None else None
        source = value.get("url", value.get("path", value.get("file", value.get("source"))))
        if source is None:
            raise ValueError(f"resource {index} has no url/path/file/source")
        relative = value.get("name", value.get("destination", value.get("relative")))
        expected_sha256 = value.get("sha256", value.get("hash"))
        expected_bytes = value.get("bytes", value.get("size"))
        return role, source, str(relative) if relative is not None else None, expected_sha256, int(expected_bytes) if expected_bytes is not None else None
    return role, value, None, None, None


def _manifest_payload(state: _FetchState, value: Any) -> tuple[dict[str, Any], Path | None, str | None]:
    """Read a source manifest and preserve it as a raw snapshot file."""

    if _is_url(value):
        destination = state.add(str(value), "source/input_manifest.json")
        payload = _read_json(destination)
        return payload if isinstance(payload, Mapping) else {}, None, str(value)
    source = _as_path(value)
    if not source.exists():
        raise FileNotFoundError(source)
    destination = state.add(source, "source/input_manifest.json")
    payload = _read_json(destination)
    return payload if isinstance(payload, Mapping) else {}, source.parent, None


def _collect_option_resources(state: _FetchState, *, allow_empty: bool = False) -> None:
    options = state.options
    manifest_value = options.get(
        "manifest",
        options.get("manifest_path", options.get("manifest_url")),
    )
    manifest: Mapping[str, Any] = {}
    base: Path | None = None
    manifest_url: str | None = None
    if manifest_value is not None:
        manifest, base, manifest_url = _manifest_payload(state, manifest_value)
    resources: list[tuple[str | None, Any, str | None, str | None, int | None]] = []

    for role in _ROLE_NAMES:
        value = options.get(role, manifest.get(role))
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for index, item in enumerate(values):
            resources.append(_resource_fields(role, item, index=index))

    files = options.get("files", manifest.get("files"))
    if isinstance(files, Mapping):
        for role, value in files.items():
            values = value if isinstance(value, list) else [value]
            for index, item in enumerate(values):
                resources.append(_resource_fields(_normalise_role(role), item, index=index))
    elif isinstance(files, list):
        for index, item in enumerate(files):
            resources.append(_resource_fields(None, item, index=index))

    urls = options.get("urls", manifest.get("urls"))
    if urls is not None:
        values = urls if isinstance(urls, list) else [urls]
        for index, item in enumerate(values):
            resources.append(_resource_fields(None, item, index=index))

    direct = options.get(
        "path",
        options.get("source_path", options.get("source", options.get("url"))),
    )
    if direct is not None and not isinstance(direct, Mapping):
        direct_path = _local_or_url(direct, base=base)
        if isinstance(direct_path, Path) and direct_path.is_dir():
            state.add_tree(direct_path)
        else:
            resources.append(_resource_fields(None, direct_path))

    if not resources and not state.layout:
        if allow_empty:
            return
        raise ConnectorError(
            "source has no materialisable files; provide options.path, options.files, "
            "options.urls, or a manifest"
        )
    for index, (role, source, relative, expected_sha256, expected_bytes) in enumerate(resources):
        source = _local_or_url(source, base=base) if not isinstance(source, Path) else source
        if isinstance(source, Path) and source.is_dir():
            state.add_tree(source)
            continue
        if relative is None:
            name = Path(str(source)).name or f"resource-{index}"
            role_name = role or _normalise_role(None, name)
            relative = f"data/{_default_name(role_name or 'resource', source, index)}"
            if role_name is None and name in {"", "."}:
                relative = f"data/resource-{index}"
        if manifest_url and not _is_url(source):
            source = urllib.parse.urljoin(manifest_url, str(source))
        state.add(source, relative, role=role or _normalise_role(None, relative), expected_sha256=expected_sha256, expected_bytes=expected_bytes)


def _source_revision(spec: Mapping[str, Any], *, pinned: bool = False) -> str:
    revision = str(spec.get("revision") or "")
    if pinned and revision.lower() in _FLOATING_REVISIONS:
        raise ConnectorError(
            f"{spec['connector']} requires a fixed revision; got {revision!r}"
        )
    return revision or "local"


def _fetch_generic(spec: dict[str, Any], state: _FetchState) -> str:
    _collect_option_resources(state)
    return _source_revision(spec)


def _fetch_beir(spec: dict[str, Any], state: _FetchState) -> str:
    options = state.options
    archive = options.get("archive", options.get("zip"))
    if archive is None and options.get("path") and str(options["path"]).lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        archive = options["path"]
    if archive is None and options.get("url") and str(options["url"]).lower().split("?", 1)[0].endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        archive = options["url"]
    if archive is not None:
        archive_source = _local_or_url(archive)
        archive_path = state.add(archive_source, "beir/archive" + (".zip" if str(archive).lower().endswith(".zip") else ".tar"), expected_sha256=options.get("sha256", options.get("archive_sha256")), expected_bytes=options.get("bytes", options.get("archive_bytes")))
        extracted = state.root / "beir" / "extracted"
        safe_extract_archive(
            archive_path,
            extracted,
            max_member_bytes=int(options.get("max_member_bytes", DEFAULT_MAX_MEMBER_BYTES)),
            max_total_bytes=int(options.get("max_archive_bytes", DEFAULT_MAX_ARCHIVE_BYTES)),
            max_files=int(options.get("max_archive_files", DEFAULT_MAX_ARCHIVE_FILES)),
            max_compression_ratio=float(options.get("max_compression_ratio", DEFAULT_MAX_COMPRESSION_RATIO)),
            overwrite=True,
        )
        for path in _find_all_files(extracted):
            relative = path.relative_to(state.root).as_posix()
            state.layout.setdefault(_normalise_role(None, relative) or "files", []).append(relative)
    else:
        _collect_option_resources(state)
    split = str(options.get("split", "test"))
    state.layout["split"] = [split]
    return _source_revision(spec)


def _hf_url(options: Mapping[str, Any], revision: str, filename: str) -> str:
    repo = options.get("dataset", options.get("repo", options.get("repo_id")))
    if not repo:
        raise ConnectorError("HF connector requires options.dataset/repo when files are URLs")
    endpoint = str(options.get("endpoint", "https://huggingface.co")).rstrip("/")
    repo_type = str(options.get("repo_type", "datasets"))
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in str(filename).replace("\\", "/").split("/"))
    return f"{endpoint}/{repo_type}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{encoded}"


def _fetch_hf(spec: dict[str, Any], state: _FetchState, *, expected_files: int | None = None) -> str:
    revision = _source_revision(spec, pinned=True)
    options = state.options
    files = options.get("files")
    # A HF ``files`` list normally contains repository-relative names.  The
    # generic materialiser quite correctly treats a relative local path as a
    # local import, so handle this form here and turn missing relative names
    # into URLs pinned to the requested revision.
    if (
        files is not None
        and not options.get("path")
        and not options.get("source_path")
        and not options.get("manifest")
        and not options.get("manifest_path")
        and not options.get("manifest_url")
        and not options.get("urls")
    ):
        values = files.items() if isinstance(files, Mapping) else enumerate(files)
        for key, item in values:
            role, source, relative, expected_sha256, expected_bytes = _resource_fields(
                _normalise_role(key), item, index=int(key) if isinstance(key, int) else 0
            )
            source_path = _as_path(source) if not _is_url(source) else None
            if source_path is not None and source_path.exists():
                source_value: str | Path = source_path
            elif _is_url(source):
                source_value = str(source)
            else:
                source_value = _hf_url(options, revision, str(source))
            if relative is None:
                relative = f"data/{Path(str(source)).name}"
            state.add(source_value, relative, role=role or _normalise_role(None, relative), expected_sha256=expected_sha256, expected_bytes=expected_bytes)
    else:
        _collect_option_resources(state, allow_empty=True)
    count = sum(len(paths) for role, paths in state.layout.items() if role in _ROLE_NAMES)
    # A local HF snapshot may use native names such as wix_kb_corpus.jsonl and
    # test.jsonl rather than canonical role names.  Count the actual JSONL
    # payloads for the fixed-file sanity check as well.
    if expected_files is not None:
        count = max(count, sum(1 for path in _find_all_files(state.root) if path.suffix.lower() in {".jsonl", ".json"}))
    if expected_files is not None and count < expected_files:
        raise ConnectorError(
            f"{spec['connector']} expected at least {expected_files} input files, found {count}"
        )
    resolved = str(options.get("resolved_revision", revision))
    if resolved.lower() in _FLOATING_REVISIONS or not resolved:
        raise ConnectorError("HF resolved_revision must be a fixed commit/revision")
    return resolved


def _fetch_wixqa(spec: dict[str, Any], state: _FetchState) -> str:
    # The released snapshot has one corpus JSONL plus expertwritten and
    # simulated test JSONL files.  A catalog may intentionally pin one split,
    # in which case corpus + that split is the complete source.
    expected = 2 if state.options.get("split") else 3
    return _fetch_hf(spec, state, expected_files=expected)


def _fetch_worksurface(spec: dict[str, Any], state: _FetchState) -> str:
    # WorkSurface has had more than one released file layout.  The pinned
    # revision is the invariant; parsing accepts either the existing adapter's
    # JSONL shape or an equivalent local file set.
    source = state.options.get("path")
    if source is not None and not _is_url(source):
        root = _as_path(source).resolve()
        if not root.is_dir():
            raise ConnectorError(f"worksurface options.path is not a directory: {root}")
        tasks = root / "data" / "tasks.jsonl"
        documents = sorted((root / "resources" / "profiles").glob("*/kb_docs/*.md"))
        if not tasks.is_file() or not documents:
            raise ConnectorError("worksurface local source needs data/tasks.jsonl and profile kb_docs Markdown")
        # Import only canonical dataset payloads. Hidden downloader metadata and
        # interrupted partial trees are provenance aids, not source records.
        state.add(tasks, "data/tasks.jsonl", role="queries")
        for path in documents:
            if path.is_symlink():
                raise ConnectorError(f"worksurface document path is a symlink: {path}")
            relative = path.relative_to(root).as_posix()
            state.add(path, f"data/{relative}", role="documents")
        return _source_revision(spec, pinned=True)
    return _fetch_hf(spec, state)


def _fetch_qasper(spec: dict[str, Any], state: _FetchState) -> str:
    options = state.options
    archive = options.get("archive", options.get("tar", options.get("tgz")))
    if archive is None and options.get("path") and str(options["path"]).lower().split("?", 1)[0].endswith((".tgz", ".tar.gz", ".tar")):
        archive = options["path"]
    if archive is None and options.get("url") and str(options["url"]).lower().split("?", 1)[0].endswith((".tgz", ".tar.gz", ".tar")):
        archive = options["url"]
    if archive is not None:
        archive_path = state.add(
            _local_or_url(archive),
            "qasper/archive.tgz" if str(archive).lower().endswith((".tgz", ".tar.gz")) else "qasper/archive.tar",
            expected_sha256=options.get("sha256", options.get("archive_sha256")),
            expected_bytes=options.get("bytes", options.get("archive_bytes")),
        )
        extracted = state.root / "qasper" / "extracted"
        safe_extract_archive(
            archive_path,
            extracted,
            max_member_bytes=int(options.get("max_member_bytes", DEFAULT_MAX_MEMBER_BYTES)),
            max_total_bytes=int(options.get("max_archive_bytes", DEFAULT_MAX_ARCHIVE_BYTES)),
            max_files=int(options.get("max_archive_files", DEFAULT_MAX_ARCHIVE_FILES)),
            max_compression_ratio=float(options.get("max_compression_ratio", DEFAULT_MAX_COMPRESSION_RATIO)),
            overwrite=True,
        )
        split = str(options.get("split", "dev")).lower()
        requested_file = options.get("file", options.get("data_file"))
        candidates = [
            path
            for path in _find_all_files(extracted)
            if path.name.endswith((".json", ".json.gz"))
            and (requested_file is None or path.name == str(requested_file))
            and (requested_file is not None or split in path.name.lower())
        ]
        if not candidates:
            raise ConnectorError(
                f"QASPER archive contains no {requested_file or split!r} JSON source"
            )
        state.layout["qasper"] = [path.relative_to(state.root).as_posix() for path in sorted(candidates)]
    else:
        _collect_option_resources(state)
    return _source_revision(spec, pinned=False)


def _github_tree(state: _FetchState, repo: str, commit: str, options: Mapping[str, Any]) -> list[dict[str, Any]]:
    tree_url = options.get("tree_url") or f"https://api.github.com/repos/{repo}/git/trees/{urllib.parse.quote(commit, safe='')}?recursive=1"
    destination = state.add(str(tree_url), "github/tree.json")
    payload = _read_json(destination)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tree"), list):
        raise ConnectorError("GitHub tree API response does not contain a tree")
    return [dict(row) for row in payload["tree"] if isinstance(row, Mapping)]


def _path_allowed(path: str, allowlist: Sequence[str]) -> bool:
    clean = path.strip("/")
    for item in allowlist:
        prefix = str(item).replace("\\", "/").strip("/")
        if clean == prefix or clean.startswith(prefix + "/"):
            return True
    return False


def _fetch_github(spec: dict[str, Any], state: _FetchState) -> str:
    revision = _source_revision(spec, pinned=True)
    options = state.options
    repo = str(options.get("repo", ""))
    if not repo and not options.get("files") and not options.get("path"):
        raise ConnectorError("github_markdown requires options.repo or explicit options.files")
    allowlist = options.get("paths", options.get("allowlist", []))
    if isinstance(allowlist, str):
        allowlist = [allowlist]
    allowlist = [str(item) for item in allowlist]
    files = options.get("files")
    if files:
        entries = files.items() if isinstance(files, Mapping) else enumerate(files)
        for key, item in entries:
            role, source, relative, expected_sha256, expected_bytes = _resource_fields(None, item, index=int(key) if isinstance(key, int) else 0)
            path_name = str(relative or (item.get("path") if isinstance(item, Mapping) else source))
            if allowlist and not _path_allowed(path_name, allowlist):
                raise ConnectorError(f"GitHub path is outside allowlist: {path_name!r}")
            if not _is_url(source) and not isinstance(source, Path) and repo and not Path(str(source)).exists():
                source = f"https://raw.githubusercontent.com/{repo}/{revision}/{path_name}"
            state.add(source if isinstance(source, Path) else str(source), f"github/files/{path_name}", role="documents", expected_sha256=expected_sha256, expected_bytes=expected_bytes)
    elif options.get("path"):
        local_root = _as_path(options["path"]).resolve()
        if not local_root.is_dir():
            raise ConnectorError(f"github_markdown options.path is not a directory: {local_root}")
        copied = 0
        for path in sorted(local_root.rglob("*")):
            if path.is_dir():
                if path.is_symlink():
                    raise ConnectorError(f"symlink directory is not allowed: {path}")
                continue
            if not path.is_file():
                raise ConnectorError(f"special source path is not allowed: {path}")
            path_name = path.relative_to(local_root).as_posix()
            if allowlist and not _path_allowed(path_name, allowlist):
                continue
            extensions = options.get("extensions")
            if extensions and path.suffix.lower() not in {str(item).lower() for item in extensions}:
                continue
            state.add(path, f"github/files/{path_name}", role="documents")
            copied += 1
        if not copied:
            raise ConnectorError("GitHub local path contained no files in the requested allowlist")
    else:
        if not allowlist:
            raise ConnectorError("github_markdown requires explicit paths/allowlist")
        tree = _github_tree(state, repo, revision, options)
        found = 0
        for entry in tree:
            path_name = str(entry.get("path", ""))
            if entry.get("type") != "blob" or not _path_allowed(path_name, allowlist):
                continue
            suffix = Path(path_name).suffix.lower()
            extensions = options.get("extensions")
            if extensions and suffix not in {str(item).lower() for item in extensions}:
                continue
            raw_base = str(options.get("raw_base_url", "https://raw.githubusercontent.com")).rstrip("/")
            url = f"{raw_base}/{repo}/{revision}/{urllib.parse.quote(path_name, safe='/')}"
            state.add(url, f"github/files/{path_name}", role="documents")
            found += 1
        if not found:
            raise ConnectorError("GitHub tree contained no files in the requested allowlist")
    return revision


def _fetch_pmc(spec: dict[str, Any], state: _FetchState) -> str:
    options = state.options
    if not options.get("urls") and not options.get("files") and not options.get("pmcids") and not options.get("pmcid") and not options.get("path"):
        raise ConnectorError("pmc_jats requires explicit options.urls/files/pmcids")
    if options.get("pmcids") or options.get("pmcid"):
        pmcids = options.get("pmcids", options.get("pmcid"))
        pmcids = pmcids if isinstance(pmcids, list) else [pmcids]
        template = options.get("url_template", "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/?report=xml")
        for index, pmcid in enumerate(pmcids):
            state.add(str(template).format(pmcid=urllib.parse.quote(str(pmcid), safe="")), f"pmc/{index}-{_safe_component(pmcid)}.xml", role="documents")
    else:
        _collect_option_resources(state)
    return _source_revision(spec)


def _fetch_explicit_html_xml(spec: dict[str, Any], state: _FetchState) -> str:
    if not state.options.get("urls") and not state.options.get("files") and not state.options.get("path"):
        raise ConnectorError(f"{spec['connector']} requires explicit options.urls/files/path")
    _collect_option_resources(state)
    return _source_revision(spec)


def _manifest_files(root: Path, manifest: Mapping[str, Any], role: str) -> list[Path]:
    paths = manifest.get("layout", {}).get(role) if isinstance(manifest.get("layout"), Mapping) else None
    if isinstance(paths, str):
        paths = [paths]
    if isinstance(paths, list):
        output = []
        for relative in paths:
            path = root / safe_relative_path(str(relative))
            if path.is_file():
                output.append(path)
        if output:
            return output
    aliases = {
        "documents": ("documents.jsonl", "corpus.jsonl", "docs.jsonl", "documents.json"),
        "queries": ("queries.jsonl", "questions.jsonl", "queries.json"),
        "qrels": ("qrels.json", "qrels.tsv", "labels.json"),
        "evidence": ("evidence.jsonl", "evidence.json", "annotations.jsonl"),
        "judgments": ("judgments.jsonl", "judgments.json"),
    }
    candidates = [path for path in _find_all_files(root) if path.name.lower() in aliases.get(role, ())]
    if role == "qrels":
        candidates.extend(path for path in _find_all_files(root) if path.suffix.lower() == ".tsv" and "qrel" in path.name.lower())
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique
    if len(unique) > 1:
        # A BEIR snapshot may contain train/dev/test qrels.  The selected split
        # is handled by the dedicated loader; generic ambiguity is an error.
        raise ConnectorError(f"multiple candidate {role} files in snapshot: {unique}")
    return []


def _verify_snapshot(spec: Mapping[str, Any], snapshot: Path) -> dict[str, Any]:
    snapshot = Path(snapshot)
    if snapshot.name.endswith(".partial"):
        raise SnapshotConflictError(f"partial source snapshot is not loadable: {snapshot}")
    manifest_path = snapshot / "source_manifest.json"
    if not manifest_path.is_file():
        raise SnapshotConflictError(f"source snapshot has no complete manifest: {snapshot}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("complete") is not True:
        raise SnapshotConflictError(f"source snapshot is not complete: {snapshot}")
    expected = _spec_fingerprint(_require_spec(spec))
    recorded = manifest.get("requested_spec_fingerprint", manifest.get("spec_fingerprint"))
    if recorded != expected:
        raise SnapshotConflictError("source spec fingerprint does not match snapshot manifest")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SnapshotConflictError("source manifest files must be a list")
    verify_file_inventory(snapshot, files)
    listed = {safe_relative_path(str(entry["path"])) for entry in files if isinstance(entry, Mapping)}
    actual = {path.relative_to(snapshot).as_posix() for path in _find_all_files(snapshot) if path.name != "source_manifest.json"}
    if actual != listed:
        raise SnapshotConflictError(
            f"source snapshot file inventory differs from manifest: extra={sorted(actual - listed)[:3]}, "
            f"missing={sorted(listed - actual)[:3]}"
        )
    return dict(manifest)


def _write_snapshot_manifest(state: _FetchState, resolved_revision: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in _find_all_files(state.root):
        relative = path.relative_to(state.root).as_posix()
        if relative == "source_manifest.json" or relative.startswith("."):
            continue
        digest, size = file_digest(path, max_bytes=DEFAULT_MAX_BYTES)
        files.append({"path": relative, "sha256": digest, "bytes": size})
    files.sort(key=lambda item: str(item["path"]))
    safe_spec = redact_secrets(_jsonable(state.spec))
    urls = list(dict.fromkeys(state.actual_urls))
    manifest = {
        "schema_version": "dci-source-snapshot-v1",
        "complete": True,
        "id": state.spec["id"],
        "connector": state.spec["connector"],
        "input_kind": state.spec["input_kind"],
        "requested_spec_fingerprint": _spec_fingerprint(state.spec),
        "spec_fingerprint": _spec_fingerprint(state.spec),
        "requested_spec": safe_spec,
        "actual_url": urls[0] if len(urls) == 1 else None,
        "actual_urls": urls,
        "resolved_revision": resolved_revision,
        "layout": {key: sorted(values) for key, values in sorted(state.layout.items())},
        "files": files,
    }
    temporary = state.root / ".source_manifest.json.partial"
    temporary.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    os.replace(temporary, state.root / "source_manifest.json")
    return manifest


def _fetch_dispatch(spec: dict[str, Any], state: _FetchState) -> str:
    fetcher = _CONNECTOR_REGISTRY[spec["connector"]].fetch
    return fetcher(spec, state)


def fetch_source(spec: Mapping[str, Any], raw_root: Path) -> Path:
    """Fetch or reuse an immutable source snapshot.

    The final directory is keyed by the complete requested-spec fingerprint.
    A failed operation remains under the same path with a ``.partial`` suffix;
    it can never be mistaken for a complete snapshot and is reused only by a
    subsequent attempt with the exact same spec.
    """

    normalized = _require_spec(spec)
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _spec_fingerprint(normalized)
    final = raw_root / _safe_component(normalized["id"]) / fingerprint
    lock_path = final.with_name(final.name + ".lock")
    with nonblocking_file_lock(lock_path):
        if final.exists():
            _verify_snapshot(normalized, final)
            return final
        partial = final.with_name(final.name + ".partial")
        if partial.exists() and not partial.is_dir():
            raise SnapshotConflictError(f"partial snapshot path is not a directory: {partial}")
        partial.mkdir(parents=True, exist_ok=True)
        state = _FetchState(normalized, partial)
        try:
            resolved_revision = _fetch_dispatch(normalized, state)
            _write_snapshot_manifest(state, resolved_revision)
            if final.exists():
                raise SnapshotConflictError(f"snapshot appeared concurrently: {final}")
            os.replace(partial, final)
        except Exception:
            # The partial directory is deliberately retained.  It has no complete
            # manifest and therefore cannot be loaded as a source.
            raise
    _verify_snapshot(normalized, final)
    return final


def _source_record(spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    options = spec.get("options") if isinstance(spec.get("options"), Mapping) else {}
    dataset = options.get("dataset", options.get("repo", spec.get("id")))
    return {
        "dataset": dataset,
        "revision": spec.get("revision") or manifest.get("resolved_revision"),
        "connector": spec.get("connector"),
    }


def _provenance(spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "domain": spec.get("domain", ""),
        "language": spec.get("language", ""),
        "connector": spec.get("connector"),
        "source_snapshot_fingerprint": manifest.get("requested_spec_fingerprint"),
    }
    options = spec.get("options") if isinstance(spec.get("options"), Mapping) else {}
    for key in ("genre", "task", "split"):
        if key in spec:
            output[key] = _jsonable(spec[key])
        elif key in options:
            output[key] = _jsonable(options[key])
    if spec.get("license") is not None:
        output["license"] = spec.get("license")
    return redact_secrets(output)


def _base_bundle(spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_kind": spec["input_kind"],
        "source": _source_record(spec, manifest),
        "documents": [],
        "queries": [],
        "qrels": {},
        "evidence": [],
        "judgments": [],
        "provenance": _provenance(spec, manifest),
    }


def _generic_paths(snapshot: Path, manifest: Mapping[str, Any], *, split: str | None = None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for role in _ROLE_NAMES:
        found = _manifest_files(snapshot, manifest, role)
        if role == "qrels" and split and len(found) != 1:
            target = split.lower()
            split_matches = [
                path
                for path in found
                if path.stem.lower() == target
                or any(part.lower() == target for part in path.relative_to(snapshot).parts[:-1])
            ]
            if split_matches:
                found = split_matches
        if len(found) == 1:
            result[role] = found[0]
        elif len(found) > 1:
            raise ConnectorError(f"multiple {role} resources found in {snapshot}: {found}")
    return result


def _load_generic(spec: Mapping[str, Any], snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _base_bundle(spec, manifest)
    options = spec.get("options") if isinstance(spec.get("options"), Mapping) else {}
    paths = _generic_paths(snapshot, manifest, split=str(options.get("split", "")) or None)
    if "documents" not in paths:
        raise ConnectorError("generic source has no documents/corpus JSONL")
    bundle["documents"] = _canonical_documents(_iter_jsonl(paths["documents"]), paths["documents"], spec.get("language"))
    if spec["input_kind"] in {"retrieval", "evidence_qa"}:
        if "queries" not in paths:
            raise ConnectorError(f"{spec['input_kind']} source has no queries/questions JSONL")
        bundle["queries"] = _canonical_queries(_iter_jsonl(paths["queries"]), paths["queries"], spec.get("language"))
    if spec["input_kind"] != "corpus":
        if "qrels" in paths:
            bundle["qrels"] = _parse_qrels(paths["qrels"])
        if "evidence" in paths:
            bundle["evidence"] = _canonical_evidence(paths["evidence"])
        if "judgments" in paths:
            bundle["judgments"] = _iter_jsonl(paths["judgments"])
        if spec["input_kind"] == "evidence_qa" and not bundle["qrels"]:
            for item in bundle["evidence"]:
                bundle["qrels"].setdefault(item["query_id"], {})[item["doc_id"]] = 1.0
        if spec["input_kind"] == "retrieval" and "qrels" not in paths:
            raise ConnectorError("retrieval source has no qrels")
    return bundle


def _load_beir(spec: Mapping[str, Any], snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    # The archive is extracted during fetch, so generic discovery handles both
    # root-level and nested BEIR layouts. Queries are published for all BEIR
    # splits in one file; retain only IDs with qrels for the requested split.
    bundle = _load_generic(spec, snapshot, manifest)
    options = spec.get("options") if isinstance(spec.get("options"), Mapping) else {}
    split = str(options.get("split", "")).strip()
    if split and bundle["qrels"]:
        original_count = len(bundle["queries"])
        labeled_ids = set(bundle["qrels"])
        bundle["queries"] = [query for query in bundle["queries"] if query["_id"] in labeled_ids]
        bundle["provenance"].update({
            "split": split,
            "queries_original_count": original_count,
            "queries_target_split_count": len(bundle["queries"]),
            "queries_without_target_split_qrels": original_count - len(bundle["queries"]),
            "query_filter": "retain only query IDs present in the selected split qrels",
        })
    return bundle


def _qasper_files(snapshot: Path, manifest: Mapping[str, Any] | None = None) -> list[Path]:
    if isinstance(manifest, Mapping) and isinstance(manifest.get("layout"), Mapping):
        selected = manifest["layout"].get("qasper")
        if isinstance(selected, str):
            selected = [selected]
        if isinstance(selected, list):
            paths = [snapshot / safe_relative_path(str(relative)) for relative in selected]
            paths = [path for path in paths if path.is_file()]
            if paths:
                return sorted(paths)
    paths = [
        path
        for path in _find_all_files(snapshot)
        if path.suffix.lower() in {".json", ".gz"}
        and path.name != "source_manifest.json"
        and "source" not in path.relative_to(snapshot).parts
    ]
    return [path for path in paths if path.name.endswith((".json", ".json.gz"))]


def _load_wixqa(spec: Mapping[str, Any], snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read the released WixQA JSONL layout without using answer text as labels."""

    bundle = _base_bundle(spec, manifest)
    options = spec.get("options") if isinstance(spec.get("options"), Mapping) else {}
    corpus_candidates = [
        path
        for path in _find_all_files(snapshot)
        if path.name == "wix_kb_corpus.jsonl" or "corpus" in path.name.lower() and path.suffix == ".jsonl"
    ]
    if not corpus_candidates:
        # Canonical local imports can still use the generic names.
        return _load_generic(spec, snapshot, manifest)
    corpus_path = sorted(corpus_candidates)[0]
    corpus_rows = _iter_jsonl(corpus_path)
    documents: list[dict[str, Any]] = []
    known: set[str] = set()
    for row in corpus_rows:
        document_id = _id_from_row(row, kind="document", index=len(documents) + 1, path=corpus_path)
        if document_id in known:
            raise ValueError(f"duplicate Wix corpus ID: {document_id!r}")
        known.add(document_id)
        documents.append(
            {
                "_id": document_id,
                "title": _text(row.get("title", "")),
                "text": _text(row.get("contents", row.get("text", row.get("content", "")))),
                "language": str(row.get("language") or spec.get("language") or ""),
                "metadata": _metadata(row, {"id", "_id", "title", "contents", "text", "content", "language", "metadata"}),
            }
        )
    split = options.get("split")
    query_files = sorted(
        path
        for path in _find_all_files(snapshot)
        if path.name == "test.jsonl" and "wixqa_" in "/".join(path.relative_to(snapshot).parts)
    )
    if split:
        target_component = f"wixqa_{str(split).lower()}"
        query_files = [
            path
            for path in query_files
            if target_component in {part.lower() for part in path.relative_to(snapshot).parts}
        ]
    if not query_files:
        return _load_generic(spec, snapshot, manifest)
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, float]] = {}
    for path in query_files:
        split_name = next((part.removeprefix("wixqa_") for part in path.relative_to(snapshot).parts if part.startswith("wixqa_")), path.parent.name)
        for index, row in enumerate(_iter_jsonl(path)):
            if spec["input_kind"] == "corpus":
                break
            query_id = str(row.get("id", row.get("query_id", f"{split_name}/{index}")))
            if any(item["_id"] == query_id for item in queries):
                raise ValueError(f"duplicate WixQA query ID: {query_id!r}")
            queries.append({
                "_id": query_id,
                "text": _text(row.get("question", row.get("text", ""))),
                "language": str(row.get("language") or spec.get("language") or ""),
                "metadata": {"split": split_name, "source_file": path.relative_to(snapshot).as_posix()},
            })
            article_ids = row.get("article_ids", row.get("relevant_ids", row.get("documents", [])))
            if isinstance(article_ids, str):
                article_ids = [article_ids]
            if not isinstance(article_ids, list):
                raise ValueError(f"WixQA article_ids for {query_id!r} is not a list")
            for document_id in article_ids:
                qrels.setdefault(query_id, {})[str(document_id)] = 1.0
    bundle["documents"] = documents
    if spec["input_kind"] == "corpus":
        return bundle
    bundle["queries"] = queries
    bundle["qrels"] = qrels
    return bundle


def _worksurface_equivalent_documents(paths: Sequence[Path]) -> list[list[str]]:
    """Find same-name source documents duplicated under different task IDs."""

    groups: dict[tuple[str, str], list[str]] = {}
    pattern = re.compile(r"\A<!-- source_task: (\d+) \| source_file: (.+?) \| surface: rag -->\n\n")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        header = pattern.match(text)
        if not header or path.name != f"t{header[1]}__{header[2]}":
            continue
        digest = hashlib.sha256(text[header.end():].encode("utf-8")).hexdigest()
        groups.setdefault((header[2], digest), []).append(path.name)
    return sorted(sorted(ids) for ids in groups.values() if len(ids) > 1)


def _worksurface_span_locator(question: str, text: str, span: str) -> dict[str, Any]:
    """Locate a native span, using exact cloze context for repeated strings."""
    if not span:
        return {}
    first = text.find(span)
    if first >= 0 and text.find(span, first + 1) < 0:
        return {"start": first, "end": first + len(span), "locator_method": "unique_source_span"}
    statements = re.findall("“([^”]*)”", question)
    candidates = set()
    for statement in statements:
        if statement.count("___") != 1:
            continue
        hole = statement.index("___")
        target = statement[:hole] + span + statement[hole + 3:]
        offset = text.find(target)
        if offset >= 0 and text.find(target, offset + 1) < 0:
            candidates.add(offset + hole)
    if len(candidates) == 1:
        start = candidates.pop()
        return {"start": start, "end": start + len(span), "locator_method": "unique_exact_cloze_context"}
    return {}


def _load_worksurface(spec: Mapping[str, Any], snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Equivalent read-only adapter for the pinned WorkSurface JSONL layout."""

    bundle = _base_bundle(spec, manifest)
    options = spec.get("options") if isinstance(spec.get("options"), Mapping) else {}
    # A local HF export may contain interrupted-download metadata under
    # '.hfd/partials-*'. Only canonical data/tasks.jsonl and profile KB
    # documents are source inputs; hidden metadata must never be parsed.
    task_files = [
        path
        for path in _find_all_files(snapshot)
        if path.relative_to(snapshot).parts[-2:] == ("data", "tasks.jsonl")
        and not any(part.startswith(".") for part in path.relative_to(snapshot).parts)
    ]
    doc_paths_all = [
        path
        for path in _find_all_files(snapshot)
        if path.suffix.lower() == ".md"
        and "resources" in path.relative_to(snapshot).parts
        and "profiles" in path.relative_to(snapshot).parts
        and path.parent.name == "kb_docs"
        and not any(part.startswith(".") for part in path.relative_to(snapshot).parts)
    ]
    if not task_files or not doc_paths_all:
        return _load_generic(spec, snapshot, manifest)
    persona = str(options.get("persona", options.get("profile", "")))
    slug_map = {
        "Backend Developer": "backend_developer",
        "Logistics Manager": "logistics_manager",
        "Operations Manager": "operations_manager",
        "Product Manager": "product_manager",
        "Researcher": "researcher",
    }
    if persona in slug_map:
        persona_slug = slug_map[persona]
    else:
        persona_slug = persona.lower().replace(" ", "_") if persona else ""
    if persona_slug:
        doc_paths = sorted(path for path in doc_paths_all if persona_slug in path.parts)
    else:
        doc_paths = sorted(doc_paths_all)
    if not doc_paths:
        raise ConnectorError("WorkSurface source has no persona kb_docs Markdown files")
    known = {path.name for path in doc_paths}
    duplicate_groups = _worksurface_equivalent_documents(doc_paths)
    equivalents = {document: group for group in duplicate_groups for document in group}
    excluded: list[dict[str, Any]] = []
    bundle["documents"] = [
        {
            "_id": path.name,
            "title": "",
            "text": path.read_text(encoding="utf-8"),
            "language": str(spec.get("language") or ""),
            "metadata": {"source_file": path.relative_to(snapshot).as_posix()},
        }
        for path in doc_paths
    ]
    texts = {document["_id"]: document["text"] for document in bundle["documents"]}
    if spec["input_kind"] == "corpus":
        bundle["provenance"].update({
            "equivalent_document_groups": duplicate_groups,
            "equivalence_policy": "retain all source document IDs; exclusions apply only to labeled QA queries",
            "excluded": excluded,
        })
        return bundle
    seen_queries: set[str] = set()
    for task_path in task_files:
        for row in _iter_jsonl(task_path):
            source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
            row_persona = str(source.get("persona", ""))
            if persona and row_persona not in {persona, persona_slug.replace("_", " ").title()}:
                continue
            if row.get("task_type") != "rag_only" or row.get("required_surfaces") != ["rag"]:
                excluded.append({"query_id": str(row.get("id", "")), "reason": "non_pure_rag_task"})
                continue
            query_id = str(row.get("id", ""))
            question = _text(row.get("question", row.get("text", "")))
            if not query_id:
                excluded.append({"query_id": query_id, "reason": "empty_query_id"})
                continue
            if query_id in seen_queries:
                raise ValueError(f"duplicate WorkSurface query ID: {query_id!r}")
            seen_queries.add(query_id)
            if not question.strip():
                excluded.append({"query_id": query_id, "reason": "empty_or_invalid_query"})
                continue
            bundle["queries"].append({"_id": query_id, "text": question, "language": str(spec.get("language") or ""), "metadata": {"persona": row_persona}})
            evidence_rows = row.get("gold_evidence", [])
            if not isinstance(evidence_rows, list) or any(
                not isinstance(item, Mapping) or item.get("surface") != "rag" or not item.get("file")
                for item in evidence_rows
            ):
                excluded.append({"query_id": query_id, "reason": "invalid_rag_evidence"})
                bundle["queries"].pop()
                continue
            document_ids = [str(item["file"]) for item in evidence_rows]
            missing = sorted(set(document_ids) - known)
            if missing:
                excluded.append({"query_id": query_id, "reason": "unknown_gold_ids", "document_ids": missing})
                bundle["queries"].pop()
                continue
            unlabelled = sorted({other for document_id in document_ids for other in equivalents.get(document_id, [])} - set(document_ids))
            if unlabelled:
                excluded.append({
                    "query_id": query_id,
                    "reason": "unlabelled_equivalent_documents",
                    "original_gold_ids": sorted(set(document_ids)),
                    "document_ids": unlabelled,
                })
                bundle["queries"].pop()
                continue
            for evidence_row in evidence_rows:
                document_id = str(evidence_row["file"])
                # Keep the native evidence/qrel and let the quality pass report
                # missing IDs; do not silently map or delete a bad annotation.
                bundle["qrels"].setdefault(query_id, {})[document_id] = 1.0
                item = {"query_id": query_id, "doc_id": document_id}
                for key in ("quote", "start", "end", "claim"):
                    if key in evidence_row:
                        item[key] = _jsonable(evidence_row[key])
                if "quote" not in item and isinstance(evidence_row.get("span"), str):
                    item["quote"] = evidence_row["span"]
                    item["source_span"] = evidence_row["span"]
                    if "start" not in item and "end" not in item:
                        item.update(_worksurface_span_locator(question, texts[document_id], item["quote"]))
                bundle["evidence"].append(item)
    bundle["provenance"].update({
        "equivalent_document_groups": duplicate_groups,
        "equivalence_policy": "same source filename and identical body after canonical provenance header; exclude whole queries with unlabelled equivalent files; retain all corpus IDs",
        "candidate_scope": "all persona kb_docs/*.md",
        "query_scope": "rag_only and required_surfaces exactly [rag]",
        "evidence_policy": "map native span to exact quote; locate repeated spans only through unique exact cloze context; unresolved locators require review",
        "excluded": excluded,
    })
    return bundle


def _flatten_evidence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_evidence(item))
        return output
    return [str(value)]


def _qasper_sections(row: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    title = _text(row.get("title", ""))
    sections: list[dict[str, Any]] = []
    abstract = row.get("abstract")
    if abstract:
        paragraphs = _flatten_evidence(abstract)
        sections.append({"title": "Abstract", "paragraphs": paragraphs})
    full_text = row.get("full_text", row.get("fulltext", row.get("sections", [])))
    if isinstance(full_text, Mapping):
        # The released QASPER JSON uses a columnar object in some versions:
        # ``section_name`` is a list while ``paragraphs`` is a list of lists.
        # Other exports use one section object per list element.  Preserve the
        # paragraph strings exactly in either representation so evidence can be
        # checked against the source rather than against a normalised copy.
        names = full_text.get("section_name", full_text.get("title", ""))
        paragraphs_value = full_text.get("paragraphs", full_text.get("text", []))
        if isinstance(names, list):
            if isinstance(paragraphs_value, list) and len(paragraphs_value) == len(names) and any(
                isinstance(item, (list, tuple)) for item in paragraphs_value
            ):
                paragraph_groups = list(paragraphs_value)
            elif len(names) == 1:
                paragraph_groups = [paragraphs_value]
            else:
                paragraph_groups = [[item] for item in _flatten_evidence(paragraphs_value)]
            for index, name in enumerate(names):
                paragraphs = _flatten_evidence(paragraph_groups[index]) if index < len(paragraph_groups) else []
                sections.append({"title": _text(name), "paragraphs": paragraphs})
        else:
            sections.append({"title": _text(names), "paragraphs": _flatten_evidence(paragraphs_value)})
    elif isinstance(full_text, list):
        for section in full_text:
            if isinstance(section, Mapping):
                name = _text(section.get("section_name", section.get("title", "")))
                paragraphs = _flatten_evidence(section.get("paragraphs", section.get("text", [])))
            else:
                name = ""
                paragraphs = _flatten_evidence(section)
            sections.append({"title": name, "paragraphs": paragraphs})
    text_parts = [title]
    for section in sections:
        if section["title"]:
            text_parts.append(str(section["title"]))
        text_parts.extend(section["paragraphs"])
    return "\n\n".join(part for part in text_parts if part), sections


def _qasper_rows(data: Any) -> list[dict[str, Any]]:
    """Return paper rows from list, single-paper, and official keyed JSON."""

    if isinstance(data, list):
        rows = data
        output: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("QASPER paper row is not an object")
            output.append(dict(row))
        return output
    if not isinstance(data, Mapping):
        raise ValueError("QASPER source must contain an object or list of papers")
    if isinstance(data.get("data"), list):
        return _qasper_rows(data["data"])
    if any(key in data for key in ("id", "paper_id", "article_id", "title", "qas", "full_text", "abstract")):
        return [dict(data)]
    output = []
    for paper_id, paper in data.items():
        if not isinstance(paper, Mapping):
            raise ValueError(f"QASPER paper {paper_id!r} is not an object")
        row = dict(paper)
        row.setdefault("_source_paper_id", str(paper_id))
        output.append(row)
    return output


def _qasper_questions(value: Any) -> list[dict[str, Any]]:
    """Expand QASPER's list-of-objects and columnar qas representations."""

    if isinstance(value, list):
        output = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError("QASPER QA entry is not an object")
            output.append(dict(item))
        return output
    if not isinstance(value, Mapping):
        raise ValueError("QASPER qas must be an object or list")
    question = value.get("question", value.get("questions", ""))
    if isinstance(question, list):
        question_ids = value.get("question_id", value.get("question_ids", []))
        answers = value.get("answers", [])
        output = []
        for index, text in enumerate(question):
            row: dict[str, Any] = {"question": text}
            if isinstance(question_ids, list) and index < len(question_ids):
                row["question_id"] = question_ids[index]
            elif not isinstance(question_ids, list) and question_ids:
                row["question_id"] = question_ids
            if isinstance(answers, list) and index < len(answers):
                row["answers"] = answers[index]
            else:
                row["answers"] = []
            for key in ("nlp_background", "topic_background", "paper_read", "search_query", "question_writer"):
                item = value.get(key)
                if isinstance(item, list) and index < len(item):
                    row[key] = item[index]
                elif item is not None and not isinstance(item, list):
                    row[key] = item
            output.append(row)
        return output
    return [dict(value)]


def _qasper_annotations(value: Any) -> list[tuple[dict[str, Any], Mapping[str, Any]]]:
    """Unwrap official ``answers[].answer`` while retaining old fixtures."""

    answers = value if isinstance(value, list) else [value]
    output: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for wrapper in answers:
        if isinstance(wrapper, Mapping):
            nested = wrapper.get("answer")
            if isinstance(nested, list):
                for item in nested:
                    payload = dict(item) if isinstance(item, Mapping) else {"answer": item}
                    output.append((payload, wrapper))
                continue
            if isinstance(nested, Mapping):
                output.append((dict(nested), wrapper))
                continue
            # Older local fixtures put evidence beside an answer string.
            output.append((dict(wrapper), wrapper))
        else:
            output.append(({"answer": wrapper}, {}))
    return output


def _qasper_evidence_payload(payload: Mapping[str, Any], wrapper: Mapping[str, Any]) -> tuple[bool, list[str]]:
    merged = dict(wrapper)
    merged.update(payload)
    value = payload.get("unanswerable", False)
    unanswerable = value is True or (isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"})
    return unanswerable, _flatten_evidence(merged.get("evidence"))


def _load_qasper(spec: Mapping[str, Any], snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _base_bundle(spec, manifest)
    files = _qasper_files(snapshot, manifest)
    if not files:
        raise ConnectorError("qasper snapshot has no .json or .json.gz source file")
    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, float]] = {}
    seen_docs: set[str] = set()
    seen_queries: set[str] = set()
    excluded: list[dict[str, Any]] = []
    for path in files:
        data = _read_json_or_jsonl(path)
        rows = _qasper_rows(data)
        for paper_index, paper in enumerate(rows, 1):
            if not isinstance(paper, Mapping):
                raise ValueError(f"QASPER paper row {paper_index} in {path} is not an object")
            paper_id = str(paper.get("id", paper.get("paper_id", paper.get("article_id", paper.get("_source_paper_id", "")))))
            if not paper_id:
                raise ValueError(f"QASPER paper row {paper_index} in {path} has no paper ID")
            if paper_id in seen_docs:
                raise ValueError(f"duplicate QASPER paper ID: {paper_id!r}")
            seen_docs.add(paper_id)
            text, sections = _qasper_sections(paper)
            # QASPER evidence is normally a paragraph, but the released
            # annotations also contain exact section headings (for example
            # ``Languages``).  Keep the heading as a source span when it is
            # present verbatim in the paper; accepting it does not require
            # normalising, truncating, or otherwise changing the annotation.
            source_paragraphs = {
                str(paragraph)
                for section in sections
                for paragraph in section["paragraphs"]
            }
            synthetic_abstract = bool(paper.get("abstract")) and bool(sections) and (
                sections[0].get("title") == "Abstract"
                and sections[0].get("paragraphs") == _flatten_evidence(paper.get("abstract"))
            )
            source_headings = {
                str(section["title"])
                for index, section in enumerate(sections)
                if str(section.get("title", "")) and not (synthetic_abstract and index == 0)
            }
            source_evidence = source_paragraphs | source_headings
            document = {
                "_id": paper_id,
                "title": _text(paper.get("title", "")),
                "text": text,
                "language": str(spec.get("language") or ""),
                "metadata": {"sections": sections},
            }
            documents.append(document)
            if spec["input_kind"] == "corpus":
                continue
            qas = _qasper_questions(paper.get("qas", paper.get("qa", [])))
            for qa_index, qa in enumerate(qas, 1):
                if not isinstance(qa, Mapping):
                    raise ValueError(f"QASPER QA {paper_id}/{qa_index} is not an object")
                query_id = str(qa.get("id", qa.get("question_id", f"{paper_id}::q{qa_index}")))
                if not query_id:
                    raise ValueError(f"QASPER question {paper_id}/{qa_index} has no ID")
                if query_id in seen_queries:
                    raise ValueError(f"duplicate QASPER question ID: {query_id!r}")
                seen_queries.add(query_id)
                question = _text(qa.get("question", qa.get("text", "")))
                annotations = _qasper_annotations(qa.get("answers", qa.get("answer", [])))
                if not annotations:
                    excluded.append({
                        "query_id": query_id,
                        "reason": "no_answer_annotations",
                        "source_file": path.name,
                        "original_evidence": [],
                    })
                    continue
                valid: list[tuple[Mapping[str, Any], list[str], str | None]] = []
                annotation_failures: list[dict[str, Any]] = []
                for answer_index, (answer, wrapper) in enumerate(annotations, 1):
                    unanswerable, quotes = _qasper_evidence_payload(answer, wrapper)
                    annotation_id = wrapper.get("annotation_id") if isinstance(wrapper, Mapping) else None
                    if isinstance(annotation_id, list):
                        annotation_id = annotation_id[0] if annotation_id else None
                    annotation_id = str(annotation_id) if annotation_id is not None else None
                    if unanswerable:
                        annotation_failures.append({
                            "annotation_id": annotation_id,
                            "reason": "unanswerable",
                            "evidence": list(quotes),
                        })
                        continue
                    textual: list[str] = []
                    figure_quotes: list[str] = []
                    unmapped_quotes: list[str] = []
                    for quote in quotes:
                        quote = str(quote)
                        if not quote.strip():
                            continue
                        if quote.lstrip().upper().startswith("FLOAT SELECTED"):
                            figure_quotes.append(quote)
                            continue
                        if quote in source_evidence:
                            textual.append(quote)
                        else:
                            unmapped_quotes.append(quote)
                    if unmapped_quotes:
                        # A partial annotation must never be converted into a
                        # positive label by dropping its bad span.  Record all
                        # original evidence and exclude the whole query below.
                        annotation_failures.append({
                            "annotation_id": annotation_id,
                            "reason": "unmapped_evidence",
                            "evidence": list(quotes),
                            "unmapped_evidence": unmapped_quotes,
                        })
                        continue
                    if not textual:
                        annotation_failures.append({
                            "annotation_id": annotation_id,
                            "reason": "figure_only" if figure_quotes else "no_text_evidence",
                            "evidence": list(quotes),
                        })
                        continue
                    valid.append((answer, list(dict.fromkeys(textual)), annotation_id))
                unmapped = [failure for failure in annotation_failures if failure["reason"] == "unmapped_evidence"]
                if unmapped:
                    excluded.append({
                        "query_id": query_id,
                        "reason": "unmapped_evidence",
                        "source_file": path.name,
                        "annotation_failures": annotation_failures,
                        "original_evidence": [
                            quote
                            for failure in unmapped
                            for quote in failure.get("evidence", [])
                        ],
                    })
                    continue
                if not valid:
                    reasons = [failure["reason"] for failure in annotation_failures]
                    reason = "unanswerable" if reasons and all(item == "unanswerable" for item in reasons) else (
                        "figure_only" if reasons and all(item == "figure_only" for item in reasons) else "no_text_evidence"
                    )
                    excluded.append({
                        "query_id": query_id,
                        "reason": reason,
                        "source_file": path.name,
                        "annotation_failures": annotation_failures,
                        "original_evidence": [
                            quote
                            for failure in annotation_failures
                            for quote in failure.get("evidence", [])
                        ],
                    })
                    continue
                evidence_sets = {frozenset(item[1]) for item in valid}
                if len(evidence_sets) > 1 or len(valid) != len(annotations):
                    # The quality/core layer cannot infer that a review marker
                    # makes a positive label unsafe.  Keep every source paper,
                    # but exclude this query as a unit so qrels are never
                    # silently chosen from one annotator's answer.
                    excluded.append({
                        "query_id": query_id,
                        "reason": "annotation_disagreement",
                        "source_file": path.name,
                        "annotation_ids": [
                            item[2] for item in valid if item[2] is not None
                        ] + [
                            failure["annotation_id"]
                            for failure in annotation_failures
                            if failure.get("annotation_id") is not None
                        ],
                        "valid_annotation_ids": [item[2] for item in valid if item[2] is not None],
                        "invalid_annotations": annotation_failures,
                        "original_evidence": [
                            quote
                            for answer, quotes, _annotation_id in valid
                            for quote in quotes
                        ] + [
                            quote
                            for failure in annotation_failures
                            for quote in failure.get("evidence", [])
                        ],
                        "policy": "exclude the whole query when annotations disagree or any annotation lacks valid textual evidence",
                    })
                    continue
                query_metadata = {
                    "paper_id": paper_id,
                    "source_file": path.name,
                    "annotation_count": len(annotations),
                    "valid_annotation_count": len(valid),
                }
                queries.append({"_id": query_id, "text": question, "language": str(spec.get("language") or ""), "metadata": query_metadata})
                for answer, quotes, annotation_id in valid:
                    for quote in quotes:
                        start = text.find(quote)
                        if start < 0:
                            # Source equality above should make this impossible;
                            # keep the invariant explicit if document assembly changes.
                            raise ValueError(f"QASPER evidence disappeared from assembled paper {paper_id!r}")
                        item = {"query_id": query_id, "doc_id": paper_id, "quote": quote, "start": start, "end": start + len(quote)}
                        if annotation_id is not None:
                            item["annotation_id"] = annotation_id
                        if item not in evidence:
                            evidence.append(item)
                qrels[query_id] = {paper_id: 1.0}
    bundle["documents"] = documents
    bundle["queries"] = queries
    bundle["evidence"] = evidence
    bundle["qrels"] = qrels
    bundle["provenance"].update({
        "excluded": excluded,
        "annotation_review": [],
        "annotation_policy": (
            "retain evidence only when all annotators agree on the exact paragraph or section-heading evidence; "
            "exclude a whole query when any annotation has unmapped evidence, no textual evidence, or disagrees"
        ),
        "evidence_policy": (
            "only exact original paragraphs or section headings are evidence; unmapped evidence excludes the "
            "whole query without truncating labels; unanswerable, figure-only, and no-text queries are excluded"
        ),
    })
    return bundle


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_element(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _local_name(element.tag) == name:
            return element
    return None


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _load_pmc(spec: Mapping[str, Any], snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _base_bundle(spec, manifest)
    files = [
        path
        for path in _find_all_files(snapshot)
        if path.suffix.lower() in {".xml", ".jats"}
        and "source" not in path.relative_to(snapshot).parts
    ]
    if not files:
        raise ConnectorError("pmc_jats snapshot has no XML files")
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise ConnectorError(f"invalid JATS XML: {path}") from exc
        license_nodes = [element for element in root.iter() if _local_name(element.tag) == "license"]
        license_text = " ".join(_element_text(element) for element in license_nodes).strip()
        if not license_text:
            raise ConnectorError(f"JATS source has no actual license element: {path}")
        ids = [element for element in root.iter() if _local_name(element.tag) in {"article-id", "article_id"}]
        pmc_id = next((_element_text(element) for element in ids if str(element.attrib.get("pub-id-type", "")).lower() in {"pmc", "pmcid"}), "")
        document_id = pmc_id or path.stem
        title = _element_text(_first_element(root, "article-title"))
        sections: list[dict[str, Any]] = []
        text_parts: list[str] = [title] if title else []
        body = _first_element(root, "body") or root
        for section in body.iter():
            if _local_name(section.tag) != "sec":
                continue
            section_title = _element_text(next((child for child in section if _local_name(child.tag) == "title"), None))
            paragraphs = [_element_text(child) for child in section if _local_name(child.tag) == "p" and _element_text(child)]
            if not paragraphs:
                continue
            sections.append({"title": section_title, "paragraphs": paragraphs})
            if section_title:
                text_parts.append(section_title)
            text_parts.extend(paragraphs)
        if not sections:
            paragraphs = [_element_text(element) for element in body.iter() if _local_name(element.tag) == "p" and _element_text(element)]
            text_parts.extend(paragraphs)
        bundle["documents"].append({
            "_id": str(document_id),
            "title": title,
            "text": "\n\n".join(text_parts),
            "language": str(spec.get("language") or ""),
            "metadata": {"license": license_text, "sections": sections, "source_file": path.name},
        })
    return bundle


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored = 0
        self._title_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._ignored += 1
        if tag == "title" and not self._ignored:
            self._title_depth += 1
        if tag in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "br", "tr"} and not self._ignored:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag in {"script", "style", "noscript", "template"} and self._ignored:
            self._ignored -= 1
        if tag in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "tr"} and not self._ignored:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        text = html.unescape(data)
        if text.strip():
            self.parts.append(text)
            if self._title_depth:
                self.title_parts.append(text)


def _load_html_xml(spec: Mapping[str, Any], snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _base_bundle(spec, manifest)
    files = [
        path
        for path in _find_all_files(snapshot)
        if path.suffix.lower() in {".html", ".htm", ".xml", ".jats"}
        and "source" not in path.relative_to(snapshot).parts
    ]
    if not files:
        raise ConnectorError(f"{spec['connector']} snapshot has no explicit HTML/XML/text files")
    for path in files:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".html", ".htm"}:
            parser = _VisibleHTML()
            parser.feed(raw)
            text = " ".join(" ".join(parser.parts).split())
            title = " ".join(" ".join(parser.title_parts).split())
        elif path.suffix.lower() in {".xml", ".jats"}:
            try:
                root = ET.fromstring(raw)
            except ET.ParseError as exc:
                raise ConnectorError(f"invalid XML source: {path}") from exc
            title = _element_text(_first_element(root, "title") or _first_element(root, "article-title"))
            text = " ".join("".join(root.itertext()).split())
        else:
            title = path.stem
            text = raw
        bundle["documents"].append({
            "_id": path.relative_to(snapshot).as_posix(),
            "title": title,
            "text": text,
            "language": str(spec.get("language") or ""),
            "metadata": {"source_file": path.relative_to(snapshot).as_posix()},
        })
    return bundle


def _load_github(spec: Mapping[str, Any], snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _base_bundle(spec, manifest)
    options = spec.get("options") if isinstance(spec.get("options"), Mapping) else {}
    root = snapshot / "github" / "files"
    files = _find_all_files(root) if root.exists() else []
    if not files:
        raise ConnectorError("github_markdown snapshot has no files")
    for path in files:
        if path.suffix.lower() not in _FILE_SUFFIXES and not options.get("include_binary", False):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ConnectorError(f"GitHub source is not UTF-8 text: {path}") from exc
        relative = path.relative_to(root).as_posix()
        parts = relative.split("/")
        repo = str(options.get("repo", ""))
        metadata = {"path": relative, "repo": repo, "commit": spec.get("revision")}
        if len(parts) >= 3 and parts[0] == "content":
            source_language = parts[1]
            family_path = "/".join(parts[2:])
            metadata["family_id"] = f"{repo}:{family_path}" if repo else family_path
            metadata["translation_status"] = (
                "native"
                if str(spec.get("language") or options.get("language", "")) == "en" or source_language == "en"
                else str(options.get("translation_status", "translated"))
            )
            if source_language != "en" and "docs" in parts[2:]:
                docs_index = parts.index("docs", 2)
                metadata["translation_of"] = "content/en/" + "/".join(parts[docs_index:])
        bundle["documents"].append({
            "_id": relative,
            "title": Path(relative).name,
            "text": text,
            "language": str(spec.get("language") or options.get("language", "")),
            "metadata": metadata,
        })
    if not bundle["documents"]:
        raise ConnectorError("github_markdown allowlist contained no supported text files")
    return bundle


@dataclass(frozen=True)
class _Connector:
    fetch: Callable[[dict[str, Any], _FetchState], str]
    load: Callable[[Mapping[str, Any], Path, Mapping[str, Any]], dict[str, Any]]


_CONNECTOR_REGISTRY: dict[str, _Connector] = {
    "generic_jsonl": _Connector(_fetch_generic, _load_generic),
    "beir": _Connector(_fetch_beir, _load_beir),
    "wixqa": _Connector(_fetch_wixqa, _load_wixqa),
    "worksurface": _Connector(_fetch_worksurface, _load_worksurface),
    "qasper": _Connector(_fetch_qasper, _load_qasper),
    "github_markdown": _Connector(_fetch_github, _load_github),
    "pmc_jats": _Connector(_fetch_pmc, _load_pmc),
    "eurlex": _Connector(_fetch_explicit_html_xml, _load_html_xml),
    "edgar": _Connector(_fetch_explicit_html_xml, _load_html_xml),
}
from .squad import load_squad
_CONNECTOR_REGISTRY["squad"] = _Connector(_fetch_generic, load_squad)
# Public read-only-by-convention view for catalog builders that need to inspect
# capabilities without reaching into private dispatch details.
CONNECTORS = _CONNECTOR_REGISTRY


def list_connectors() -> list[str]:
    """Return the stable connector registry names."""

    return list(CONNECTOR_NAMES)


def load_source(spec: Mapping[str, Any], snapshot: Path) -> dict[str, Any]:
    """Load a verified raw snapshot into the canonical DCI bundle."""

    normalized = _require_spec(spec)
    manifest = _verify_snapshot(normalized, Path(snapshot))
    connector = _CONNECTOR_REGISTRY[normalized["connector"]]
    bundle = connector.load(normalized, Path(snapshot), manifest)
    if not isinstance(bundle, dict):
        raise ConnectorError(f"connector {normalized['connector']} returned a non-dict bundle")
    # Keep the public contract explicit even when a connector's source is
    # sparse.  Do not silently fill missing labels for retrieval inputs.
    for key, default in (("documents", []), ("queries", []), ("qrels", {}), ("evidence", []), ("judgments", [])):
        bundle.setdefault(key, default)
    bundle["schema_version"] = SCHEMA_VERSION
    bundle["input_kind"] = normalized["input_kind"]
    return bundle


def get_connector(name: str) -> _Connector:
    try:
        return _CONNECTOR_REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown source connector {name!r}; available: {list_connectors()}") from exc


__all__ = [
    "CONNECTOR_NAMES",
    "CONNECTORS",
    "INPUT_KINDS",
    "SCHEMA_VERSION",
    "ConnectorError",
    "DownloadError",
    "IntegrityError",
    "SnapshotConflictError",
    "fetch_source",
    "get_connector",
    "list_connectors",
    "load_source",
]
