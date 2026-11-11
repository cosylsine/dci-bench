#!/usr/bin/env python3
"""Acquire and publish the English ACORD clause-retrieval source.

The repository archive contains one nested data archive.  This script keeps
that archive and every extracted source file in an immutable raw snapshot,
records the original 0--4 grades in ``judgments.jsonl``, and publishes a
generic-jsonl bundle whose positive qrels use the explicit ``grade - 1``
mapping for grades 2--4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.connectors import (
    _find_all_files,
    _require_spec,
    _spec_fingerprint,
    load_source,
)
from dci_bench.data.pipeline.core import build_source, load_bundle, verify_release
from dci_bench.data.pipeline.download import (
    DEFAULT_MAX_ARCHIVE_BYTES,
    DEFAULT_MAX_ARCHIVE_FILES,
    DEFAULT_MAX_MEMBER_BYTES,
    canonical_json,
    copy_file,
    download_url,
    file_digest,
    safe_relative_path,
)


BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
RAW = BASE / "raw_data/gapfill_acord_en/20261031_v1"
RELEASE = BASE / "eval_data/pipeline_v1/gapfill_acord_en/20261031_v1"
RESEARCH = BASE / "eval_data/source_research_20261031_v1/acord"

OFFICIAL_REPO = "https://github.com/The-Atticus-Project/acord"
COMMIT = "942b33a7c6fed7ead362c7c296401086f8bc0321"
ARCHIVE_URL = f"https://codeload.github.com/The-Atticus-Project/acord/zip/{COMMIT}"
DATASET_MEMBER = "ACORD Dataset & ReadMe.zip"
DATASET_BYTES = 3_131_196
DATASET_SHA256 = "c9e851456a14aed161033b856e9f40f3cc9f46cf7282c2007b143420e93a65c3"
DATASET_BLOB_SHA1 = "6de8c6c01a5221fe7669bc3117cfc02e5bc02fbc"

SPEC = _require_spec(
    {
        "id": "gapfill_acord_en_20261031_v1",
        "dataset": "The-Atticus-Project/acord",
        "connector": "generic_jsonl",
        "input_kind": "retrieval",
        "revision": COMMIT,
        "domain": "legal-contract",
        "language": "en",
        "genre": "contract-clause-retrieval",
        "license": "CC BY 4.0 for dataset text (README); MIT for repository wrapper (LICENSE)",
        "official_urls": [OFFICIAL_REPO],
        "options": {
            "dataset": "The-Atticus-Project/acord",
            "split": "all",
            "archive_bytes": DATASET_BYTES,
            "archive_sha256": DATASET_SHA256,
            "archive_git_blob_sha1": DATASET_BLOB_SHA1,
            "query_source": "queries.jsonl only",
            "supplement_queries": "excluded",
        },
    }
)

RAW_POLICY = {
    "max_positive_per_query": None,
    "exclude_review_required": True,
    "drop_near_duplicates": False,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(path)
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(f"immutable destination differs: {path}")
        return
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    partial.write_bytes(payload)
    partial.rename(path)


def _write_json_once(path: Path, value: Any) -> None:
    _write_once(path, (canonical_json(value) + "\n").encode("utf-8"))


def _write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows]
    _write_once(path, "".join(lines).encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(dict(value))
    return rows


def _archive_member_name(name: str) -> str:
    return name.replace("\\", "/").rstrip("/").split("/")[-1]


def _copy_archive_member(archive: Path, member: str, destination: Path) -> bytes:
    with ZipFile(archive) as source:
        matches = [info for info in source.infolist() if _archive_member_name(info.filename) == member]
        if len(matches) != 1:
            raise ValueError(f"expected one {member!r} in {archive}, found {len(matches)}")
        data = source.read(matches[0])
    _write_once(destination, data)
    return data


def _safe_extract(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with ZipFile(archive) as source:
            infos = source.infolist()
            if len(infos) > DEFAULT_MAX_ARCHIVE_FILES:
                raise ValueError(f"archive has too many members: {len(infos)}")
            total = 0
            for info in infos:
                name = info.filename.replace("\\", "/")
                if not name or name.endswith("/"):
                    continue
                relative = safe_relative_path(name)
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError(f"archive symlink is not allowed: {name}")
                if info.file_size > DEFAULT_MAX_MEMBER_BYTES:
                    raise ValueError(f"archive member is too large: {name}")
                total += info.file_size
                if total > DEFAULT_MAX_ARCHIVE_BYTES:
                    raise ValueError("archive exceeds total size limit")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                data = source.read(info)
                if len(data) != info.file_size:
                    raise ValueError(f"truncated archive member: {name}")
                _write_once(target, data)
    except (BadZipFile, OSError) as exc:
        raise ValueError(f"invalid archive: {archive}") from exc


def _find_existing_archive(explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    candidates = sorted(
        path
        for path in (RAW / "source_inputs").glob("*")
        if path.is_file() and path.suffix.lower() == ".zip" and path.name != "acord-dataset-20261031.zip"
    )
    for candidate in candidates:
        try:
            with ZipFile(candidate) as source:
                if any(_archive_member_name(info.filename) == DATASET_MEMBER for info in source.infolist()):
                    return candidate
        except BadZipFile:
            continue
    return None


def _acquire_repository_archive(explicit: Path | None) -> tuple[Path, str]:
    existing = _find_existing_archive(explicit)
    if existing is not None:
        digest, size = file_digest(existing)
        return existing, digest
    with tempfile.TemporaryDirectory(prefix="acord-download-") as temporary:
        temporary_path = Path(temporary) / "acord-repository.zip"
        digest, size = download_url(
            ARCHIVE_URL,
            temporary_path,
            timeout=60,
            retries=4,
            max_bytes=64 * 1024 * 1024,
            overwrite=False,
        )
        destination = RAW / "source_inputs" / f"acord-repository-{COMMIT}.zip"
        copy_file(temporary_path, destination, expected_sha256=digest, expected_bytes=size, overwrite=False)
        return destination, digest


def _extract_data_archive(repository_archive: Path) -> tuple[Path, Path]:
    dataset_archive = RAW / "source_inputs" / "acord-dataset-20261031.zip"
    data = _copy_archive_member(repository_archive, DATASET_MEMBER, dataset_archive)
    if len(data) != DATASET_BYTES or _sha256_bytes(data) != DATASET_SHA256 or _git_blob_sha1(data) != DATASET_BLOB_SHA1:
        raise ValueError("ACORD nested archive failed pinned bytes/SHA256/Git-blob checks")
    temporary = Path(tempfile.mkdtemp(prefix="acord-data-"))
    extracted = temporary / "dataset"
    _safe_extract(dataset_archive, extracted)
    return dataset_archive, extracted


def _locate_dataset_root(extracted: Path) -> Path:
    candidates = sorted(path for path in extracted.rglob("corpus.jsonl") if path.is_file())
    if len(candidates) != 1:
        raise ValueError(f"expected one corpus.jsonl in nested archive, found {candidates}")
    return candidates[0].parent


def _copy_original_files(dataset_root: Path, repository_archive: Path) -> dict[str, str]:
    original_root = RAW / "source_inputs" / "acord_dataset"
    copied: dict[str, str] = {}
    for source in sorted(path for path in dataset_root.rglob("*") if path.is_file()):
        relative = source.relative_to(dataset_root).as_posix()
        destination = original_root / relative
        digest, size = file_digest(source)
        copy_file(source, destination, expected_sha256=digest, expected_bytes=size, overwrite=False)
        copied[relative] = f"sha256:{digest}:{size}"
    repository_destination = RAW / "source_inputs" / f"acord-repository-{COMMIT}.zip"
    if repository_archive.resolve() != repository_destination.resolve():
        digest, size = file_digest(repository_archive)
        copy_file(repository_archive, repository_destination, expected_sha256=digest, expected_bytes=size, overwrite=False)
    with ZipFile(repository_archive) as source:
        for member_name in ("LICENSE", "README.md"):
            matches = [info for info in source.infolist() if _archive_member_name(info.filename) == member_name]
            if len(matches) != 1:
                raise ValueError(f"expected one repository {member_name!r}, found {len(matches)}")
            _write_once(RAW / "source_inputs" / "acord_repository" / member_name, source.read(matches[0]))
    return copied


def _parse_qrels(dataset_root: Path, query_map: Mapping[str, Mapping[str, Any]], document_ids: set[str]) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    qrels: dict[str, dict[str, int]] = {}
    judgments: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for split in ("train", "valid", "test"):
        source_file = f"qrels/{split}.tsv"
        path = dataset_root / source_file
        if not path.is_file():
            raise FileNotFoundError(path)
        split_rows = 0
        split_grades: Counter[str] = Counter()
        split_queries: set[str] = set()
        positive = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["query-id", "corpus-id", "score"]:
                raise ValueError(f"unexpected qrels header in {path}: {reader.fieldnames}")
            for row_number, row in enumerate(reader, 2):
                query_id = str(row["query-id"])
                document_id = str(row["corpus-id"])
                if query_id not in query_map:
                    raise ValueError(f"unknown query {query_id!r} at {path}:{row_number}")
                if query_map[query_id].get("metadata", {}).get("split") != split:
                    raise ValueError(f"qrels/query split mismatch for {query_id!r} at {path}:{row_number}")
                if document_id not in document_ids:
                    raise ValueError(f"unknown document {document_id!r} at {path}:{row_number}")
                grade = int(row["score"])
                if grade not in range(5):
                    raise ValueError(f"grade outside 0..4 at {path}:{row_number}: {grade}")
                pair = (query_id, document_id)
                if pair in seen_pairs:
                    raise ValueError(f"duplicate qrel pair: {query_id!r}, {document_id!r}")
                seen_pairs.add(pair)
                canonical_relevance = max(grade - 1, 0)
                if canonical_relevance > 0:
                    qrels.setdefault(query_id, {})[document_id] = canonical_relevance
                judgments.append(
                    {
                        "query_id": query_id,
                        "doc_id": document_id,
                        "source_split": split,
                        "source_file": source_file,
                        "source_row": row_number,
                        "raw_grade": grade,
                        "stars": grade + 1,
                        "canonical_relevance": canonical_relevance,
                        "mapping": "max(raw_grade-1,0)",
                    }
                )
                split_rows += 1
                split_grades[str(grade)] += 1
                split_queries.add(query_id)
                positive += int(canonical_relevance > 0)
        exclusions.extend(
            {
                "kind": "label_mapping",
                "source_split": split,
                "source_file": source_file,
                "raw_grade": grade,
                "rows": split_grades[str(grade)],
                "reason": "retained in judgments; canonical qrels includes only raw_grade>=2",
                "canonical_relevance": max(grade - 1, 0),
            }
            for grade in (0, 1)
        )
        counts[split] = {
            "queries": len(split_queries),
            "pairs": split_rows,
            "positive_pairs": positive,
            "grades": dict(sorted(split_grades.items(), key=lambda item: int(item[0]))),
        }
    return qrels, judgments, exclusions, counts


def _build_canonical_inputs(dataset_root: Path) -> dict[str, Any]:
    documents_source = dataset_root / "corpus.jsonl"
    queries_source = dataset_root / "queries.jsonl"
    documents_source_rows = _read_jsonl(documents_source)
    query_source_rows = _read_jsonl(queries_source)
    documents: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    for row in documents_source_rows:
        document_id = str(row.get("_id", ""))
        text = row.get("text")
        if not document_id or not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid ACORD corpus row: {row}")
        if document_id in document_ids:
            raise ValueError(f"duplicate ACORD document ID: {document_id}")
        document_ids.add(document_id)
        documents.append(
            {
                "_id": document_id,
                "title": "",
                "text": text,
                "language": "en",
                "metadata": {
                    "source_id": document_id,
                    "source_file": "corpus.jsonl",
                    "source_split": "all",
                    "record_type": "contract_clause",
                    "source_text_sha256": _sha256_bytes(text.encode("utf-8")),
                },
            }
        )
    queries: list[dict[str, Any]] = []
    query_map: dict[str, dict[str, Any]] = {}
    for row in query_source_rows:
        query_id = str(row.get("_id", ""))
        text = row.get("text")
        source_metadata = row.get("metadata")
        if not query_id or not isinstance(text, str) or not text.strip() or not isinstance(source_metadata, Mapping):
            raise ValueError(f"invalid ACORD query row: {row}")
        if query_id in query_map:
            raise ValueError(f"duplicate ACORD query ID: {query_id}")
        split = str(source_metadata.get("split", ""))
        category = str(source_metadata.get("category", ""))
        if split not in {"train", "valid", "test"} or not category:
            raise ValueError(f"invalid ACORD query metadata: {row}")
        parent = str(source_metadata.get("parent_query_id", ""))
        metadata = dict(source_metadata)
        metadata.update(
            {
                "source_query_id": query_id,
                "source_file": "queries.jsonl",
                "source_split": split,
                "group_id": f"category:{category}",
                "variant_group_id": parent or f"query:{query_id}",
                "supplement_query_variant": False,
            }
        )
        canonical = {"_id": query_id, "text": text, "language": "en", "metadata": metadata}
        queries.append(canonical)
        query_map[query_id] = canonical
    qrels, judgments, exclusions, split_counts = _parse_qrels(dataset_root, query_map, document_ids)
    supplement = dataset_root / "supplemental materials" / "acord query (short_medium_long).tsv"
    with supplement.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["ACORD Query", "Medium Format", "Long Format"]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected supplement header: {reader.fieldnames}")
        for row_number, row in enumerate(reader, 2):
            exclusions.append(
                {
                    "kind": "query_variant",
                    "source_file": "supplemental materials/acord query (short_medium_long).tsv",
                    "source_row": row_number,
                    "query": row["ACORD Query"],
                    "medium": row["Medium Format"],
                    "long": row["Long Format"],
                    "reason": "supplement short/medium/long variants are retained as source evidence but not canonical queries",
                }
            )
    canonical_root = RAW / "canonical_input"
    _write_jsonl_once(canonical_root / "documents.jsonl", documents)
    _write_jsonl_once(canonical_root / "queries.jsonl", queries)
    _write_json_once(canonical_root / "qrels.json", {query_id: row for query_id, row in qrels.items()})
    _write_jsonl_once(canonical_root / "judgments.jsonl", judgments)
    _write_jsonl_once(canonical_root / "exclusions.jsonl", exclusions)
    return {
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "judgments": judgments,
        "exclusions": exclusions,
        "split_counts": split_counts,
    }


def _write_source_manifest(repository_archive: Path, dataset_archive: Path, canonical: Mapping[str, Any]) -> dict[str, Any]:
    files = []
    for path in _find_all_files(RAW):
        relative = path.relative_to(RAW).as_posix()
        if relative == "source_manifest.json":
            continue
        digest, size = file_digest(path)
        files.append({"path": relative, "sha256": digest, "bytes": size})
    files.sort(key=lambda item: str(item["path"]))
    manifest = {
        "schema_version": "dci-source-snapshot-v1",
        "complete": True,
        "id": SPEC["id"],
        "connector": SPEC["connector"],
        "input_kind": SPEC["input_kind"],
        "requested_spec_fingerprint": _spec_fingerprint(SPEC),
        "spec_fingerprint": _spec_fingerprint(SPEC),
        "requested_spec": SPEC,
        "actual_url": ARCHIVE_URL,
        "actual_urls": [OFFICIAL_REPO, ARCHIVE_URL],
        "resolved_revision": COMMIT,
        "layout": {
            "documents": ["canonical_input/documents.jsonl"],
            "queries": ["canonical_input/queries.jsonl"],
            "qrels": ["canonical_input/qrels.json"],
            "judgments": ["canonical_input/judgments.jsonl"],
        },
        "source_archive": {
            "repository_archive": str(repository_archive.relative_to(RAW)),
            "repository_archive_sha256": file_digest(repository_archive)[0],
            "dataset_archive": str(dataset_archive.relative_to(RAW)),
            "dataset_archive_bytes": DATASET_BYTES,
            "dataset_archive_sha256": DATASET_SHA256,
            "dataset_archive_git_blob_sha1": DATASET_BLOB_SHA1,
        },
        "counts": {
            "documents": len(canonical["documents"]),
            "queries": len(canonical["queries"]),
            "raw_judgments": len(canonical["judgments"]),
            "canonical_positive_pairs": sum(len(row) for row in canonical["qrels"].values()),
            "exclusions": len(canonical["exclusions"]),
            "source_split": canonical["split_counts"],
        },
        "files": files,
    }
    _write_json_once(RAW / "source_manifest.json", manifest)
    return manifest


def _write_research_receipt(canonical: Mapping[str, Any], raw_manifest: Mapping[str, Any], release_receipt: Mapping[str, Any]) -> Path:
    grade_semantics = {
        "0": {"stars": 1, "source_semantics": "irrelevant", "canonical_relevance": 0},
        "1": {"stars": 2, "source_semantics": "irrelevant but helpful", "canonical_relevance": 0},
        "2": {"stars": 3, "source_semantics": "partially relevant", "canonical_relevance": 1},
        "3": {"stars": 4, "source_semantics": "relevant", "canonical_relevance": 2},
        "4": {"stars": 5, "source_semantics": "highly relevant", "canonical_relevance": 3},
    }
    receipt = {
        "schema_version": "acord-acquisition-receipt-v1",
        "source": {
            "official_repository": OFFICIAL_REPO,
            "archive_url": ARCHIVE_URL,
            "revision": COMMIT,
            "dataset_archive_bytes": DATASET_BYTES,
            "dataset_archive_sha256": DATASET_SHA256,
            "dataset_archive_git_blob_sha1": DATASET_BLOB_SHA1,
            "raw_manifest_sha256": file_digest(RAW / "source_manifest.json")[0],
        },
        "license": {
            "dataset_declaration": "CC BY 4.0 (ACORD readme.docx)",
            "repository_license": "MIT (repository LICENSE)",
            "verification": "both files preserved under raw/source_inputs/acord_dataset and repository archive",
        },
        "counts": {
            "documents": len(canonical["documents"]),
            "queries": len(canonical["queries"]),
            "raw_judgments": len(canonical["judgments"]),
            "canonical_positive_pairs": sum(len(row) for row in canonical["qrels"].values()),
            "exclusions": len(canonical["exclusions"]),
            "split_counts": canonical["split_counts"],
        },
        "label_policy": {
            "source_scale": "ACORD stars 1..5 represented as raw grade 0..4",
            "canonical_mapping": "canonical_relevance=max(raw_grade-1,0)",
            "canonical_positive": "raw_grade>=2 (3..5 stars)",
            "raw_grade_0_and_1": "retained in judgments.jsonl and original qrels files; not emitted as positive qrels",
            "unjudged": "absent from qrels; never synthesized as negatives",
            "grade_semantics": grade_semantics,
        },
        "query_policy": {
            "canonical_source": "queries.jsonl",
            "supplement_variants": "excluded from canonical queries and retained in exclusions.jsonl plus original supplement file",
            "test_queries": canonical["split_counts"]["test"]["queries"],
            "all_queries": len(canonical["queries"]),
        },
        "raw_manifest": raw_manifest,
        "release": release_receipt,
    }
    mapping_destination = RESEARCH / "acord_20261031_label_mapping.json"
    _write_json_once(
        mapping_destination,
        {
            "schema_version": "acord-label-mapping-receipt-v1",
            "source": receipt["source"],
            "mapping": grade_semantics,
            "judgments": "eval_data/pipeline_v1/gapfill_acord_en/20261031_v1/judgments.jsonl",
            "canonical_qrels": "eval_data/pipeline_v1/gapfill_acord_en/20261031_v1/qrels.json",
            "unjudged_policy": "unjudged pairs are absent and not negatives",
        },
    )
    destination = RESEARCH / "acord_20261031_receipt.json"
    _write_json_once(destination, receipt)
    return destination


def acquire(*, archive: Path | None = None, skip_download: bool = False) -> dict[str, Any]:
    RAW.mkdir(parents=True, exist_ok=True)
    if skip_download:
        repository_archive = _find_existing_archive(archive)
        if repository_archive is None:
            raise FileNotFoundError("--skip-download requires a repository archive in raw/source_inputs or --archive")
        repository_digest = file_digest(repository_archive)[0]
    else:
        repository_archive, repository_digest = _acquire_repository_archive(archive)
    dataset_archive, extracted = _extract_data_archive(repository_archive)
    try:
        dataset_root = _locate_dataset_root(extracted)
        _copy_original_files(dataset_root, repository_archive)
        canonical = _build_canonical_inputs(dataset_root)
        repository_snapshot = RAW / "source_inputs" / f"acord-repository-{COMMIT}.zip"
        raw_manifest = _write_source_manifest(repository_snapshot, dataset_archive, canonical)
    finally:
        shutil.rmtree(extracted.parent, ignore_errors=False)
    # The explicit snapshot is loaded and checked before publication so a raw
    # manifest/layout mistake cannot create a release artifact.
    load_source(SPEC, RAW)
    release_receipt = build_source(SPEC, RAW.parent.parent.parent, RELEASE, snapshot=RAW, policy=RAW_POLICY)
    verified = verify_release(RELEASE)
    if verified != release_receipt:
        raise ValueError("release receipt changed during verify_release")
    bundle = load_bundle(RELEASE)
    if len(bundle["documents"]) < 1000 or len(bundle["queries"]) < 60:
        raise ValueError("ACORD release does not meet the minimum document/query counts")
    receipt_path = _write_research_receipt(canonical, raw_manifest, verified)
    return {
        "repository_archive": str(repository_archive),
        "repository_archive_sha256": repository_digest,
        "dataset_archive": str(dataset_archive),
        "raw": str(RAW),
        "release": str(RELEASE),
        "research_receipt": str(receipt_path),
        "label_mapping_receipt": str(RESEARCH / "acord_20261031_label_mapping.json"),
        "release_receipt": verified,
        "counts": {
            "documents": len(bundle["documents"]),
            "queries": len(bundle["queries"]),
            "positive_pairs": sum(len(row) for row in bundle["qrels"].values()),
            "judgments": len(bundle["judgments"]),
            "source_split": canonical["split_counts"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="existing official repository zip to reuse")
    parser.add_argument("--skip-download", action="store_true", help="reuse an archive already in raw/source_inputs")
    args = parser.parse_args()
    result = acquire(archive=args.archive, skip_download=args.skip_download)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
