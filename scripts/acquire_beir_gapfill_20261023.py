#!/usr/bin/env python3
"""Acquire and verify the bounded BEIR/Software gap-fill cells.

The default Android v1 cell is obtained from the pinned ModelScope MTEB mirror.
That mirror is a complete title-only sentence view, so v1 is not a claim that
the original Android corpus body text is present.  With ``--android-fulltext``
the script instead fetches the complete official BEIR Android corpus, queries,
and qrels members from the large ZIP using bounded HTTP Range requests; the
v2-fulltext variant retains title and body text and verifies each member's
central-directory metadata, CRC, size, and SHA-256.  The Quora cell is an
existing complete release and is only re-verified against its raw snapshot.
This script never overwrites an existing raw snapshot, release, or research
receipt.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping
import urllib.request
import zlib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.connectors import fetch_source, load_source  # noqa: E402
from dci_bench.data.pipeline.core import (  # noqa: E402
    load_bundle,
    publish_bundle,
    select_queries,
    verify_release,
)
from dci_bench.data.pipeline.download import download_url  # noqa: E402


BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
RAW_ROOT = BASE / "raw_data"
EVAL_ROOT = BASE / "eval_data"
DEFAULT_CONFIG = ROOT / "configs/data_pipeline/beir_gapfill_20261023.json"
DEFAULT_RESEARCH_ROOT = EVAL_ROOT / "source_research_20261023_round3" / "beir"

ANDROID_ID = "gapfill_beir_cqadupstack_android"
QUORA_ID = "gapfill_beir_quora"

MODELSCOPE_REPO = "MTEB/cqadupstack-retrieval"
MODELSCOPE_REVISION = "c48cee56fcdafa78803dfb8674c53b3a9752cef9"
MODELSCOPE_FILE = "data/android-00000-of-00001.json.gz"
MODELSCOPE_URL = (
    "https://www.modelscope.cn/api/v1/datasets/MTEB/cqadupstack-retrieval/repo"
    "?Source=SDK&Revision=c48cee56fcdafa78803dfb8674c53b3a9752cef9"
    "&FilePath=data/android-00000-of-00001.json.gz"
)
MODELSCOPE_BYTES = 555_190
MODELSCOPE_SHA256 = "b4e7db4c1c3879939404677876a363af2f14e0c94e21ad4455b49f790cd6bddf"

OFFICIAL_CQADUPSTACK_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"
OFFICIAL_CQADUPSTACK_BYTES = 5_343_728_040
OFFICIAL_CQADUPSTACK_CENTRAL_OFFSET = 5_343_721_733
OFFICIAL_CQADUPSTACK_CENTRAL_BYTES = 6_209
OFFICIAL_CQADUPSTACK_CENTRAL_SHA256 = "c1f8e7ec15d84e1514824dab6da3d9d5b732b9fefada956f4460ea65ecac965b"
OFFICIAL_ANDROID_MEMBERS = {
    "cqadupstack/android/corpus.jsonl": {
        "relative": "source_inputs/android/corpus.jsonl",
        "compressed_relative": "source_inputs/range_payload/android-corpus.jsonl.deflate",
        "compressed_bytes": 5_469_421,
        "uncompressed_bytes": 16_002_547,
        "crc32": "95513d82",
        "sha256": "aa1bad1bd193e374da8bf30299487d24059c7555a2d437f33e15b7e071cbae95",
        "local_offset": 5_245_822_960,
        "data_start": 5_245_823_050,
    },
    "cqadupstack/android/queries.jsonl": {
        "relative": "source_inputs/android/queries.jsonl",
        "compressed_relative": "source_inputs/range_payload/android-queries.jsonl.deflate",
        "compressed_bytes": 92_429_171,
        "uncompressed_bytes": 264_625_137,
        "crc32": "85bafb88",
        "sha256": "bb3bf827bc5d295297ce311ff3e6a3c48b0df4384688c8a3d762993ff3926997",
        "local_offset": 5_251_292_471,
        "data_start": 5_251_292_562,
    },
    "cqadupstack/android/qrels/test.tsv": {
        "relative": "source_inputs/android/qrels/test.tsv",
        "compressed_relative": "source_inputs/range_payload/android-qrels-test.tsv.deflate",
        "compressed_bytes": 8_649,
        "uncompressed_bytes": 24_781,
        "crc32": "dc792f03",
        "sha256": "8965c2259e76144f857a36b197f2dedf073e838c5196e6720990ba10251e322b",
        "local_offset": 5_245_814_219,
        "data_start": 5_245_814_311,
    },
}

EXPECTED_ANDROID = {
    "corpus_documents": 22_998,
    "test_queries": 699,
    "test_positive_pairs": 1_696,
}
EXPECTED_QUORA = {
    "corpus_documents": 522_931,
    "test_queries": 10_000,
    "test_positive_pairs": 15_675,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_bytes_once(path: Path, payload: bytes) -> str:
    """Write once, preserving a conflicting partial or completed file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"immutable destination is not a regular file: {path}")
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(f"immutable destination differs: {path}")
        return digest
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        if partial.is_symlink() or not partial.is_file():
            raise RuntimeError(f"preserved partial is not a regular file: {partial}")
        if partial.read_bytes() != payload:
            raise RuntimeError(f"preserved partial differs; refusing overwrite: {partial}")
    else:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(partial, path)
    return digest


def _write_json_once(path: Path, value: Any) -> str:
    return _write_bytes_once(path, _canonical_json(value))


def _write_jsonl_once(path: Path, rows: list[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    return _write_bytes_once(path, payload)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _http_range(url: str, start: int, end: int, archive_bytes: int) -> bytes:
    """Read one bounded byte range and fail closed on a non-range response."""

    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        content_range = response.headers.get("Content-Range")
        expected = end - start + 1
        expected_range = f"bytes {start}-{end}/{archive_bytes}"
        if response.status != 206 or len(payload) != expected or content_range != expected_range:
            raise RuntimeError(
                f"bounded ZIP range failed: status={response.status}, content-range={content_range!r}, "
                f"bytes={len(payload)}, expected={expected_range}"
            )
    return payload


def _parse_zip64_central_directory(payload: bytes) -> dict[str, dict[str, Any]]:
    """Parse central-directory records, including ZIP64 offset substitutions."""

    entries: dict[str, dict[str, Any]] = {}
    cursor = 0
    while cursor < len(payload):
        if payload[cursor:cursor + 4] != b"PK\x01\x02":
            raise ValueError(f"invalid ZIP central-directory signature at offset {cursor}")
        fields = struct.unpack_from("<4s6H3I5H2I", payload, cursor)
        _, _, _, flag, method, _, _, crc32, compressed_bytes, uncompressed_bytes, name_bytes, extra_bytes, comment_bytes, disk, _, _, local_offset = fields
        name_start = cursor + 46
        name = payload[name_start:name_start + name_bytes].decode("utf-8")
        extra_start = name_start + name_bytes
        extra = payload[extra_start:extra_start + extra_bytes]
        extra_cursor = 0
        while extra_cursor + 4 <= len(extra):
            header_id, field_bytes = struct.unpack_from("<HH", extra, extra_cursor)
            body = extra[extra_cursor + 4:extra_cursor + 4 + field_bytes]
            if header_id == 0x0001:
                body_cursor = 0
                if uncompressed_bytes == 0xFFFFFFFF:
                    uncompressed_bytes = struct.unpack_from("<Q", body, body_cursor)[0]
                    body_cursor += 8
                if compressed_bytes == 0xFFFFFFFF:
                    compressed_bytes = struct.unpack_from("<Q", body, body_cursor)[0]
                    body_cursor += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack_from("<Q", body, body_cursor)[0]
                    body_cursor += 8
                if disk == 0xFFFFFFFF:
                    disk = struct.unpack_from("<I", body, body_cursor)[0]
            extra_cursor += 4 + field_bytes
        entries[name] = {
            "name": name,
            "flag": flag,
            "method": method,
            "crc32": f"{crc32:08x}",
            "compressed_bytes": compressed_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "local_offset": local_offset,
        }
        cursor += 46 + name_bytes + extra_bytes + comment_bytes
    if cursor != len(payload):
        raise ValueError(f"central-directory parse left {len(payload) - cursor} bytes")
    return entries


def _parse_official_fulltext_files(stage: Path) -> dict[str, Any]:
    """Validate the complete official Android members without changing them."""

    corpus_path = stage / "source_inputs/android/corpus.jsonl"
    queries_path = stage / "source_inputs/android/queries.jsonl"
    qrels_path = stage / "source_inputs/android/qrels/test.tsv"
    corpus_ids: set[str] = set()
    corpus_text_chars = 0
    corpus_title_chars = 0
    with corpus_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, Mapping) or "_id" not in row:
                raise ValueError(f"official Android corpus row {index} has no _id")
            document_id = str(row["_id"])
            if document_id in corpus_ids:
                raise ValueError(f"duplicate official Android corpus ID: {document_id}")
            text = row.get("text")
            title = row.get("title", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"official Android corpus row {index} has empty text")
            if not isinstance(title, str):
                raise ValueError(f"official Android corpus row {index} has non-string title")
            corpus_ids.add(document_id)
            corpus_text_chars += len(text)
            corpus_title_chars += len(title)

    query_ids: set[str] = set()
    with queries_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, Mapping) or "_id" not in row:
                raise ValueError(f"official Android query row {index} has no _id")
            query_id = str(row["_id"])
            if query_id in query_ids:
                raise ValueError(f"duplicate official Android query ID: {query_id}")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"official Android query row {index} has empty text")
            query_ids.add(query_id)

    pair_set: set[tuple[str, str]] = set()
    qrel_query_ids: set[str] = set()
    grades: set[int] = set()
    with qrels_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if index == 1 and fields == ["query-id", "corpus-id", "score"]:
                continue
            if len(fields) != 3:
                raise ValueError(f"official Android qrels row {index} has {len(fields)} fields")
            query_id, document_id, score_text = fields
            score = int(score_text)
            pair = (str(query_id), str(document_id))
            if pair in pair_set:
                raise ValueError(f"duplicate official Android qrel pair: {pair}")
            if pair[0] not in query_ids or pair[1] not in corpus_ids:
                raise ValueError(f"official Android qrel references unknown ID: {pair}")
            pair_set.add(pair)
            qrel_query_ids.add(pair[0])
            grades.add(score)
    if len(corpus_ids) != EXPECTED_ANDROID["corpus_documents"]:
        raise ValueError(f"official Android corpus count mismatch: {len(corpus_ids)}")
    if len(query_ids) != EXPECTED_ANDROID["test_queries"]:
        raise ValueError(f"official Android query count mismatch: {len(query_ids)}")
    if len(pair_set) != EXPECTED_ANDROID["test_positive_pairs"]:
        raise ValueError(f"official Android qrel pair count mismatch: {len(pair_set)}")
    if grades != {1} or qrel_query_ids != query_ids:
        raise ValueError(f"official Android qrel grades/query coverage mismatch: grades={grades}")
    counts_by_query: dict[str, int] = {}
    for query_id, _ in pair_set:
        counts_by_query[query_id] = counts_by_query.get(query_id, 0) + 1
    return {
        "corpus_documents": len(corpus_ids),
        "test_queries": len(query_ids),
        "test_positive_pairs": len(pair_set),
        "max_positive_per_query": max(counts_by_query.values()),
        "corpus_text_chars": corpus_text_chars,
        "corpus_title_chars": corpus_title_chars,
        "qrel_grades": sorted(grades),
    }


def _expected_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    corpus = payload.get("corpus")
    queries = payload.get("queries")
    relevant = payload.get("relevant_docs")
    if not isinstance(corpus, Mapping) or not isinstance(queries, Mapping) or not isinstance(relevant, Mapping):
        raise ValueError("ModelScope payload must contain corpus, queries, and relevant_docs objects")
    corpus_ids, corpus_sentences = corpus.get("ids"), corpus.get("sentences")
    query_ids, query_sentences = queries.get("ids"), queries.get("sentences")
    rel_ids, rel_rows = relevant.get("ids"), relevant.get("relevant_docs")
    fields = (corpus_ids, corpus_sentences, query_ids, query_sentences, rel_ids, rel_rows)
    if not all(isinstance(value, list) for value in fields):
        raise ValueError("ModelScope corpus/query/relevant_docs fields must be lists")
    if len(corpus_ids) != len(corpus_sentences):
        raise ValueError("ModelScope corpus ids/sentences length mismatch")
    if len(query_ids) != len(query_sentences) or len(query_ids) != len(rel_ids) or len(query_ids) != len(rel_rows):
        raise ValueError("ModelScope query/relevant_docs length mismatch")
    if len(set(map(str, corpus_ids))) != len(corpus_ids):
        raise ValueError("ModelScope corpus contains duplicate IDs")
    if len(set(map(str, query_ids))) != len(query_ids):
        raise ValueError("ModelScope queries contain duplicate IDs")
    if [str(item) for item in query_ids] != [str(item) for item in rel_ids]:
        raise ValueError("ModelScope relevant_docs IDs do not align with query IDs")
    if any(not isinstance(text, str) or not text.strip() for text in corpus_sentences + query_sentences):
        raise ValueError("ModelScope corpus or query contains empty/non-string text")
    corpus_id_set = {str(item) for item in corpus_ids}
    positive_pairs = 0
    for index, row in enumerate(rel_rows):
        if not isinstance(row, list) or not row:
            raise ValueError(f"ModelScope relevant_docs row {index} is empty or not a list")
        seen: set[str] = set()
        for document_id in row:
            document_id = str(document_id)
            if document_id in seen:
                raise ValueError(f"duplicate ModelScope qrel pair at query index {index}: {document_id}")
            if document_id not in corpus_id_set:
                raise ValueError(f"ModelScope qrel references unknown document {document_id}")
            seen.add(document_id)
        positive_pairs += len(row)
    return {
        "corpus_documents": len(corpus_ids),
        "test_queries": len(query_ids),
        "test_positive_pairs": positive_pairs,
        "max_positive_per_query": max(len(row) for row in rel_rows),
    }


def _raw_partials(raw_root: Path, source_id: str) -> list[dict[str, Any]]:
    root = Path(raw_root) / source_id
    if not root.exists():
        return []
    result = []
    for path in sorted(root.rglob("*.partial")):
        if path.is_file():
            result.append({"path": str(path), "bytes": path.stat().st_size})
    return result


def _stage_candidates(raw_root: Path) -> list[Path]:
    root = Path(raw_root) / ANDROID_ID
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob("acquisition-*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )


def _stage_is_verified(stage: Path) -> bool:
    source_path = stage / "source_inputs" / Path(MODELSCOPE_FILE).name
    if not source_path.is_file():
        return False
    digest, size = _sha256_file(source_path)
    if digest != MODELSCOPE_SHA256 or size != MODELSCOPE_BYTES:
        return False
    derivation = stage / "data" / "derivation_manifest.json"
    if not derivation.is_file():
        return False
    try:
        manifest = _load_json(derivation)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("source", {}).get("sha256") != MODELSCOPE_SHA256:
        return False
    for name in ("documents.jsonl", "queries.jsonl", "qrels.tsv"):
        if not (stage / "data" / name).is_file():
            return False
    return True


def _official_fulltext_stage_is_verified(stage: Path) -> bool:
    manifest_path = stage / "acquisition_manifest.json"
    central_path = stage / "source_inputs/archive.central-directory.bin"
    if not manifest_path.is_file() or not central_path.is_file():
        return False
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("source_kind") != "official_beir_cqadupstack_zip_range"
        or manifest.get("archive", {}).get("bytes") != OFFICIAL_CQADUPSTACK_BYTES
        or manifest.get("archive", {}).get("central_directory_sha256") != OFFICIAL_CQADUPSTACK_CENTRAL_SHA256
    ):
        return False
    central_digest, central_bytes = _sha256_file(central_path)
    if central_digest != OFFICIAL_CQADUPSTACK_CENTRAL_SHA256 or central_bytes != OFFICIAL_CQADUPSTACK_CENTRAL_BYTES:
        return False
    for member_name, expected in OFFICIAL_ANDROID_MEMBERS.items():
        member = manifest.get("members", {}).get(member_name)
        if not isinstance(member, Mapping):
            return False
        for key in ("uncompressed_bytes", "compressed_bytes", "sha256", "crc32"):
            if member.get(key) != expected[key]:
                return False
        for relative in (expected["relative"], expected["compressed_relative"]):
            path = stage / relative
            if not path.is_file():
                return False
            digest, size = _sha256_file(path)
            expected_size = expected["uncompressed_bytes"] if relative == expected["relative"] else expected["compressed_bytes"]
            if relative == expected["relative"] and (digest != expected["sha256"] or size != expected_size):
                return False
            if relative == expected["compressed_relative"] and size != expected_size:
                return False
    try:
        counts = _parse_official_fulltext_files(stage)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return counts == manifest.get("counts")


def _make_official_fulltext_stage(raw_root: Path) -> tuple[Path, dict[str, Any]]:
    source_root = Path(raw_root) / ANDROID_ID
    for candidate in sorted(
        (path for path in source_root.glob("acquisition-*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    ):
        if _official_fulltext_stage_is_verified(candidate):
            return candidate, {"status": "reused_verified", "manifest": _load_json(candidate / "acquisition_manifest.json")}

    source_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = source_root / f"acquisition-{stamp}-official-fulltext"
    if stage.exists():
        raise FileExistsError(f"acquisition stage already exists: {stage}")
    (stage / "source_inputs/android/qrels").mkdir(parents=True)
    (stage / "source_inputs/range_payload").mkdir(parents=True)
    central = _http_range(
        OFFICIAL_CQADUPSTACK_URL,
        OFFICIAL_CQADUPSTACK_CENTRAL_OFFSET,
        OFFICIAL_CQADUPSTACK_CENTRAL_OFFSET + OFFICIAL_CQADUPSTACK_CENTRAL_BYTES - 1,
        OFFICIAL_CQADUPSTACK_BYTES,
    )
    central_sha = hashlib.sha256(central).hexdigest()
    if central_sha != OFFICIAL_CQADUPSTACK_CENTRAL_SHA256:
        raise ValueError(f"official CQADupStack central-directory hash mismatch: {central_sha}")
    _write_bytes_once(stage / "source_inputs/archive.central-directory.bin", central)
    entries = _parse_zip64_central_directory(central)
    member_manifest: dict[str, Any] = {}
    for member_name, expected in OFFICIAL_ANDROID_MEMBERS.items():
        entry = entries.get(member_name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"official CQADupStack central directory lacks {member_name}")
        for key in ("compressed_bytes", "uncompressed_bytes", "local_offset"):
            if entry.get(key) != expected[key] and key != "local_offset":
                raise ValueError(f"central-directory {member_name} {key} mismatch: {entry.get(key)}")
        if entry.get("crc32") != expected["crc32"]:
            raise ValueError(f"central-directory {member_name} CRC mismatch: {entry.get('crc32')}")
        local_offset = int(entry["local_offset"])
        local_header = _http_range(OFFICIAL_CQADUPSTACK_URL, local_offset, local_offset + 65_535, OFFICIAL_CQADUPSTACK_BYTES)
        if local_header[:4] != b"PK\x03\x04":
            raise ValueError(f"invalid local-header signature for {member_name}")
        _, _, local_flag, local_method, _, _, _, _, _, local_name_bytes, local_extra_bytes = struct.unpack_from(
            "<4s5H3I2H", local_header, 0
        )
        local_name_start = 30
        local_name = local_header[local_name_start:local_name_start + local_name_bytes].decode("utf-8")
        if local_name != member_name or local_flag != entry["flag"] or local_method != entry["method"]:
            raise ValueError(f"local-header mismatch for {member_name}: {local_name}")
        data_start = local_offset + 30 + local_name_bytes + local_extra_bytes
        if data_start != expected["data_start"]:
            raise ValueError(f"local data offset mismatch for {member_name}: {data_start}")
        compressed = _http_range(
            OFFICIAL_CQADUPSTACK_URL,
            data_start,
            data_start + int(entry["compressed_bytes"]) - 1,
            OFFICIAL_CQADUPSTACK_BYTES,
        )
        compressed_sha = hashlib.sha256(compressed).hexdigest()
        if len(compressed) != expected["compressed_bytes"]:
            raise ValueError(f"compressed member size mismatch for {member_name}")
        if int(entry["method"]) == 8:
            unpacked = zlib.decompress(compressed, -15)
        elif int(entry["method"]) == 0:
            unpacked = compressed
        else:
            raise ValueError(f"unsupported ZIP method {entry['method']} for {member_name}")
        crc_actual = f"{zlib.crc32(unpacked) & 0xFFFFFFFF:08x}"
        if len(unpacked) != expected["uncompressed_bytes"] or crc_actual != expected["crc32"]:
            raise ValueError(f"CRC/size mismatch for {member_name}: {len(unpacked)}, {crc_actual}")
        _write_bytes_once(stage / expected["relative"], unpacked)
        _write_bytes_once(stage / expected["compressed_relative"], compressed)
        member_manifest[member_name] = {
            **dict(expected),
            "central_directory": dict(entry),
            "local_name": local_name,
            "data_start": data_start,
            "compressed_sha256": compressed_sha,
            "crc32_verified": crc_actual,
        }
    counts = _parse_official_fulltext_files(stage)
    acquisition_manifest = {
        "schema_version": "dci-beir-gapfill-official-range-acquisition-v1",
        "source_kind": "official_beir_cqadupstack_zip_range",
        "created_at": _now(),
        "archive": {
            "url": OFFICIAL_CQADUPSTACK_URL,
            "bytes": OFFICIAL_CQADUPSTACK_BYTES,
            "central_directory_offset": OFFICIAL_CQADUPSTACK_CENTRAL_OFFSET,
            "central_directory_bytes": OFFICIAL_CQADUPSTACK_CENTRAL_BYTES,
            "central_directory_sha256": central_sha,
            "range_method": "HTTP Range; complete independent Android members only",
        },
        "members": member_manifest,
        "counts": counts,
        "raw_inputs": {
            "corpus": str(stage / "source_inputs/android/corpus.jsonl"),
            "queries": str(stage / "source_inputs/android/queries.jsonl"),
            "qrels": str(stage / "source_inputs/android/qrels/test.tsv"),
        },
    }
    _write_json_once(stage / "acquisition_manifest.json", acquisition_manifest)
    _write_json_once(stage / "data" / "derivation_manifest.json", {
        "schema_version": "dci-beir-gapfill-fulltext-derivation-v1",
        "source": acquisition_manifest["archive"],
        "members": member_manifest,
        "counts": counts,
        "derivation": "Use the complete official BEIR Android corpus.jsonl, queries.jsonl, and qrels/test.tsv members; preserve every title, body, query ID, qrel ID, and grade. No sampling, truncation, translation, or positive cap.",
        "source_metadata": {
            "source_family": "CQADupStack",
            "source_dataset": "BeIR/cqadupstack",
            "source_scope": "complete official BEIR CQADupStack Android corpus with title+body text, native test queries, and test qrels",
            "task_origin": "BEIR duplicate-question retrieval; Android Stack Exchange forum",
            "translation_status": "native",
            "source_language": "en",
            "source_forum": "android.stackexchange.com",
        },
    })
    return stage, {"status": "fetched_verified", "manifest": acquisition_manifest}


def _make_android_stage(raw_root: Path, spec: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    for candidate in _stage_candidates(raw_root):
        if _stage_is_verified(candidate):
            derivation = _load_json(candidate / "data" / "derivation_manifest.json")
            return candidate, {"status": "reused_verified", "derivation": derivation}

    source_root = Path(raw_root) / ANDROID_ID
    source_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = source_root / f"acquisition-{stamp}"
    if stage.exists():
        raise FileExistsError(f"acquisition stage already exists: {stage}")
    (stage / "source_inputs").mkdir(parents=True)
    (stage / "data").mkdir()
    source_path = stage / "source_inputs" / Path(MODELSCOPE_FILE).name
    digest, size = download_url(
        MODELSCOPE_URL,
        source_path,
        timeout=60,
        retries=1,
        max_bytes=300 * 1024 * 1024,
        expected_sha256=MODELSCOPE_SHA256,
        expected_bytes=MODELSCOPE_BYTES,
        overwrite=False,
    )
    with gzip.open(source_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    counts = _expected_counts(payload)
    if counts["corpus_documents"] != EXPECTED_ANDROID["corpus_documents"] or counts["test_queries"] != EXPECTED_ANDROID["test_queries"] or counts["test_positive_pairs"] != EXPECTED_ANDROID["test_positive_pairs"]:
        raise ValueError(f"ModelScope Android counts differ from pinned BEIR counts: {counts}")

    corpus = payload["corpus"]
    queries = payload["queries"]
    relevant = payload["relevant_docs"]
    source_meta = {
        "source_family": "CQADupStack",
        "source_dataset": "BeIR/cqadupstack",
        "source_scope": "complete BEIR CQADupStack Android title-only sentence view from the pinned MTEB mirror; original corpus body text is unavailable in this view",
        "task_origin": "BEIR duplicate-question retrieval; Android Stack Exchange forum",
        "translation_status": "native",
        "source_language": "en",
        "source_forum": "android.stackexchange.com",
        "source_revision": f"{MODELSCOPE_REPO}@{MODELSCOPE_REVISION}",
        "source_file": MODELSCOPE_FILE,
    }
    documents = [
        {
            "_id": str(document_id),
            "title": "",
            "text": text,
            "language": "en",
            "metadata": {
                **source_meta,
                "source_record_kind": "corpus",
                "source_id": str(document_id),
                "title_field": "not_available_in_mteb_mirror_sentence_view",
            },
        }
        for document_id, text in zip(corpus["ids"], corpus["sentences"])
    ]
    query_rows = [
        {
            "_id": str(query_id),
            "text": text,
            "language": "en",
            "metadata": {
                **source_meta,
                "source_record_kind": "test_query",
                "source_id": str(query_id),
                "split": "test",
            },
        }
        for query_id, text in zip(queries["ids"], queries["sentences"])
    ]
    qrel_lines = ["query-id\tcorpus-id\tscore"]
    for query_id, row in zip(relevant["ids"], relevant["relevant_docs"]):
        qrel_lines.extend(f"{query_id}\t{document_id}\t1" for document_id in row)
    qrels_payload = ("\n".join(qrel_lines) + "\n").encode("utf-8")
    document_path = stage / "data" / "documents.jsonl"
    query_path = stage / "data" / "queries.jsonl"
    qrel_path = stage / "data" / "qrels.tsv"
    document_sha = _write_jsonl_once(document_path, documents)
    query_sha = _write_jsonl_once(query_path, query_rows)
    qrel_sha = _write_bytes_once(qrel_path, qrels_payload)
    derivation = {
        "schema_version": "dci-beir-gapfill-raw-derivation-v1",
        "source": {
            "repository": MODELSCOPE_REPO,
            "revision": MODELSCOPE_REVISION,
            "file": MODELSCOPE_FILE,
            "url": MODELSCOPE_URL,
            "bytes": size,
            "sha256": digest,
        },
        "format": "MTEB retrieval dictionary: corpus.ids/sentences, queries.ids/sentences, relevant_docs.ids/relevant_docs",
        "derivation": "Copy every source ID and sentence; retain every source relevant_docs relation and map each BEIR positive to score 1. No query/document/positive is sampled or invented.",
        "source_metadata": source_meta,
        "counts": counts,
        "derived_files": {
            "data/documents.jsonl": {"bytes": document_path.stat().st_size, "sha256": document_sha},
            "data/queries.jsonl": {"bytes": query_path.stat().st_size, "sha256": query_sha},
            "data/qrels.tsv": {"bytes": qrel_path.stat().st_size, "sha256": qrel_sha},
        },
        "qrels": {
            "split": "test",
            "grade_mapping": "all entries in the source relevant_docs list are the official BEIR positive grade 1",
            "qrels_source": "https://huggingface.co/datasets/BeIR/cqadupstack-qrels/blob/main/android/test.tsv",
        },
    }
    _write_json_once(stage / "data" / "derivation_manifest.json", derivation)
    _write_json_once(
        stage / "acquisition_manifest.json",
        {
            "schema_version": "dci-beir-gapfill-acquisition-v1",
            "source_id": ANDROID_ID,
            "created_at": _now(),
            "source": derivation["source"],
            "counts": counts,
            "raw_input": str(source_path),
            "derivation_manifest": str(stage / "data" / "derivation_manifest.json"),
        },
    )
    return stage, {"status": "fetched_verified", "derivation": derivation}


def _dynamic_android_spec(spec: Mapping[str, Any], stage: Path) -> dict[str, Any]:
    result = copy.deepcopy(dict(spec))
    options = dict(result.get("options") or {})
    options["files"] = [
        {
            "role": "documents",
            "path": str(stage / "data" / "documents.jsonl"),
            "name": "data/documents.jsonl",
            "bytes": (stage / "data" / "documents.jsonl").stat().st_size,
            "sha256": _sha256_file(stage / "data" / "documents.jsonl")[0],
        },
        {
            "role": "queries",
            "path": str(stage / "data" / "queries.jsonl"),
            "name": "data/queries.jsonl",
            "bytes": (stage / "data" / "queries.jsonl").stat().st_size,
            "sha256": _sha256_file(stage / "data" / "queries.jsonl")[0],
        },
        {
            "role": "qrels",
            "path": str(stage / "data" / "qrels.tsv"),
            "name": "data/qrels.tsv",
            "bytes": (stage / "data" / "qrels.tsv").stat().st_size,
            "sha256": _sha256_file(stage / "data" / "qrels.tsv")[0],
        },
    ]
    result["options"] = options
    return result


def _dynamic_android_fulltext_spec(spec: Mapping[str, Any], stage: Path) -> dict[str, Any]:
    result = copy.deepcopy(dict(spec))
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "source_scope": "complete official BEIR CQADupStack Android corpus with title+body text, native test queries, and test qrels",
            "task_origin": "BEIR duplicate-question retrieval; Android Stack Exchange forum",
            "translation_status": "native",
            "source_archive": OFFICIAL_CQADUPSTACK_URL,
            "source_archive_bytes": OFFICIAL_CQADUPSTACK_BYTES,
            "source_archive_central_directory_sha256": OFFICIAL_CQADUPSTACK_CENTRAL_SHA256,
            "source_archive_member_scope": "cqadupstack/android/{corpus.jsonl,queries.jsonl,qrels/test.tsv}",
            "title_limit": "None for official fulltext corpus members; queries retain the official BEIR query text field.",
        }
    )
    result["metadata"] = metadata
    result["revision"] = "official-beir-cqadupstack.zip@2021-04-20"
    options = dict(result.get("options") or {})
    options.update(
        {
            "adapter": "official_beir_zip_android_members_range",
            "dataset": "BeIR/cqadupstack",
            "split": "test",
            "max_bytes": 300 * 1024 * 1024,
            "timeout": 120,
            "retries": 0,
            "files": [
                {
                    "role": "documents",
                    "path": str(stage / "source_inputs/android/corpus.jsonl"),
                    "name": "data/documents.jsonl",
                    "bytes": OFFICIAL_ANDROID_MEMBERS["cqadupstack/android/corpus.jsonl"]["uncompressed_bytes"],
                    "sha256": OFFICIAL_ANDROID_MEMBERS["cqadupstack/android/corpus.jsonl"]["sha256"],
                },
                {
                    "role": "queries",
                    "path": str(stage / "source_inputs/android/queries.jsonl"),
                    "name": "data/queries.jsonl",
                    "bytes": OFFICIAL_ANDROID_MEMBERS["cqadupstack/android/queries.jsonl"]["uncompressed_bytes"],
                    "sha256": OFFICIAL_ANDROID_MEMBERS["cqadupstack/android/queries.jsonl"]["sha256"],
                },
                {
                    "role": "qrels",
                    "path": str(stage / "source_inputs/android/qrels/test.tsv"),
                    "name": "data/qrels.tsv",
                    "bytes": OFFICIAL_ANDROID_MEMBERS["cqadupstack/android/qrels/test.tsv"]["uncompressed_bytes"],
                    "sha256": OFFICIAL_ANDROID_MEMBERS["cqadupstack/android/qrels/test.tsv"]["sha256"],
                },
            ],
        }
    )
    for key in ("modelscope_file", "modelscope_url", "modelscope_revision", "modelscope_bytes", "modelscope_sha256"):
        options.pop(key, None)
    result["options"] = options
    return result


def _bundle_identity(bundle: Mapping[str, Any]) -> dict[str, Any]:
    documents = bundle.get("documents", [])
    queries = bundle.get("queries", [])
    qrels = bundle.get("qrels", {})
    return {
        "document_ids": {str(row["_id"]): (row.get("title", ""), row.get("text", "")) for row in documents},
        "query_ids": {str(row["_id"]): row.get("text", "") for row in queries},
        "qrels": {str(query_id): {str(doc_id): score for doc_id, score in row.items()} for query_id, row in qrels.items()},
    }


def _compare_release_to_raw(release: Path, raw_snapshot: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    manifest = verify_release(release)
    raw_bundle = load_source(spec, raw_snapshot)
    release_bundle = load_bundle(release)
    raw_identity = _bundle_identity(raw_bundle)
    release_identity = _bundle_identity(release_bundle)
    if raw_identity["query_ids"] != release_identity["query_ids"]:
        raise ValueError(f"release queries differ from raw source: {release}")
    if raw_identity["qrels"] != release_identity["qrels"]:
        raise ValueError(f"release qrels differ from raw source: {release}")
    if raw_identity["document_ids"] != release_identity["document_ids"]:
        raise ValueError(f"release documents differ from raw source: {release}")
    expected = {
        "documents": len(raw_identity["document_ids"]),
        "queries": len(raw_identity["query_ids"]),
        "positive_pairs": sum(len(row) for row in raw_identity["qrels"].values()),
    }
    observed = {
        "documents": manifest.get("num_documents"),
        "queries": manifest.get("num_queries"),
        "positive_pairs": manifest.get("num_positive_pairs"),
    }
    if observed != expected:
        raise ValueError(f"release manifest counts differ from raw source: {release}: {observed} != {expected}")
    return {
        "release": str(release),
        "raw_snapshot": str(raw_snapshot),
        "release_manifest": manifest,
        "raw_release_exact_ids_text_qrels": True,
        "counts": expected,
    }


def _verify_existing_release(release: Path) -> dict[str, Any]:
    manifest = verify_release(release)
    raw_snapshot_value = manifest.get("raw_snapshot")
    if not raw_snapshot_value:
        raise ValueError(f"release has no raw_snapshot: {release}")
    raw_snapshot = Path(raw_snapshot_value)
    # ``source_manifest.requested_spec`` is redacted for URLs by the generic
    # snapshot writer.  The release manifest keeps the exact specification
    # used for its raw snapshot, so prefer that for the fingerprint check.
    spec = manifest.get("specification")
    if not isinstance(spec, Mapping):
        raw_manifest = _load_json(raw_snapshot / "source_manifest.json")
        spec = raw_manifest.get("requested_spec")
    if not isinstance(spec, Mapping):
        raise ValueError(f"raw snapshot has no requested_spec: {raw_snapshot}")
    return _compare_release_to_raw(release, raw_snapshot, spec)


def _build_android(spec: Mapping[str, Any], stage: Path, release: Path, policy: Mapping[str, Any], raw_root: Path) -> dict[str, Any]:
    dynamic_spec = _dynamic_android_spec(spec, stage)
    snapshot = fetch_source(dynamic_spec, raw_root)
    if release.exists():
        return _verify_existing_release(release)
    bundle = load_source(dynamic_spec, snapshot)
    selection = dynamic_spec.get("selection", {})
    bundle = select_queries(bundle, limit=selection.get("max_queries"), seed=selection.get("seed", 20261022))
    metadata = dynamic_spec.get("metadata") if isinstance(dynamic_spec.get("metadata"), Mapping) else {}
    bundle.setdefault("provenance", {}).update(
        {
            "domain": dynamic_spec.get("domain", "software"),
            "language": dynamic_spec.get("language", "en"),
            "genre": dynamic_spec.get("genre", dynamic_spec.get("options", {}).get("genre", "duplicate_question_retrieval")),
            "source_id": dynamic_spec["id"],
            "source_family": metadata.get("source_family", "CQADupStack"),
            "source_scope": metadata.get("source_scope"),
            "task_origin": metadata.get("task_origin"),
            "translation_status": metadata.get("translation_status", "native"),
            "split": dynamic_spec.get("options", {}).get("split", "test"),
        }
    )
    manifest = publish_bundle(
        bundle,
        release,
        specification=dynamic_spec,
        raw_snapshot=snapshot,
        policy=dict(policy),
    )
    return _compare_release_to_raw(release, snapshot, dynamic_spec) | {"build_manifest": manifest}


def _build_android_fulltext(spec: Mapping[str, Any], stage: Path, release: Path, policy: Mapping[str, Any], raw_root: Path) -> dict[str, Any]:
    dynamic_spec = _dynamic_android_fulltext_spec(spec, stage)
    if release.exists():
        return _verify_existing_release(release)
    snapshot = fetch_source(dynamic_spec, raw_root)
    bundle = load_source(dynamic_spec, snapshot)
    expected = {
        "documents": EXPECTED_ANDROID["corpus_documents"],
        "queries": EXPECTED_ANDROID["test_queries"],
        "positive_pairs": EXPECTED_ANDROID["test_positive_pairs"],
    }
    observed = {
        "documents": len(bundle["documents"]),
        "queries": len(bundle["queries"]),
        "positive_pairs": sum(len(row) for row in bundle["qrels"].values()),
    }
    if observed != expected:
        raise ValueError(f"official Android fulltext bundle counts differ: {observed} != {expected}")
    metadata = dynamic_spec.get("metadata") if isinstance(dynamic_spec.get("metadata"), Mapping) else {}
    bundle.setdefault("provenance", {}).update(
        {
            "domain": dynamic_spec.get("domain", "software"),
            "language": dynamic_spec.get("language", "en"),
            "genre": dynamic_spec.get("genre", "duplicate_question_retrieval"),
            "source_id": dynamic_spec["id"],
            "source_family": metadata.get("source_family", "CQADupStack"),
            "source_scope": metadata.get("source_scope"),
            "task_origin": metadata.get("task_origin"),
            "translation_status": metadata.get("translation_status", "native"),
            "split": dynamic_spec.get("options", {}).get("split", "test"),
            "source_archive": OFFICIAL_CQADUPSTACK_URL,
        }
    )
    manifest = publish_bundle(
        bundle,
        release,
        specification=dynamic_spec,
        raw_snapshot=snapshot,
        policy=dict(policy),
    )
    result = _compare_release_to_raw(release, snapshot, dynamic_spec) | {"build_manifest": manifest}
    if result["counts"] != {
        "documents": EXPECTED_ANDROID["corpus_documents"],
        "queries": EXPECTED_ANDROID["test_queries"],
        "positive_pairs": EXPECTED_ANDROID["test_positive_pairs"],
    }:
        raise ValueError(f"official Android fulltext release counts differ: {result['counts']}")
    return result


def _verify_quora(release: Path) -> dict[str, Any]:
    result = _verify_existing_release(release)
    counts = result["counts"]
    if counts != {
        "documents": EXPECTED_QUORA["corpus_documents"],
        "queries": EXPECTED_QUORA["test_queries"],
        "positive_pairs": EXPECTED_QUORA["test_positive_pairs"],
    }:
        raise ValueError(f"existing Quora release counts differ from pinned counts: {counts}")
    return result


def _acquire_android_fulltext(
    config_path: Path,
    research_root: Path,
    release_name: str,
) -> dict[str, Any]:
    config = _load_json(config_path)
    if not isinstance(config, Mapping) or not isinstance(config.get("sources"), list):
        raise ValueError(f"invalid BEIR gap-fill config: {config_path}")
    sources = {str(item["id"]): item for item in config["sources"]}
    if ANDROID_ID not in sources:
        raise ValueError(f"config has no {ANDROID_ID}")
    raw_root = Path(config.get("raw_root", RAW_ROOT))
    eval_root = Path(config.get("eval_root", EVAL_ROOT))
    policy = config.get("quality_policy", {"max_positive_per_query": None, "exclude_review_required": False})
    if not isinstance(policy, Mapping):
        raise ValueError("quality_policy must be an object")
    research_root.mkdir(parents=True, exist_ok=True)
    stage, acquisition = _make_official_fulltext_stage(raw_root)
    release = eval_root / "pipeline_v1" / ANDROID_ID / release_name
    build = _build_android_fulltext(sources[ANDROID_ID], stage, release, policy, raw_root)
    source = {
        "id": ANDROID_ID,
        "variant": "official_fulltext",
        "domain": "software",
        "language": "en",
        "source_family": "CQADupStack",
        "source_scope": "complete official BEIR CQADupStack Android corpus with title+body text, native test queries, and test qrels",
        "task_origin": "BEIR duplicate-question retrieval; Android Stack Exchange forum",
        "translation_status": "native",
        "acquisition": acquisition,
        "acquisition_stage": str(stage),
        "raw_snapshot": build["raw_snapshot"],
        "release": build["release"],
        "counts": build["counts"],
        "state": build["release_manifest"].get("state"),
        "bundle_hash": build["release_manifest"].get("bundle_hash"),
        "raw_release_exact_ids_text_qrels": build["raw_release_exact_ids_text_qrels"],
        "status": "published_verified",
        "official_member_validation": acquisition["manifest"]["counts"],
    }
    result: dict[str, Any] = {
        "schema_version": "dci-beir-gapfill-acquisition-v2-fulltext",
        "variant": "official_fulltext",
        "created_at": _now(),
        "config_path": str(config_path),
        "raw_root": str(raw_root),
        "eval_root": str(eval_root),
        "research_root": str(research_root),
        "quality_policy": dict(policy),
        "download_limits": config.get("download_limits", {}),
        "sources": [source],
        "preserved_partials": {ANDROID_ID: _raw_partials(raw_root, ANDROID_ID)},
        "blocked_or_not_used": [
            {
                "source": "beir_cqadupstack_full_download",
                "status": "bounded_members_only",
                "reason": "The 5,343,728,040-byte archive was not downloaded as a whole; only complete Android corpus, queries, and test qrels members were fetched by verified HTTP Range requests.",
                "url": OFFICIAL_CQADUPSTACK_URL,
            },
            {
                "source": "Hugging Face direct Android parquet",
                "status": "not_retried",
                "reason": "Round2 receipts record network-unreachable direct HF attempts; official BEIR ZIP members were available through bounded Range requests.",
                "url": "https://huggingface.co/datasets/BeIR/cqadupstack/tree/main/android",
            },
        ],
        "official_links": {
            "beir_catalog": "https://github.com/beir-cellar/beir/blob/main/README.md",
            "beir_paper": "https://arxiv.org/abs/2104.08663",
            "cqadupstack_dataset_card": "https://huggingface.co/datasets/BeIR/cqadupstack",
            "cqadupstack_qrels": "https://huggingface.co/datasets/BeIR/cqadupstack-qrels/blob/main/android/test.tsv",
            "official_archive": OFFICIAL_CQADUPSTACK_URL,
            "modelscope_title_only_reference": "https://www.modelscope.cn/datasets/MTEB/cqadupstack-retrieval",
        },
    }
    result["completed_at"] = _now()
    result["acquisition_status"] = "published_verified"
    research_path = research_root / f"{ANDROID_ID}.{release_name}.research.json"
    if research_path.exists():
        existing = _load_json(research_path)
        if not isinstance(existing, Mapping):
            raise ValueError(f"fulltext research record is not an object: {research_path}")
        stable_existing = copy.deepcopy(dict(existing))
        stable_current = copy.deepcopy(dict(result))
        for value in (stable_existing, stable_current):
            for item in value.get("sources", []):
                if isinstance(item, Mapping):
                    item.pop("acquisition", None)
        for key in ("sources", "official_links", "quality_policy", "acquisition_status"):
            if stable_existing.get(key) != stable_current.get(key):
                raise ValueError(f"sealed fulltext research record differs for {key}: {research_path}")
        return dict(existing)
    _write_json_once(research_path, result)
    receipt_path = research_root / f"{ANDROID_ID}.{release_name}.receipt.json"
    receipt = {
        "schema_version": "dci-beir-gapfill-receipt-v2-fulltext",
        "created_at": result["created_at"],
        "completed_at": result["completed_at"],
        "source": source,
        "official_links": result["official_links"],
        "quality_policy": dict(policy),
        "validation_boundary": "complete official Android ZIP members fetched by Range, CRC/hash verified, and raw/release exactness verified; no model benchmark implied",
    }
    _write_json_once(receipt_path, receipt)
    return result


def acquire(config_path: Path = DEFAULT_CONFIG, research_root: Path = DEFAULT_RESEARCH_ROOT, source_ids: list[str] | None = None, release_name: str = "v1", android_fulltext: bool = False) -> dict[str, Any]:
    if android_fulltext:
        if source_ids and set(source_ids) != {ANDROID_ID}:
            raise ValueError("--android-fulltext accepts only gapfill_beir_cqadupstack_android")
        if release_name == "v1":
            release_name = "v2-fulltext"
        return _acquire_android_fulltext(config_path, research_root, release_name)
    config = _load_json(config_path)
    if not isinstance(config, Mapping) or not isinstance(config.get("sources"), list):
        raise ValueError(f"invalid BEIR gap-fill config: {config_path}")
    sources = {str(item["id"]): item for item in config["sources"]}
    selected = source_ids or [ANDROID_ID, QUORA_ID]
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")
    raw_root = Path(config.get("raw_root", RAW_ROOT))
    eval_root = Path(config.get("eval_root", EVAL_ROOT))
    policy = config.get("quality_policy", {"max_positive_per_query": None, "exclude_review_required": False})
    if not isinstance(policy, Mapping):
        raise ValueError("quality_policy must be an object")
    research_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": "dci-beir-gapfill-acquisition-v2",
        "created_at": _now(),
        "config_path": str(config_path),
        "raw_root": str(raw_root),
        "eval_root": str(eval_root),
        "research_root": str(research_root),
        "quality_policy": dict(policy),
        "download_limits": config.get("download_limits", {}),
        "sources": [],
        "preserved_partials": {
            source_id: _raw_partials(raw_root, source_id)
            for source_id in selected
        },
        "blocked_or_not_used": [
            {
                "source": "beir_cqadupstack_full_download",
                "status": "blocked",
                "reason": "Official full CQADupStack archive is 5,343,728,040 bytes and violates the per-source/total limits; one complete Android forum is used instead.",
                "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            },
            {
                "source": "Hugging Face direct Android parquet",
                "status": "not_retried",
                "reason": "Round2 receipts record network-unreachable direct HF attempts; the pinned ModelScope MTEB mirror is used.",
                "url": "https://huggingface.co/datasets/BeIR/cqadupstack/tree/main/android",
            },
        ],
        "official_links": {
            "beir_catalog": "https://github.com/beir-cellar/beir/blob/main/README.md",
            "beir_paper": "https://arxiv.org/abs/2104.08663",
            "cqadupstack_qrels": "https://huggingface.co/datasets/BeIR/cqadupstack-qrels/blob/main/android/test.tsv",
            "modelscope_mirror": "https://www.modelscope.cn/datasets/MTEB/cqadupstack-retrieval",
            "modelscope_revision": f"https://www.modelscope.cn/datasets/MTEB/cqadupstack-retrieval/tree/{MODELSCOPE_REVISION}",
            "quora_archive": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
        },
    }

    if ANDROID_ID in selected:
        stage, acquisition = _make_android_stage(raw_root, sources[ANDROID_ID])
        release = eval_root / "pipeline_v1" / ANDROID_ID / release_name
        android_result = _build_android(sources[ANDROID_ID], stage, release, policy, raw_root)
        result["sources"].append(
            {
                "id": ANDROID_ID,
                "domain": "software",
                "language": "en",
                "source_family": "CQADupStack",
                "source_scope": "complete Android forum title-only sentence view from the pinned MTEB mirror; original corpus body text is unavailable in this view",
                "source_view": "mteb_title_only_sentence_view",
                "corpus_text_completeness": "title_only; original BEIR corpus body text is unavailable in this v1 view",
                "task_origin": "BEIR duplicate-question retrieval",
                "translation_status": "native",
                "acquisition": acquisition,
                "acquisition_stage": str(stage),
                "raw_snapshot": android_result["raw_snapshot"],
                "release": android_result["release"],
                "counts": android_result["counts"],
                "state": android_result["release_manifest"].get("state"),
                "bundle_hash": android_result["release_manifest"].get("bundle_hash"),
                "raw_release_exact_ids_text_qrels": android_result["raw_release_exact_ids_text_qrels"],
                "status": "published_verified",
            }
        )

    if QUORA_ID in selected:
        release = eval_root / "pipeline_v1" / QUORA_ID / release_name
        quora_result = _verify_quora(release)
        result["sources"].append(
            {
                "id": QUORA_ID,
                "domain": "general_qa",
                "language": "en",
                "source_family": "Quora",
                "source_scope": "complete official BEIR Quora corpus with test queries and test qrels",
                "task_origin": "BEIR duplicate-question retrieval",
                "translation_status": "native",
                "raw_snapshot": quora_result["raw_snapshot"],
                "release": quora_result["release"],
                "counts": quora_result["counts"],
                "state": quora_result["release_manifest"].get("state"),
                "bundle_hash": quora_result["release_manifest"].get("bundle_hash"),
                "raw_release_exact_ids_text_qrels": quora_result["raw_release_exact_ids_text_qrels"],
                "status": "reused_verified",
            }
        )
    result["completed_at"] = _now()
    result["acquisition_status"] = "published_verified" if all(item["status"] in {"published_verified", "reused_verified"} for item in result["sources"]) else "partial_or_failed"
    # The round3 directory is new by contract.  On a rerun, preserve the first
    # sealed research record instead of overwriting evidence with new timestamps.
    research_path = research_root / "research.json"
    if research_path.exists():
        existing = _load_json(research_path)
        if not isinstance(existing, Mapping):
            raise ValueError(f"research record is not an object: {research_path}")
        def stable_sources(value: Any) -> Any:
            if not isinstance(value, list):
                return value
            stable = []
            for item in value:
                if not isinstance(item, Mapping):
                    stable.append(item)
                    continue
                row = dict(item)
                row.pop("acquisition", None)
                stable.append(row)
            return stable

        if stable_sources(existing.get("sources")) != stable_sources(result.get("sources")):
            raise ValueError(f"sealed research record differs for sources: {research_path}")
        for key in ("official_links", "quality_policy", "acquisition_status"):
            if existing.get(key) != result.get(key):
                raise ValueError(f"sealed research record differs for {key}: {research_path}")
        return dict(existing)
    else:
        _write_json_once(research_path, result)
    for source in result["sources"]:
        receipt_path = research_root / f"{source['id']}.receipt.json"
        receipt = {
            "schema_version": "dci-beir-gapfill-receipt-v2",
            "created_at": result["created_at"],
            "completed_at": result["completed_at"],
            "source": source,
            "official_links": result["official_links"],
            "quality_policy": dict(policy),
            "validation_boundary": "data prepared and raw/release exactness verified; no model benchmark implied",
        }
        if receipt_path.exists():
            if _load_json(receipt_path) != receipt:
                raise FileExistsError(f"research receipt already exists and differs: {receipt_path}")
        else:
            _write_json_once(receipt_path, receipt)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--release-name", default="v1")
    parser.add_argument("--source", nargs="+", dest="source_ids")
    parser.add_argument(
        "--android-fulltext",
        action="store_true",
        help="fetch complete official Android ZIP members by HTTP Range and publish the v2-fulltext variant",
    )
    args = parser.parse_args(argv)
    result = acquire(
        config_path=args.config,
        research_root=args.research_root,
        source_ids=args.source_ids,
        release_name=args.release_name,
        android_fulltext=args.android_fulltext,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["acquisition_status"] == "published_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
