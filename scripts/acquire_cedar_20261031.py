#!/usr/bin/env python3
"""Acquire and conservatively adapt the official CEDAR Chinese debate data.

The archive is retained byte-for-byte.  Queries and qrels are derived only
from explicit ``relations`` edges whose source and target text map uniquely to
source annotation units in the same debate.  Same-topic text is never linked
implicitly.  The occurrence-level parent release retains its audit review
rows; a separate derived release excludes ambiguous queries and collapses
exact-text aliases while retaining their provenance.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.connectors import _require_spec, _spec_fingerprint
from dci_bench.data.pipeline.core import build_source, load_bundle, publish_bundle, verify_release
from dci_bench.data.pipeline.download import download_url, file_digest


BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
SOURCE = "gapfill_cedar_zh"
REVISION = "cb46d01672c9f1fd203eca6ee57bca4f4bc9fbb7"
TREE_SHA = "426d2957dc9322108f1b45640bcbbeb0d25d4175"
ARCHIVE_BLOB_SHA1 = "711ca03fcbb6dfab20806b9eb4c643603f0ecd25"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/VelikayaScarlet/CEDAR/"
    f"{REVISION}/CEDAR.zip"
)
ARCHIVE_BYTES = 21_326_349
ARCHIVE_SHA256 = "c7f903292feec69b1aba98ae96843881708c94700f6d36b039ff685dd2a3abdd"
ARCHIVE_MEMBER = "CEDAR.json"
ARCHIVE_MEMBER_BYTES = 82_868_267
ARCHIVE_MEMBER_SHA256 = "594809cf4dd1b82c44115b594f7068b8cd71ea434b64169df37d9078dabb87e7"
LICENSE = "Apache-2.0"
LANGUAGE = "zh"
SEED = 42

RAW = BASE / "raw_data/gapfill_cedar_zh/20261031_v1"
CANONICAL = RAW / "canonical_input"
ARCHIVE = RAW / "source_inputs/CEDAR.zip"
RELEASE = BASE / "eval_data/pipeline_v1/gapfill_cedar_zh/20261031_v1"
REVIEWED_RELEASE = BASE / "eval_data/pipeline_v1/gapfill_cedar_zh/20261031_v1_reviewed"
RESEARCH = BASE / "eval_data/source_research_20261031_v1/cedar"
REVIEWED_RECEIPT = RESEARCH / "cedar_20261031_reviewed.json"
REVIEWED_ALIASES = RESEARCH / "cedar_20261031_reviewed_aliases.json"

POLICY = {
    "max_positive_per_query": None,
    "exclude_review_required": False,
    "drop_near_duplicates": False,
    "max_document_bytes": 16_777_216,
    "max_query_bytes": 16_777_216,
}

SPEC: dict[str, Any] = {
    "id": SOURCE,
    "connector": "generic_jsonl",
    "input_kind": "retrieval",
    "domain": "argumentation",
    "language": LANGUAGE,
    "revision": REVISION,
    "license": LICENSE,
    "metadata": {
        "source_family": "CEDAR",
        "source_scope": "600 native Mandarin oral debates; source annotation units only",
        "translation_status": "non_translated",
        "task_origin": "Explicit relation edge to claim retrieval adaptation",
        "document_granularity": "source annotated claim/evidence span",
        "official_urls": [
            "https://github.com/VelikayaScarlet/CEDAR",
            "https://aclanthology.org/2026.acl-long.238.pdf",
        ],
        "label_basis": "CEDAR.json relations.from_text/to_text/relation",
    },
    "options": {
        "dataset": "VelikayaScarlet/CEDAR",
        "path": str(CANONICAL),
    },
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Immutable artifact differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def json_once(path: Path, value: Any) -> None:
    once(path, json_bytes(value))


def write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    encoded = b"".join(
        (json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n").encode()
        for row in rows
    )
    once(path, encoded)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _topic_group(topic: str) -> str:
    return "topic-" + hashlib.sha256(topic.encode("utf-8")).hexdigest()[:16]


def _debate_id(index: int, competition: str) -> str:
    suffix = hashlib.sha256(competition.encode("utf-8")).hexdigest()[:12]
    return f"debate-{index:04d}-{suffix}"


def _pointer(debate_index: int, *parts: str | int) -> str:
    return "/".join(["CEDAR.json", str(debate_index), *(str(part) for part in parts)])


def _valid_span(payload: Mapping[str, Any], text: str) -> tuple[bool, str | None]:
    start = payload.get("start_offset")
    end = payload.get("end_offset")
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        return False, "invalid_offset_type"
    if start < 0 or end <= start or end - start != len(text):
        return False, "invalid_or_truncated_offset"
    return True, None


def _record_annotation(
    *,
    role: str,
    debate_index: int,
    entry_index: int,
    source_type: str,
    source_key: str | None,
    payload: Mapping[str, Any],
    topic: str,
    competition: str,
    topic_type: Any,
    result: Any,
    debate_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    text = _text(payload.get("content"))
    if text is None:
        return None, "missing_or_empty_text"
    valid, reason = _valid_span(payload, text)
    if not valid:
        return None, reason
    annotation_id = f"{role}-{debate_index:04d}-{entry_index:05d}"
    source_path = _pointer(
        debate_index,
        source_type,
        entry_index,
        *( [source_key] if source_key is not None else [] ),
    )
    group_id = _topic_group(topic)
    metadata = {
        "source_family": "CEDAR",
        "annotation_source": "source",
        "label_basis": "CEDAR native human annotation",
        "translation_status": "non_translated",
        "source_json": "CEDAR.json",
        "source_json_pointer": source_path,
        "source_annotation_id": annotation_id,
        "source_record_0based": debate_index,
        "debate_id": debate_id,
        "competition": competition,
        "topic": topic,
        "topic_type": topic_type,
        "result": result,
        "group_id": group_id,
        "source_start_offset": payload["start_offset"],
        "source_end_offset": payload["end_offset"],
        "source_span_length": len(text),
        "annotation_role": role,
    }
    if role == "claim":
        metadata["stance"] = payload.get("stance", "")
        doc_id = annotation_id
        query_id = "query-" + annotation_id
    else:
        metadata["evidence_type"] = source_key
        doc_id = annotation_id
        query_id = None
    return {
        "annotation_id": annotation_id,
        "doc_id": doc_id,
        "query_id": query_id,
        "text": text,
        "role": role,
        "source_path": source_path,
        "metadata": metadata,
    }, None


def _new_exclusion(exclusions: list[dict[str, Any]], reason: str, **fields: Any) -> None:
    exclusions.append({"reason": reason, **fields})


def assign_protocol_groups(group_ids: Iterable[str], *, seed: int = SEED) -> dict[str, str]:
    """Assign whole topic groups; no topic can cross dev/test/train."""
    ordered = sorted(
        set(group_ids),
        key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest(),
    )
    if len(ordered) < 60:
        raise ValueError(f"CEDAR has fewer than 60 valid topic groups: {len(ordered)}")
    return {
        group: ("dev" if index < 10 else "test" if index < 60 else "train")
        for index, group in enumerate(ordered)
    }


def _parse_records(records: list[Mapping[str, Any]]) -> tuple[list[dict], list[dict], dict, list[dict], dict]:
    documents: dict[str, dict[str, Any]] = {}
    claims_by_debate: dict[int, defaultdict[str, list[dict[str, Any]]]] = {}
    evidence_by_debate: dict[int, defaultdict[str, list[dict[str, Any]]]] = {}
    all_claims: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    counts = Counter(raw_records=len(records))
    topic_by_debate: dict[int, str] = {}
    debate_by_index: dict[int, str] = {}

    for debate_index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            _new_exclusion(exclusions, "invalid_debate_record", debate_index=debate_index)
            counts["invalid_debate_records"] += 1
            continue
        topic = _text(raw.get("topic"))
        competition = _text(raw.get("competition"))
        if topic is None or competition is None:
            _new_exclusion(exclusions, "missing_debate_identity", debate_index=debate_index)
            counts["invalid_debate_records"] += 1
            continue
        topic_by_debate[debate_index] = topic
        debate_id = _debate_id(debate_index, competition)
        debate_by_index[debate_index] = debate_id
        claims_by_debate[debate_index] = defaultdict(list)
        evidence_by_debate[debate_index] = defaultdict(list)
        common = {
            "topic": topic,
            "competition": competition,
            "topic_type": raw.get("topicType"),
            "result": raw.get("result"),
            "debate_id": debate_id,
        }
        claims = raw.get("claims", [])
        if not isinstance(claims, list):
            _new_exclusion(exclusions, "claims_not_list", debate_index=debate_index)
            claims = []
        for claim_index, payload in enumerate(claims):
            if not isinstance(payload, Mapping):
                _new_exclusion(exclusions, "invalid_claim_annotation", debate_index=debate_index, entry_index=claim_index)
                counts["invalid_claim_annotations"] += 1
                continue
            annotation, reason = _record_annotation(
                role="claim", debate_index=debate_index, entry_index=claim_index,
                source_type="claims", source_key=None, payload=payload, **common,
            )
            if annotation is None:
                _new_exclusion(exclusions, reason or "invalid_claim_annotation", debate_index=debate_index, entry_index=claim_index)
                counts["excluded_claim_annotations"] += 1
                continue
            all_claims.append(annotation)
            claims_by_debate[debate_index][annotation["text"]].append(annotation)
            documents[annotation["doc_id"]] = {
                "_id": annotation["doc_id"], "title": topic, "text": annotation["text"],
                "language": LANGUAGE, "metadata": annotation["metadata"],
            }
            counts["valid_claim_annotations"] += 1

        evidences = raw.get("evidences", [])
        if not isinstance(evidences, list):
            _new_exclusion(exclusions, "evidences_not_list", debate_index=debate_index)
            evidences = []
        evidence_index = 0
        for evidence_container_index, container in enumerate(evidences):
            if not isinstance(container, Mapping):
                _new_exclusion(exclusions, "invalid_evidence_container", debate_index=debate_index, entry_index=evidence_container_index)
                counts["invalid_evidence_containers"] += 1
                continue
            for evidence_type, payload in container.items():
                current_index = evidence_index
                evidence_index += 1
                if not isinstance(payload, Mapping):
                    _new_exclusion(exclusions, "invalid_evidence_annotation", debate_index=debate_index, entry_index=current_index, evidence_type=evidence_type)
                    counts["invalid_evidence_annotations"] += 1
                    continue
                annotation, reason = _record_annotation(
                    role="evidence", debate_index=debate_index, entry_index=current_index,
                    source_type="evidences", source_key=str(evidence_type), payload=payload, **common,
                )
                if annotation is None:
                    _new_exclusion(exclusions, reason or "invalid_evidence_annotation", debate_index=debate_index, entry_index=current_index, evidence_type=evidence_type)
                    counts["excluded_evidence_annotations"] += 1
                    continue
                all_evidence.append(annotation)
                evidence_by_debate[debate_index][annotation["text"]].append(annotation)
                documents[annotation["doc_id"]] = {
                    "_id": annotation["doc_id"], "title": topic, "text": annotation["text"],
                    "language": LANGUAGE, "metadata": annotation["metadata"],
                }
                counts["valid_evidence_annotations"] += 1

    relation_rows: list[dict[str, Any]] = []
    raw_relation_labels = Counter()
    for debate_index, raw in enumerate(records):
        if debate_index not in claims_by_debate:
            continue
        relations = raw.get("relations", [])
        if not isinstance(relations, list):
            _new_exclusion(exclusions, "relations_not_list", debate_index=debate_index)
            counts["relations_not_list"] += 1
            continue
        for relation_index, relation in enumerate(relations):
            if not isinstance(relation, Mapping):
                _new_exclusion(exclusions, "invalid_relation_record", debate_index=debate_index, relation_index=relation_index)
                counts["invalid_relations"] += 1
                continue
            label = relation.get("relation")
            raw_relation_labels[str(label)] += 1
            from_text = relation.get("from_text")
            to_text = relation.get("to_text")
            if not isinstance(from_text, str) or not from_text.strip() or not isinstance(to_text, str) or not to_text.strip():
                _new_exclusion(exclusions, "relation_missing_endpoint", debate_index=debate_index, relation_index=relation_index)
                counts["relations_missing_endpoint"] += 1
                continue
            source_roles: list[str] = []
            if from_text in evidence_by_debate[debate_index]:
                source_roles.append("evidence")
            if from_text in claims_by_debate[debate_index]:
                source_roles.append("claim")
            target_roles: list[str] = []
            if to_text in claims_by_debate[debate_index]:
                target_roles.append("claim")
            if to_text in evidence_by_debate[debate_index]:
                target_roles.append("evidence")
            if len(source_roles) != 1 or source_roles[0] not in {"evidence", "claim"} or target_roles != ["claim"]:
                _new_exclusion(
                    exclusions, "relation_role_unknown_or_collision", debate_index=debate_index,
                    relation_index=relation_index, source_roles=source_roles, target_roles=target_roles,
                )
                counts["relations_role_unknown_or_collision"] += 1
                continue
            source_map = evidence_by_debate[debate_index] if source_roles[0] == "evidence" else claims_by_debate[debate_index]
            source_candidates = source_map[from_text]
            target_candidates = claims_by_debate[debate_index][to_text]
            if len(source_candidates) != 1 or len(target_candidates) != 1:
                _new_exclusion(
                    exclusions, "relation_endpoint_text_ambiguous", debate_index=debate_index,
                    relation_index=relation_index, source_candidates=len(source_candidates),
                    target_candidates=len(target_candidates),
                )
                counts["relations_ambiguous_endpoint_text"] += 1
                continue
            if label not in {"support", "against"}:
                _new_exclusion(exclusions, "unknown_relation_label", debate_index=debate_index, relation_index=relation_index, label=label)
                counts["unknown_relation_labels"] += 1
                continue
            source = source_candidates[0]
            target = target_candidates[0]
            relation_rows.append({
                "query_id": target["query_id"],
                "doc_id": source["doc_id"],
                "relation": label,
                "source_role": source["role"],
                "target_role": "claim",
                "debate_index": debate_index,
                "relation_pointer": _pointer(debate_index, "relations", relation_index),
                "source_annotation_id": source["annotation_id"],
                "target_annotation_id": target["annotation_id"],
            })
            counts["explicit_relation_rows"] += 1

    pair_rows: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        pair_rows[(row["query_id"], row["doc_id"])].append(row)
    qrels: dict[str, dict[str, int]] = {}
    evidence_rows: list[dict[str, Any]] = []
    conflicts = []
    for (query_id, doc_id), rows in sorted(pair_rows.items()):
        labels = sorted({row["relation"] for row in rows})
        if len(labels) != 1:
            conflicts.append({
                "query_id": query_id, "doc_id": doc_id, "labels": labels,
                "relation_pointers": sorted(row["relation_pointer"] for row in rows),
            })
            _new_exclusion(exclusions, "conflicting_relation_labels", query_id=query_id, doc_id=doc_id, labels=labels)
            counts["conflicting_relation_pairs"] += 1
            continue
        qrels.setdefault(query_id, {})[doc_id] = 1
        source_roles = sorted({row["source_role"] for row in rows})
        evidence_rows.append({
            "query_id": query_id,
            "doc_id": doc_id,
            "quote": documents[doc_id]["text"],
            "relation": labels[0],
            "relation_row_count": len(rows),
            "relation_pointers": sorted(row["relation_pointer"] for row in rows),
            "source_annotation_ids": sorted({row["source_annotation_id"] for row in rows}),
            "target_annotation_ids": sorted({row["target_annotation_id"] for row in rows}),
            "source_roles": source_roles,
            "annotation_source": "CEDAR native relation annotation",
        })

    query_by_id = {claim["query_id"]: claim for claim in all_claims if claim["query_id"] in qrels}
    # Assign only groups that actually contribute a retained query.  A topic
    # with claims but no uniquely mapped positive relation must not consume one
    # of the protocol's ten dev or fifty test groups.
    protocol = assign_protocol_groups(
        claim["metadata"]["group_id"] for claim in query_by_id.values()
    )
    queries: list[dict[str, Any]] = []
    for query_id in sorted(query_by_id):
        claim = query_by_id[query_id]
        metadata = dict(claim["metadata"])
        metadata.update({
            "source_query_id": query_id,
            "group_id": metadata["group_id"],
            "protocol_group_split": protocol[metadata["group_id"]],
            "query_label_basis": "explicit source relation to this claim",
            "positive_relation_labels": sorted({
                row["relation"]
                for doc_id in qrels[query_id]
                for row in pair_rows[(query_id, doc_id)]
            }),
        })
        queries.append({"_id": query_id, "text": claim["text"], "language": LANGUAGE, "metadata": metadata})

    group_query_counts = Counter(
        query["metadata"]["group_id"] for query in queries
    )
    # The IDs are retained as a deterministic nested eligibility witness; this
    # is not a benchmark pool and is never written to the shared pool tree.
    ordered_docs = sorted(
        documents,
        key=lambda doc_id: hashlib.sha256(f"{SEED}\0{doc_id}".encode()).hexdigest(),
    )
    full1000 = ordered_docs[:1000]
    lite200 = full1000[:200]
    duplicate_counts = Counter(doc["text"] for doc in documents.values())
    duplicate_groups = [count for count in duplicate_counts.values() if count > 1]
    conversion = {
        "status": "adapted_review_required",
        "counts": {
            **dict(counts),
            "documents": len(documents),
            "queries": len(queries),
            "positive_pairs": sum(len(row) for row in qrels.values()),
            "evidence_rows": len(evidence_rows),
            "query_groups": len(group_query_counts),
            "valid_topic_groups": len(protocol),
            "protocol_dev_groups": sum(value == "dev" for value in protocol.values()),
            "protocol_test_groups": sum(value == "test" for value in protocol.values()),
            "protocol_train_groups": sum(value == "train" for value in protocol.values()),
            "exact_duplicate_document_text_groups": len(duplicate_groups),
            "exact_duplicate_document_occurrences_beyond_first": sum(count - 1 for count in duplicate_groups),
        },
        "raw_relation_labels": dict(raw_relation_labels),
        "relation_mapping": {
            "direction": "query=explicit relation.to_text claim; document=explicit relation.from_text evidence or claim",
            "positive_score": 1,
            "support_and_against_both_relevant": True,
            "same_topic_inference": False,
            "role_collisions_rejected": True,
            "ambiguous_text_endpoints_rejected": True,
            "conflicting_relation_labels_rejected": True,
        },
        "protocol_grouping": {
            "group_field": "topic",
            "group_id": "sha256(topic)[:16]",
            "seed": SEED,
            "assignment": "stable sha256(seed\\0group_id), first 10 dev, next 50 test, remaining train",
            "group_query_counts": dict(sorted(group_query_counts.items())),
            "group_splits": dict(sorted(protocol.items())),
            "topic_group_overlap": False,
        },
        "nested_eligibility": {
            "full_size": len(full1000),
            "lite_size": len(lite200),
            "lite_subset_full": lite200 == full1000[:200],
            "full_document_ids": full1000,
            "lite_document_ids": lite200,
            "is_benchmark_pool": False,
        },
        "exclusions": exclusions,
        "conflicting_relation_pairs": conflicts,
        "limitations": [
            "CEDAR relation endpoints are text rather than stable annotation IDs; 75 ambiguous endpoint rows and cross-role collisions are excluded.",
            "The evidence annotation is source-local/candidate-focused in the paper, not an exhaustive judgment of every corpus document.",
            "This is a faithful relation-to-retrieval adaptation, not an official CEDAR retrieval benchmark.",
            "Exact duplicate source annotation texts are retained as occurrence-level documents and trigger pipeline review_required audit.",
            "Source offsets are retained and validated for integer range/length consistency; no reconstructed transcript offsets are used for qrels.",
        ],
        "query_source_ids": sorted(query_by_id),
    }
    return list(documents.values()), queries, qrels, evidence_rows, conversion


def _verify_archive() -> dict[str, Any]:
    if not ARCHIVE.exists():
        download_url(
            ARCHIVE_URL, ARCHIVE, timeout=45, retries=2, max_bytes=30_000_000,
            expected_sha256=ARCHIVE_SHA256, expected_bytes=ARCHIVE_BYTES, overwrite=False,
        )
    archive_sha, archive_bytes = file_digest(ARCHIVE, max_bytes=ARCHIVE_BYTES)
    if (archive_sha, archive_bytes) != (ARCHIVE_SHA256, ARCHIVE_BYTES):
        raise ValueError(f"CEDAR.zip integrity mismatch: {(archive_sha, archive_bytes)}")
    blob_sha = hashlib.sha1(b"blob " + str(archive_bytes).encode() + b"\0" + ARCHIVE.read_bytes()).hexdigest()
    if blob_sha != ARCHIVE_BLOB_SHA1:
        raise ValueError(f"Git blob mismatch: {blob_sha}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError("CEDAR.zip CRC check failed")
        if archive.namelist() != [ARCHIVE_MEMBER]:
            raise ValueError(f"Unexpected CEDAR.zip members: {archive.namelist()}")
        info = archive.getinfo(ARCHIVE_MEMBER)
        if info.file_size != ARCHIVE_MEMBER_BYTES:
            raise ValueError(f"Unexpected CEDAR.json size: {info.file_size}")
        hasher = hashlib.sha256()
        size = 0
        with archive.open(info) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
                size += len(block)
        if (size, hasher.hexdigest()) != (ARCHIVE_MEMBER_BYTES, ARCHIVE_MEMBER_SHA256):
            raise ValueError("CEDAR.json member integrity mismatch")
    return {
        "url": ARCHIVE_URL,
        "revision": REVISION,
        "tree_sha": TREE_SHA,
        "archive_git_blob_sha1": ARCHIVE_BLOB_SHA1,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "member": ARCHIVE_MEMBER,
        "member_bytes": ARCHIVE_MEMBER_BYTES,
        "member_sha256": ARCHIVE_MEMBER_SHA256,
        "zip_crc_checked": True,
    }


def _load_raw_records() -> list[Mapping[str, Any]]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(ARCHIVE_MEMBER) as stream:
            records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("CEDAR.json top-level value is not a list")
    return records


def _write_raw_manifest(archive_info: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    normalized = _require_spec(spec)
    files = []
    for path in sorted(RAW.rglob("*")):
        if path.is_file() and path.name != "source_manifest.json":
            sha, size = file_digest(path)
            files.append({"path": path.relative_to(RAW).as_posix(), "sha256": sha, "bytes": size})
    manifest = {
        "schema_version": "dci-source-snapshot-v1",
        "complete": True,
        "id": SOURCE,
        "connector": "generic_jsonl",
        "input_kind": "retrieval",
        "requested_spec_fingerprint": _spec_fingerprint(normalized),
        "spec_fingerprint": _spec_fingerprint(normalized),
        "requested_spec": normalized,
        "actual_url": ARCHIVE_URL,
        "actual_urls": [ARCHIVE_URL],
        "resolved_revision": REVISION,
        "layout": {
            "documents": ["canonical_input/documents.jsonl"],
            "queries": ["canonical_input/queries.jsonl"],
            "qrels": ["canonical_input/qrels.json"],
            "evidence": ["canonical_input/evidence.jsonl"],
        },
        "archive_pin": dict(archive_info),
        "files": files,
    }
    json_once(RAW / "source_manifest.json", manifest)


def _build_or_verify(spec: Mapping[str, Any]) -> tuple[dict, dict]:
    if RELEASE.exists():
        manifest = verify_release(RELEASE)
        bundle = load_bundle(RELEASE, verify=False)
    else:
        build_source(spec, BASE / "raw_data", RELEASE, snapshot=RAW, policy=POLICY)
        manifest = verify_release(RELEASE)
        bundle = load_bundle(RELEASE, verify=False)
    return manifest, bundle


def _reviewed_bundle(bundle: Mapping[str, Any], parent_audit: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    """Remove reviewed ambiguous queries and collapse exact text occurrences.

    The source release keeps every occurrence-level annotation.  This derived
    view removes queries whose positive text has an unlabelled exact
    occurrence, then aliases identical document text to one canonical ID.  A
    canonical document carries every original ID and source pointer, so the
    collapse cannot silently erase relation provenance.
    """
    review_required = parent_audit.get("review_required", [])
    ambiguous_query_ids = {
        row.get("query_id")
        for row in review_required
        if isinstance(row, Mapping)
        and row.get("code") == "equivalent_unlabelled_document"
        and isinstance(row.get("query_id"), str)
    }
    documents_by_text: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.get("documents", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("_id"), str) or not isinstance(row.get("text"), str):
            raise ValueError("parent release contains a malformed document")
        documents_by_text[row["text"]].append(copy.deepcopy(dict(row)))

    canonical_documents: list[dict[str, Any]] = []
    document_aliases: dict[str, list[dict[str, Any]]] = {}
    document_id_map: dict[str, str] = {}
    for text, rows in sorted(documents_by_text.items(), key=lambda pair: min(row["_id"] for row in pair[1])):
        rows.sort(key=lambda row: row["_id"])
        canonical_id = rows[0]["_id"]
        aliases = []
        for row in rows:
            metadata = row.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError(f"document metadata is malformed: {row['_id']}")
            aliases.append({
                "document_id": row["_id"],
                "source_annotation_id": metadata.get("source_annotation_id"),
                "source_json_pointer": metadata.get("source_json_pointer"),
                "annotation_role": metadata.get("annotation_role"),
                "debate_id": metadata.get("debate_id"),
                "topic": metadata.get("topic"),
            })
            document_id_map[row["_id"]] = canonical_id
        canonical = rows[0]
        canonical_metadata = dict(canonical.get("metadata", {}))
        canonical_metadata.update({
            "exact_text_collapse": True,
            "canonical_document_id": canonical_id,
            "source_occurrence_count": len(rows),
            "source_aliases": aliases,
        })
        canonical["metadata"] = canonical_metadata
        canonical_documents.append(canonical)
        document_aliases[canonical_id] = aliases

    # Re-map source relation rows after the exact-text collapse.  A newly
    # introduced support/against conflict would make the query ambiguous and
    # is therefore excluded as a whole rather than assigning a guessed label.
    relation_labels: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in bundle.get("evidence", []):
        if not isinstance(row, Mapping):
            continue
        query_id = row.get("query_id")
        original_document_id = row.get("doc_id")
        canonical_id = document_id_map.get(original_document_id)
        if query_id in ambiguous_query_ids or canonical_id is None:
            continue
        label = row.get("relation")
        if isinstance(query_id, str) and isinstance(label, str):
            relation_labels[(query_id, canonical_id)].add(label)
    post_collapse_conflict_queries = {
        query_id
        for (query_id, _), labels in relation_labels.items()
        if len(labels) > 1
    }
    excluded_query_ids = ambiguous_query_ids | post_collapse_conflict_queries

    candidates_by_group: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.get("queries", []):
        if not isinstance(row, Mapping) or row.get("_id") in excluded_query_ids:
            continue
        metadata = row.get("metadata", {})
        group_id = metadata.get("group_id") if isinstance(metadata, Mapping) else None
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"reviewed query has no source group_id: {row.get('_id')!r}")
        candidates_by_group[group_id].append(copy.deepcopy(dict(row)))

    # Keep one deterministic query per topic group.  This preserves topic
    # leakage isolation while making the v2 sampler's exact individual
    # dev=10/test=50 quotas feasible.  The discarded non-ambiguous queries
    # remain auditable through the parent release and the receipt counts.
    selected_queries: list[dict[str, Any]] = []
    for group_id, rows in sorted(candidates_by_group.items()):
        selected = min(rows, key=lambda row: row["_id"])
        selected["group_id"] = group_id
        selected_metadata = dict(selected.get("metadata", {}))
        selected_metadata.update({
            "group_id": group_id,
            "source_group_id": group_id,
            "reviewed_query_selection": "one deterministic query per non-ambiguous topic group",
            "sampling_authority": "sample_global_query_splits(seed=42,dev_size=10,test_size=50)",
        })
        source_query_document_id = selected_metadata.get("source_query_document_id")
        if not isinstance(source_query_document_id, str):
            source_query_document_id = selected["_id"].removeprefix("query-")
        canonical_query_document_id = document_id_map.get(source_query_document_id)
        if canonical_query_document_id is None:
            raise ValueError(f"query source document is absent from reviewed documents: {selected['_id']}")
        selected_metadata.update({
            "source_query_document_id": source_query_document_id,
            "canonical_query_document_id": canonical_query_document_id,
            "self_match_document_ids": [canonical_query_document_id],
            "self_match_policy": "remove canonical_query_document_id from retrieval ranks before pool export",
        })
        selected["metadata"] = selected_metadata
        selected_queries.append(selected)
    selected_queries.sort(key=lambda row: row["_id"])
    selected_query_ids = {row["_id"] for row in selected_queries}

    reviewed_qrels: dict[str, dict[str, float]] = {}
    for query in selected_queries:
        query_id = query["_id"]
        source_row = bundle.get("qrels", {}).get(query_id, {})
        if not isinstance(source_row, Mapping):
            raise ValueError(f"missing qrels row in parent release: {query_id}")
        remapped = {}
        for original_document_id, score in source_row.items():
            canonical_id = document_id_map.get(original_document_id)
            if canonical_id is None:
                raise ValueError(f"qrels references unknown parent document: {original_document_id}")
            remapped[canonical_id] = max(float(score), float(remapped.get(canonical_id, 0)))
        if not remapped:
            raise ValueError(f"reviewed query has no positive qrels: {query_id}")
        reviewed_qrels[query_id] = remapped

    # Aggregate evidence rows by the remapped pair while retaining every
    # original document ID and relation pointer that contributed to it.
    evidence_by_pair: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in bundle.get("evidence", []):
        if not isinstance(row, Mapping):
            continue
        query_id = row.get("query_id")
        original_document_id = row.get("doc_id")
        canonical_id = document_id_map.get(original_document_id)
        if query_id in selected_query_ids and canonical_id in reviewed_qrels.get(query_id, {}):
            evidence_by_pair[(query_id, canonical_id)].append(row)
    reviewed_evidence = []
    canonical_text_by_id = {row["_id"]: row["text"] for row in canonical_documents}
    for query_id, doc_id in sorted((query_id, doc_id) for query_id in reviewed_qrels for doc_id in reviewed_qrels[query_id]):
        rows = evidence_by_pair.get((query_id, doc_id), [])
        if not rows:
            raise ValueError(f"reviewed qrels pair has no source evidence row: {query_id}/{doc_id}")
        labels = sorted({row.get("relation") for row in rows if isinstance(row.get("relation"), str)})
        if len(labels) != 1:
            raise ValueError(f"reviewed pair has conflicting relation labels: {query_id}/{doc_id}")
        reviewed_evidence.append({
            "query_id": query_id,
            "doc_id": doc_id,
            "quote": canonical_text_by_id[doc_id],
            "relation": labels[0],
            "relation_labels": labels,
            "original_evidence_row_count": len(rows),
            "original_document_ids": sorted({str(row["doc_id"]) for row in rows}),
            "source_relation_pointers": sorted({pointer for row in rows for pointer in row.get("relation_pointers", [])}),
            "source_annotation_ids": sorted({annotation for row in rows for annotation in row.get("source_annotation_ids", [])}),
            "target_annotation_ids": sorted({annotation for row in rows for annotation in row.get("target_annotation_ids", [])}),
            "annotation_source": "CEDAR native relation annotation; exact-text aliases retained",
        })

    reviewed = copy.deepcopy(dict(bundle))
    reviewed["documents"] = sorted(canonical_documents, key=lambda row: row["_id"])
    reviewed["queries"] = selected_queries
    reviewed["qrels"] = reviewed_qrels
    reviewed["evidence"] = reviewed_evidence
    reviewed["judgments"] = []
    provenance = dict(reviewed.get("provenance", {}))
    provenance.update({
        "derived_from_release": str(RELEASE),
        "reviewed_excluded_query_ids": sorted(excluded_query_ids),
        "reviewed_ambiguous_query_count": len(ambiguous_query_ids),
        "reviewed_post_collapse_conflict_query_count": len(post_collapse_conflict_queries),
        "reviewed_source_query_count_before_group_reduction": len(bundle.get("queries", [])) - len(excluded_query_ids),
        "reviewed_source_query_count_after_group_reduction": len(selected_queries),
        "reviewed_exact_text_document_alias_count": sum(len(aliases) - 1 for aliases in document_aliases.values()),
        "reviewed_exact_text_document_collapse": "Exact decoded source text only; every original document ID and source pointer is retained in metadata and alias manifest.",
        "reviewed_sampling_group": "topic group_id; one query per group; sample_global_query_splits is authoritative for dev/test",
        "ranked_pool_status": "not_built",
    })
    reviewed["provenance"] = provenance
    report = {
        "ambiguous_query_ids": sorted(ambiguous_query_ids),
        "post_collapse_conflict_query_ids": sorted(post_collapse_conflict_queries),
        "selected_query_ids": sorted(selected_query_ids),
        "document_aliases": document_aliases,
        "counts": {
            "parent_documents": len(bundle.get("documents", [])),
            "reviewed_documents": len(reviewed["documents"]),
            "collapsed_alias_occurrences": sum(len(aliases) - 1 for aliases in document_aliases.values()),
            "parent_queries": len(bundle.get("queries", [])),
            "ambiguous_queries_excluded": len(ambiguous_query_ids),
            "post_collapse_conflict_queries_excluded": len(post_collapse_conflict_queries),
            "non_ambiguous_queries_before_group_reduction": len(bundle.get("queries", [])) - len(excluded_query_ids),
            "reviewed_queries": len(selected_queries),
            "reviewed_query_groups": len(candidates_by_group),
            "reviewed_positive_pairs": sum(len(row) for row in reviewed_qrels.values()),
            "reviewed_evidence_rows": len(reviewed_evidence),
        },
        "grouping": {
            "group_field": "topic",
            "group_id_top_level_flattened": True,
            "one_query_per_group": True,
            "sample_sizes": {"dev": 10, "test": 50, "smoke": 1},
            "seed": 42,
        },
    }
    return reviewed, report, document_aliases


def _verify_reviewed_sampling(bundle: Mapping[str, Any]) -> dict[str, Any]:
    from dci_bench.data.benchmark_v2 import sample_global_query_splits

    rows = []
    for row in bundle.get("queries", []):
        flattened = dict(row)
        metadata = row.get("metadata", {})
        flattened["group_id"] = metadata.get("group_id")
        rows.append(flattened)
    result = sample_global_query_splits(
        {SOURCE: rows}, dev_size=10, test_size=50, smoke_size=1, seed=42,
    )[SOURCE]
    return {
        "counts": {split: len(ids) for split, ids in result.items()},
        "dev_query_ids": result["dev"],
        "test_query_ids": result["test"],
        "smoke_query_ids": result["smoke"],
        "group_leakage_checked": True,
        "metadata_group_id_flattened": True,
    }


def _verify_reviewed_query_identity(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that every query can identify its own canonical document.

    Older immutable reviewed bytes carry the target claim's
    ``source_annotation_id`` and the alias manifest, while newly derived
    bundles additionally carry explicit canonical/self-match fields.  Accept
    both representations without rewriting an existing release.
    """
    canonical_by_annotation: dict[str, str] = {}
    for document in bundle.get("documents", []):
        if not isinstance(document, Mapping) or not isinstance(document.get("_id"), str):
            raise ValueError("reviewed document is malformed while checking query identity")
        metadata = document.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"reviewed document metadata is malformed: {document['_id']}")
        aliases = metadata.get("source_aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(f"reviewed document aliases are malformed: {document['_id']}")
        source_ids = [metadata.get("source_annotation_id")]
        source_ids.extend(
            alias.get("source_annotation_id")
            for alias in aliases
            if isinstance(alias, Mapping)
        )
        for source_id in source_ids:
            if not isinstance(source_id, str) or not source_id:
                continue
            previous = canonical_by_annotation.setdefault(source_id, document["_id"])
            if previous != document["_id"]:
                raise ValueError(f"source annotation maps to multiple reviewed documents: {source_id}")

    checked = 0
    for query in bundle.get("queries", []):
        if not isinstance(query, Mapping):
            raise ValueError("reviewed query is malformed while checking self-match identity")
        metadata = query.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"reviewed query metadata is malformed: {query.get('_id')!r}")
        source_annotation_id = metadata.get("source_annotation_id")
        if not isinstance(source_annotation_id, str) or not source_annotation_id:
            raise ValueError(f"reviewed query has no target source_annotation_id: {query.get('_id')!r}")
        canonical_id = canonical_by_annotation.get(source_annotation_id)
        if canonical_id is None:
            raise ValueError(f"reviewed query target annotation is absent: {source_annotation_id}")
        explicit_id = metadata.get("canonical_query_document_id")
        if explicit_id is not None and explicit_id != canonical_id:
            raise ValueError(f"reviewed query canonical self-match disagrees: {query.get('_id')!r}")
        explicit_self_matches = metadata.get("self_match_document_ids")
        if explicit_self_matches is not None and explicit_self_matches != [canonical_id]:
            raise ValueError(f"reviewed query self-match list disagrees: {query.get('_id')!r}")
        checked += 1
    return {
        "queries_checked": checked,
        "source_annotation_identity_retained": True,
        "canonical_self_match_resolvable": True,
    }


def _build_or_verify_reviewed(parent_bundle: Mapping[str, Any], parent_manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_audit = json.loads((RELEASE / "audit.json").read_text())
    reviewed, report, aliases = _reviewed_bundle(parent_bundle, parent_audit)
    sampling = _verify_reviewed_sampling(reviewed)
    report["sampling_verification"] = sampling
    json_once(REVIEWED_ALIASES, aliases)
    if REVIEWED_RELEASE.exists():
        manifest = verify_release(REVIEWED_RELEASE)
        bundle = load_bundle(REVIEWED_RELEASE, verify=False)
    else:
        manifest = publish_bundle(
            reviewed,
            REVIEWED_RELEASE,
            specification=SPEC,
            raw_snapshot=RAW,
            policy=POLICY,
            parent_release=RELEASE,
        )
        manifest = verify_release(REVIEWED_RELEASE)
        bundle = load_bundle(REVIEWED_RELEASE, verify=False)
    _verify_reviewed_query_identity(bundle)
    if manifest.get("state") not in {"labels_ready", "review_required"}:
        raise ValueError(f"unexpected reviewed release state: {manifest.get('state')}")
    if len(bundle["documents"]) < 1000 or len(bundle["queries"]) < 60:
        raise ValueError("reviewed release is below protocol floors")
    if manifest["num_documents"] != report["counts"]["reviewed_documents"] or manifest["num_queries"] != report["counts"]["reviewed_queries"]:
        raise ValueError("reviewed release counts differ from reviewed adapter")
    receipt = {
        "id": SOURCE,
        "derived_id": SOURCE + "_reviewed",
        "status": "reviewed_source_relation_adapter",
        "ready_for_benchmark": False,
        "release_state": manifest["state"],
        "release_path": str(REVIEWED_RELEASE),
        "release_manifest": str(REVIEWED_RELEASE / "manifest.json"),
        "release_complete": str(REVIEWED_RELEASE / "_COMPLETE.json"),
        "parent_release": str(RELEASE),
        "parent_bundle_hash": parent_manifest["bundle_hash"],
        "review_report": report,
        "alias_manifest": str(REVIEWED_ALIASES),
        "ranked_pool_status": "not_built; Lite/Full eligibility is not a ranked pool",
        "limitations": [
            "All 4,180 parent audit-ambiguous query IDs are excluded as whole queries.",
            "Non-ambiguous queries are reduced to one deterministic query per topic group so sample_global_query_splits can form exact individual dev=10/test=50 without topic leakage.",
            "Exact duplicate documents are collapsed by decoded text only; all original IDs, source annotation IDs and source JSON pointers remain in document metadata and alias_manifest.",
            "CEDAR relations are candidate/local annotations; the derived release does not claim exhaustive negative judgments.",
            "No ranked retrieval pools were built; nested Lite/Full eligibility IDs are not evidence of ranked pools.",
        ],
        "checks": [
            "Parent review_required query IDs excluded as complete query units",
            "Exact-text document aliases collapsed with source provenance retained",
            "Derived release verify_release/load_bundle passed",
            "sample_global_query_splits passed on top-level flattened group_id with dev=10/test=50/smoke=1",
            "No GPU, ranking or shared benchmark code was run or changed",
        ],
    }
    json_once(REVIEWED_RECEIPT, receipt)
    return receipt


def acquire() -> dict[str, Any]:
    archive_info = _verify_archive()
    pin = {
        "repository": "VelikayaScarlet/CEDAR",
        "commit": REVISION,
        "tree_sha": TREE_SHA,
        "archive_blob_sha1": ARCHIVE_BLOB_SHA1,
        "archive_url": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "license": LICENSE,
        "paper_url": "https://aclanthology.org/2026.acl-long.238.pdf",
        "archive_verification": archive_info,
    }
    json_once(RAW / "source_inputs/github_pin.json", pin)
    records = _load_raw_records()
    documents, queries, qrels, evidence, conversion = _parse_records(records)
    if len(documents) < 1000 or len(queries) < 60:
        raise ValueError(f"CEDAR adaptation is below protocol floor: docs={len(documents)}, queries={len(queries)}")
    write_jsonl_once(CANONICAL / "documents.jsonl", sorted(documents, key=lambda row: row["_id"]))
    write_jsonl_once(CANONICAL / "queries.jsonl", sorted(queries, key=lambda row: row["_id"]))
    json_once(CANONICAL / "qrels.json", {query_id: qrels[query_id] for query_id in sorted(qrels)})
    write_jsonl_once(CANONICAL / "evidence.jsonl", sorted(evidence, key=lambda row: (row["query_id"], row["doc_id"])))
    json_once(RAW / "conversion_manifest.json", conversion)
    _write_raw_manifest(archive_info, SPEC)
    manifest, bundle = _build_or_verify(SPEC)
    document_texts = {row["_id"]: row["text"] for row in documents}
    query_texts = {row["_id"]: row["text"] for row in queries}
    assert {row["_id"]: row["text"] for row in bundle["documents"]} == document_texts
    assert {row["_id"]: row["text"] for row in bundle["queries"]} == query_texts
    assert bundle["qrels"] == qrels
    assert len(bundle["documents"]) >= 1000 and len(bundle["queries"]) >= 60
    assert conversion["nested_eligibility"]["lite_document_ids"] == conversion["nested_eligibility"]["full_document_ids"][:200]
    receipt = {
        "id": SOURCE,
        "status": "adapted_review_required" if manifest["state"] == "review_required" else "adapted_source_relation_release",
        "ready_for_benchmark": manifest["state"] == "labels_ready",
        "domain": "argumentation",
        "language": LANGUAGE,
        "native_language": "Chinese Mandarin (source declares native/manual annotation)",
        "source_family": "CEDAR",
        "source_url": "https://github.com/VelikayaScarlet/CEDAR",
        "paper_url": "https://aclanthology.org/2026.acl-long.238.pdf",
        "revision": REVISION,
        "tree_sha": TREE_SHA,
        "archive_url": ARCHIVE_URL,
        "archive_git_blob_sha1": ARCHIVE_BLOB_SHA1,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "license": LICENSE,
        "counts": conversion["counts"],
        "release_path": str(RELEASE),
        "release_state": manifest["state"],
        "release_manifest": str(RELEASE / "manifest.json"),
        "release_complete": str(RELEASE / "_COMPLETE.json"),
        "raw_path": str(RAW),
        "source_manifest": str(RAW / "source_manifest.json"),
        "conversion_manifest": str(RAW / "conversion_manifest.json"),
        "group_strategy": conversion["protocol_grouping"],
        "nested_eligibility": conversion["nested_eligibility"],
        "label_status": {
            "qrels": "explicit source relations only",
            "same_topic_inference": False,
            "review_reason": conversion["limitations"],
        },
        "checks": [
            "Pinned Git commit/tree/blob, archive size and SHA256 verified",
            "ZIP CRC and CEDAR.json member size/SHA256 verified",
            "Original archive retained under source_inputs without modification",
            "Every published document/query preserves source annotation path and source text",
            "Every qrel comes from a unique same-debate explicit relation edge",
            "Role collisions, ambiguous duplicate endpoints, unknown endpoints and conflicting labels audited/excluded",
            "Pipeline load_bundle/verify_release and document/query/qrel text identity checks passed",
            "Lite 200 IDs are a prefix of deterministic Full 1000 eligibility IDs",
        ],
        "unknown_or_unresolved": [
            "CEDAR relation endpoints are text-only, so excluded ambiguous rows cannot be recovered without an external stable-ID annotation layer.",
            "The paper describes candidate evidence annotation near claims; completeness over every corpus document is not claimed.",
            "This adapter is derived retrieval and requires the release_state shown above before benchmark export.",
        ],
    }
    json_once(RESEARCH / "cedar_20261031.json", receipt)
    reviewed_receipt = _build_or_verify_reviewed(bundle, manifest)
    return {"base": receipt, "reviewed": reviewed_receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="verify an existing raw/release artifact without creating it")
    args = parser.parse_args()
    if args.check_only:
        info = _verify_archive()
        manifest = verify_release(RELEASE)
        base_bundle = load_bundle(RELEASE, verify=False)
        reviewed_manifest = verify_release(REVIEWED_RELEASE)
        reviewed_bundle = load_bundle(REVIEWED_RELEASE, verify=False)
        print(json.dumps({
            "archive": info,
            "release": manifest,
            "reviewed_release": reviewed_manifest,
            "reviewed_sampling": _verify_reviewed_sampling(reviewed_bundle),
        }, ensure_ascii=False))
        return
    receipt = acquire()
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
