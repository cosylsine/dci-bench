#!/usr/bin/env python3
"""Prepare the native English PrivacyQA legal retrieval source.

The source is the frozen PrivacyQA test set (human questions and expert
sentence annotations) plus the train policies as corpus-only material.  The
adapter deliberately keeps the source sentence text byte-for-byte after a
small deterministic policy identity header.  That header makes otherwise
identical sentences from different policies distinct without changing the
retrieval evidence.

Normal execution publishes an immutable release and a DCI document export.
``--check-only`` reads and verifies those existing artifacts and never writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.expansion import verify_document_task
from dci_bench.data.benchmark_v2 import _prepare_query_eligibility
from dci_bench.data.pipeline.core import (
    export_dci,
    file_digest,
    load_bundle,
    publish_bundle,
    verify_release,
)


SOURCE_ID = "gapfill_privacyqa_legal_en"
DATASET = "AbhilashaRavichander/PrivacyQA_EMNLP"
REVISION = "5178ff3eaf61cbdaea03d85ff23b676b3fea451f"
DEFAULT_DATA_ROOT = Path(
    "/mnt/afs2/202611/embedding_models/dci-bench/eval_data/"
    "benchmark_legal_pos10_20261116_v1"
)

TRAIN_FILENAME = "policy_train_data.csv"
TEST_FILENAME = "policy_test_data.csv"
TRAIN_SHA256 = "81aa674452fc822a49d7d96aead91fc85199ec99b6a56c7a862bb9deb02f004a"
TEST_SHA256 = "3822409246bfc733cdb17bb7bea85a92d906d76dae48763350c3306ee24a924a"
TRAIN_BYTES = 62602240
TEST_BYTES = 22409457

REPO_URL = "https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP"
TRAIN_URL = (
    f"https://raw.githubusercontent.com/AbhilashaRavichander/PrivacyQA_EMNLP/"
    f"{REVISION}/data/{TRAIN_FILENAME}"
)
TEST_URL = (
    f"https://raw.githubusercontent.com/AbhilashaRavichander/PrivacyQA_EMNLP/"
    f"{REVISION}/data/{TEST_FILENAME}"
)
PAPER_URL = "https://aclanthology.org/D19-1500/"
LICENSE_URL = f"{REPO_URL}/blob/{REVISION}/LICENSE"

EXPECTED_TRAIN_ROWS = 185200
EXPECTED_TEST_ROWS = 62150
EXPECTED_DOCUMENTS = 4657
EXPECTED_TEST_QUERIES = 400
EXPECTED_ELIGIBLE_QUERIES = 167
EXPECTED_POSITIVE_PAIRS = 990

SPECIFICATION = {
    "id": SOURCE_ID,
    "connector": "privacyqa_pos10",
    "input_kind": "retrieval",
    "domain": "legal",
    "language": "en",
    "genre": "privacy_policies",
    "revision": REVISION,
    "options": {"dataset": DATASET},
}


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _policy_title(raw_doc_id: str) -> str:
    """Return the product title while retaining the exact source DocID elsewhere."""

    title = raw_doc_id.rsplit(" _", 1)[0]
    return title.strip() or raw_doc_id.strip() or raw_doc_id


def _document_identity(raw_doc_id: str, segment: str) -> tuple[str, str]:
    """Return the exact source key and a stable task-local document ID."""

    key = f"{raw_doc_id}\0{segment}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return key, f"privacyqa-doc-{digest[:32]}"


def _query_identity(raw_query_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(raw_query_id.encode("utf-8")).hexdigest()
    query_id = f"privacyqa-query-{digest[:32]}"
    group = f"privacyqa-query:{raw_query_id}"
    return query_id, group


def _required_fields(split: str) -> set[str]:
    common = {"Folder", "DocID", "QueryID", "SentID", "Split", "Query", "Segment"}
    if split == "test":
        return common | {"Any_Relevant", "Ann1", "Ann2", "Ann3", "Ann4", "Ann5", "Ann6"}
    return common | {"Label"}


def _iter_rows(path: Path, split: str) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = sorted(_required_fields(split) - fields)
        if missing:
            raise ValueError(f"{path}: missing PrivacyQA fields: {missing}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{line_number}: malformed tabular row")
            normalized = {key: value for key, value in row.items() if key is not None}
            for field in ("DocID", "QueryID", "SentID", "Split", "Query", "Segment"):
                if not isinstance(normalized.get(field), str) or not normalized[field]:
                    raise ValueError(f"{path}:{line_number}: empty {field}")
            if normalized["Split"].strip().casefold() != split:
                raise ValueError(
                    f"{path}:{line_number}: Split={normalized['Split']!r}, expected {split!r}"
                )
            yield normalized


def _verify_raw_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing PrivacyQA raw file: {path}")
    size, digest = _sha256_file(path)
    if size != expected_bytes or digest != expected_sha256:
        raise ValueError(
            f"Changed PrivacyQA raw file {path}: bytes={size} sha256={digest}; "
            f"expected bytes={expected_bytes} sha256={expected_sha256}"
        )
    return {"path": path.name, "bytes": size, "sha256": digest}


def _source_paths(data_root: Path) -> tuple[Path, Path, Path, Path]:
    raw = data_root / "raw" / "privacyqa"
    return (
        raw / TRAIN_FILENAME,
        raw / TEST_FILENAME,
        data_root / "releases" / SOURCE_ID,
        data_root / "run" / "source",
    )


def _source_doc_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "source_doc_id": record["raw_doc_id"],
        "policy_title": record["policy_title"],
        "source_segment": record["segment"],
        "source_segment_sha256": hashlib.sha256(record["segment"].encode("utf-8")).hexdigest(),
        "source_segment_chars": len(record["segment"]),
        "source_segment_bytes": len(record["segment"].encode("utf-8")),
        "source_sent_ids": sorted(record["sent_ids"]),
        "source_query_ids": sorted(record["query_ids"]),
        "source_splits": sorted(record["splits"]),
        "annotation_source": "human",
        "translation_status": "native",
        "native": True,
        "verified_language": "en",
        "domain": "legal",
        "genre": "privacy_policies",
        "segment_start_char": record["segment_start_char"],
        "segment_end_char": record["segment_end_char"],
    }


def _add_source_row(
    docs: dict[tuple[str, str], dict[str, Any]],
    row: dict[str, str],
    split: str,
) -> tuple[str, str]:
    raw_doc_id = row["DocID"]
    segment = row["Segment"]
    if not segment:
        raise ValueError(f"PrivacyQA {split} row has an empty Segment: {row['SentID']}")
    key = (raw_doc_id, segment)
    if key not in docs:
        _, document_id = _document_identity(raw_doc_id, segment)
        title = _policy_title(raw_doc_id)
        header = f"Privacy policy: {title}\n\n"
        docs[key] = {
            "_id": document_id,
            "raw_doc_id": raw_doc_id,
            "policy_title": title,
            "segment": segment,
            "header": header,
            "segment_start_char": len(header),
            "segment_end_char": len(header) + len(segment),
            "sent_ids": set(),
            "query_ids": set(),
            "splits": set(),
        }
    record = docs[key]
    record["sent_ids"].add(row["SentID"])
    record["query_ids"].add(row["QueryID"])
    record["splits"].add(split)
    return key


def _new_test_query(row: dict[str, str]) -> dict[str, Any]:
    return {
        "raw_query_id": row["QueryID"],
        "raw_doc_id": row["DocID"],
        "raw_query": row["Query"],
        "rows": [],
    }


def _judgment_row(
    query_id: str,
    doc_id: str,
    row: dict[str, str],
) -> dict[str, Any]:
    relevant = row.get("Any_Relevant", "").strip().casefold() == "relevant"
    annotations = {f"Ann{i}": row.get(f"Ann{i}", "") for i in range(1, 7)}
    return {
        "query_id": query_id,
        "doc_id": doc_id,
        "score": 1 if relevant else 0,
        "status": "judged",
        "label": 1 if relevant else 0,
        "annotation_source": "human",
        "source": "PrivacyQA expert annotation",
        "source_split": "test",
        "raw_query_id": row["QueryID"],
        "raw_doc_id": row["DocID"],
        "raw_sent_id": row["SentID"],
        "raw_query": row["Query"],
        "raw_segment": row["Segment"],
        "any_relevant": row.get("Any_Relevant", ""),
        "annotations": annotations,
    }


def _build_bundle(
    train_path: Path,
    test_path: Path,
    *,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_files = []
    if verify_hashes:
        raw_files.append(_verify_raw_file(train_path, TRAIN_BYTES, TRAIN_SHA256))
        raw_files.append(_verify_raw_file(test_path, TEST_BYTES, TEST_SHA256))

    docs: dict[tuple[str, str], dict[str, Any]] = {}
    train_rows = 0
    for row in _iter_rows(train_path, "train"):
        _add_source_row(docs, row, "train")
        train_rows += 1

    test_rows = 0
    test_queries: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in _iter_rows(test_path, "test"):
        key = _add_source_row(docs, row, "test")
        query = test_queries.setdefault(row["QueryID"], _new_test_query(row))
        if query["raw_doc_id"] != row["DocID"] or query["raw_query"] != row["Query"]:
            raise ValueError(f"PrivacyQA QueryID has multiple policy/question values: {row['QueryID']}")
        query["rows"].append((row, key))
        test_rows += 1

    if train_rows != EXPECTED_TRAIN_ROWS or test_rows != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Unexpected PrivacyQA row counts: train={train_rows}, test={test_rows}; "
            f"expected train={EXPECTED_TRAIN_ROWS}, test={EXPECTED_TEST_ROWS}"
        )
    if len(test_queries) != EXPECTED_TEST_QUERIES:
        raise ValueError(f"Unexpected PrivacyQA test query count: {len(test_queries)}")

    documents: list[dict[str, Any]] = []
    doc_by_key: dict[tuple[str, str], str] = {}
    for key, record in sorted(docs.items(), key=lambda pair: pair[1]["_id"]):
        document = {
            "_id": record["_id"],
            "title": f"{record['policy_title']} privacy policy ({record['raw_doc_id']})",
            "text": record["header"] + record["segment"],
            "language": "en",
            "metadata": _source_doc_metadata(record),
        }
        documents.append(document)
        doc_by_key[key] = record["_id"]

    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, int]] = {}
    evidence: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    query_audit: list[dict[str, Any]] = []
    eligible_count = 0
    positive_pairs = 0
    for raw_query_id, query in test_queries.items():
        query_id, group_id = _query_identity(raw_query_id)
        positive_doc_ids: set[str] = set()
        positive_rows: list[tuple[dict[str, str], str]] = []
        for row, key in query["rows"]:
            doc_id = doc_by_key[key]
            if row.get("Any_Relevant", "").strip().casefold() == "relevant":
                positive_doc_ids.add(doc_id)
                positive_rows.append((row, doc_id))

        positive_count = len(positive_doc_ids)
        eligible = 1 <= positive_count <= 10
        reason = None if eligible else (
            "no_positive_annotations" if positive_count == 0 else "positive_count_over_10"
        )
        query_audit.append({
            "raw_query_id": raw_query_id,
            "raw_doc_id": query["raw_doc_id"],
            "raw_query": query["raw_query"],
            "row_count": len(query["rows"]),
            "positive_row_count": len(positive_rows),
            "positive_document_count": positive_count,
            "eligible_1_to_10": eligible,
            "exclusion_reason": reason,
            "published_query_id": query_id if eligible else None,
        })
        if not eligible:
            continue

        eligible_count += 1
        positive_pairs += positive_count
        title = _policy_title(query["raw_doc_id"])
        query_text = (
            f"Regarding the {title} privacy policy: " + query["raw_query"]
        )
        metadata = {
            "source_id": SOURCE_ID,
            "source_dataset": DATASET,
            "source_revision": REVISION,
            "source_split": "test",
            "raw_query_id": raw_query_id,
            "raw_doc_id": query["raw_doc_id"],
            "original_query": query["raw_query"],
            "policy_title": title,
            "group_id": group_id,
            "positive_row_count": len(positive_rows),
            "positive_document_count": positive_count,
            "source_row_count": len(query["rows"]),
            "annotation_source": "human",
            "query_text_policy": "fixed policy identity prefix; raw Query copied after prefix",
        }
        queries.append({"_id": query_id, "text": query_text, "language": "en", "metadata": metadata})
        qrels[query_id] = {doc_id: 1 for doc_id in sorted(positive_doc_ids)}

        seen_evidence: set[str] = set()
        for row, doc_id in positive_rows:
            judgments.append(_judgment_row(query_id, doc_id, row))
            if doc_id in seen_evidence:
                continue
            seen_evidence.add(doc_id)
            document = docs[(row["DocID"], row["Segment"])]
            evidence.append({
                "query_id": query_id,
                "doc_id": doc_id,
                "quote": row["Segment"],
                "start": document["segment_start_char"],
                "end": document["segment_end_char"],
                "source_split": "test",
                "raw_query_id": raw_query_id,
                "raw_doc_id": row["DocID"],
                "raw_sent_id": row["SentID"],
                "annotation_source": "human",
            })

        for row, key in query["rows"]:
            # Preserve every annotation row for retained queries, including
            # irrelevant sentences, without changing the qrel definition.
            if row.get("Any_Relevant", "").strip().casefold() != "relevant":
                judgments.append(_judgment_row(query_id, doc_by_key[key], row))

    if len(docs) != EXPECTED_DOCUMENTS:
        raise ValueError(f"Unexpected PrivacyQA canonical document count: {len(docs)}")
    if eligible_count != EXPECTED_ELIGIBLE_QUERIES:
        raise ValueError(f"Unexpected PrivacyQA eligible query count: {eligible_count}")
    if positive_pairs != EXPECTED_POSITIVE_PAIRS:
        raise ValueError(f"Unexpected PrivacyQA positive pair count: {positive_pairs}")

    source = {
        "dataset": DATASET,
        "revision": REVISION,
        "resolved_revision": REVISION,
        "connector": "privacyqa_pos10",
    }
    provenance = {
        "source_id": SOURCE_ID,
        "domain": "legal",
        "language": "en",
        "native_language": "en",
        "translation_status": "native",
        "genre": "privacy_policies",
        "annotation_source": "human",
        "label_source": "human",
        "qrels_label": "Any_Relevant human annotations",
        "query_text_policy": "fixed policy identity prefix; original test Query retained",
        "train_usage": "corpus_only; no train queries published",
        "canonical_document_key": "exact (DocID, Segment), across train and test",
        "canonical_document_id": "privacyqa-doc- plus first 32 hex SHA256(DocID NUL Segment)",
        "policy_identity_header": "Privacy policy title prepended; exact source DocID remains in metadata; Segment unchanged",
        "test_query_filter": "whole-query keep iff canonical Any_Relevant positives are in 1..10",
        "query_group_policy": "metadata.group_id=privacyqa-query:<raw QueryID>; policy identity is not used as a group",
        "source_repository": REPO_URL,
        "license": "MIT",
        "license_url": LICENSE_URL,
        "paper_url": PAPER_URL,
        "paper_citation": "PrivacyQA: A Question Answering Dataset for Privacy Policies",
        "source_urls": {"train": TRAIN_URL, "test": TEST_URL},
        "source_files": raw_files,
        "source_row_counts": {"train": train_rows, "test": test_rows},
        "source_policy_counts": {"canonical_documents": len(docs), "test_queries": len(test_queries)},
        "published_counts": {
            "documents": len(documents),
            "queries": len(queries),
            "positive_pairs": positive_pairs,
            "evidence": len(evidence),
            "judgments": len(judgments),
        },
        "query_audit": query_audit,
        "labels_ready_basis": "source Any_Relevant and Ann1..Ann6 annotations; no model labels",
    }
    bundle = {
        "schema_version": "dci-data-bundle-v1",
        "input_kind": "retrieval",
        "source": source,
        "documents": documents,
        "queries": sorted(queries, key=lambda row: row["_id"]),
        "qrels": {key: qrels[key] for key in sorted(qrels)},
        "evidence": sorted(evidence, key=lambda row: (row["query_id"], row["doc_id"])),
        "judgments": judgments,
        "provenance": provenance,
    }
    counts = {
        "source_train_rows": train_rows,
        "source_test_rows": test_rows,
        "source_test_queries": len(test_queries),
        "canonical_documents": len(docs),
        "eligible_queries": len(queries),
        "positive_pairs": positive_pairs,
        "evidence": len(evidence),
        "judgments": len(judgments),
    }
    return bundle, counts


def _assert_bundle_ready(bundle: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    source = bundle.get("source", {})
    provenance = bundle.get("provenance", {})
    if (
        not isinstance(source, dict)
        or source.get("dataset") != DATASET
        or source.get("revision") != REVISION
        or source.get("connector") != "privacyqa_pos10"
    ):
        raise ValueError(f"{SOURCE_ID}: release source identity is not PrivacyQA {REVISION}")
    if not isinstance(provenance, dict) or provenance.get("source_id") != SOURCE_ID:
        raise ValueError(f"{SOURCE_ID}: source_id provenance is missing")
    if manifest.get("state") != "labels_ready":
        raise ValueError(f"{SOURCE_ID}: release state is {manifest.get('state')!r}")
    if manifest.get("automatic_labels"):
        raise ValueError(f"{SOURCE_ID}: release declares automatic labels")
    documents = bundle.get("documents", [])
    queries = bundle.get("queries", [])
    qrels = bundle.get("qrels", {})
    if len(documents) < 1000 or len(queries) < 60:
        raise ValueError(f"{SOURCE_ID}: release is below 1000 documents/60 queries")
    document_ids = [row.get("_id") for row in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"{SOURCE_ID}: document IDs are not unique")
    if set(qrels) != {row.get("_id") for row in queries}:
        raise ValueError(f"{SOURCE_ID}: query/qrels coverage differs")
    positive_counts = []
    for query_id, gold in qrels.items():
        if not isinstance(gold, dict) or not 1 <= len(gold) <= 10:
            raise ValueError(f"{SOURCE_ID}/{query_id}: positive count is outside 1..10")
        if not set(gold).issubset(document_ids):
            raise ValueError(f"{SOURCE_ID}/{query_id}: qrels reference unknown documents")
        positive_counts.append(len(gold))
    if provenance.get("annotation_source") != "human":
        raise ValueError(f"{SOURCE_ID}: human annotation provenance is missing")
    if provenance.get("translation_status") != "native":
        raise ValueError(f"{SOURCE_ID}: native English provenance is missing")
    metadata = [row.get("metadata", {}) for row in queries]
    if any(not item.get("raw_query_id") or not item.get("original_query") for item in metadata):
        raise ValueError(f"{SOURCE_ID}: raw query provenance is incomplete")
    eligibility_rows = [
        {"_id": row["_id"], "text": row["text"], "group_id": row["metadata"]["group_id"]}
        for row in queries
    ]
    eligible, exclusions, _ = _prepare_query_eligibility(eligibility_rows, qrels)
    if len(eligible) < 60:
        raise ValueError(
            f"{SOURCE_ID}: fewer than 60 queries after policy-prefixed text eligibility"
        )
    return {
        "source_id": SOURCE_ID,
        "state": manifest["state"],
        "automatic_labels": manifest.get("automatic_labels"),
        "documents": len(documents),
        "queries": len(queries),
        "positive_pairs": sum(positive_counts),
        "positive_min": min(positive_counts),
        "positive_max": max(positive_counts),
        "source_revision": REVISION,
        "benchmark_eligible_queries": len(eligible),
        "benchmark_query_exclusions": exclusions,
    }


def _export_task_status(export_root: Path) -> tuple[bool, bool]:
    paths = [
        export_root / "metadata" / SOURCE_ID,
        export_root / "workspaces" / SOURCE_ID,
        export_root / "canonical" / SOURCE_ID,
    ]
    return paths[0].exists(), any(path.exists() for path in paths[1:])


def _check_existing(data_root: Path, *, require_export: bool = True) -> dict[str, Any]:
    _, _, release, export_root = _source_paths(data_root)
    manifest = verify_release(release, verify_content_hashes=True)
    bundle = load_bundle(release, verify=True, verify_content_hashes=True)
    receipt = _assert_bundle_ready(bundle, manifest)
    metadata_exists, sibling_exists = _export_task_status(export_root)
    if require_export and not metadata_exists:
        raise FileNotFoundError(f"Missing DCI export task: {export_root / 'metadata' / SOURCE_ID}")
    if metadata_exists:
        if not sibling_exists:
            raise ValueError(f"Partial DCI export task exists under {export_root}")
        source_manifest_path = export_root / "metadata" / SOURCE_ID / "source_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("source_fingerprint") != manifest.get("bundle_hash"):
            raise ValueError(f"{SOURCE_ID}: export source fingerprint differs from release bundle")
        exported = verify_document_task(export_root, SOURCE_ID, io_workers=4, verify_content_hashes=True)
        receipt["export"] = {
            "task": SOURCE_ID,
            "metadata_completion_sha256": file_digest(
                export_root / "metadata" / SOURCE_ID / "_COMPLETE.json"
            ),
            "verified": True,
            "manifest": exported.get("task", SOURCE_ID),
        }
    return receipt


def prepare(data_root: Path, *, check_only: bool = False) -> dict[str, Any]:
    data_root = Path(data_root)
    train_path, test_path, release, export_root = _source_paths(data_root)
    if check_only:
        return _check_existing(data_root)

    if release.exists() or release.is_symlink():
        receipt = _check_existing(data_root, require_export=False)
        metadata_exists, sibling_exists = _export_task_status(export_root)
        if metadata_exists:
            if not sibling_exists:
                raise ValueError(f"Partial DCI export task exists under {export_root}")
            return receipt
        export_dci(release, export_root, SOURCE_ID, io_workers=4)
        return _check_existing(data_root)

    task_paths = [
        export_root / "metadata" / SOURCE_ID,
        export_root / "workspaces" / SOURCE_ID,
        export_root / "canonical" / SOURCE_ID,
    ]
    if any(path.exists() or path.is_symlink() for path in task_paths):
        raise FileExistsError(f"Refusing to use partial pre-existing DCI export: {SOURCE_ID}")

    bundle, counts = _build_bundle(train_path, test_path, verify_hashes=True)
    receipt = publish_bundle(bundle, release, specification=SPECIFICATION)
    manifest = verify_release(release, verify_content_hashes=True)
    published = load_bundle(release, verify=True, verify_content_hashes=True)
    ready = _assert_bundle_ready(published, manifest)
    export_dci(release, export_root, SOURCE_ID, io_workers=4)
    verified = _check_existing(data_root)
    verified["source_counts"] = counts
    verified["release_bundle_hash"] = receipt["bundle_hash"]
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.data_root, check_only=args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
