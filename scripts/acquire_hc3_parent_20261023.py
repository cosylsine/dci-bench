#!/usr/bin/env python3
"""Acquire the pinned HC3 Finance parent corpus and preserve the old qrels.

The upstream file contains question rows with human and ChatGPT answer lists.
Every answer is a corpus candidate.  Exact duplicate answer text is grouped,
while every old MTEB document ID is retained as its own document when it maps
to the same text.  The legacy 100-query query/qrel view is copied verbatim;
answer side (human versus ChatGPT) is provenance only and never creates labels.

This is intentionally a CPU/data-only stage.  It uses the generic_jsonl source
connector for the immutable raw snapshot and publishes a custom bundle because
the retrieval labels come from the read-only legacy canonical release.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq

from dci_bench.data.pipeline.connectors import _base_bundle, fetch_source
from dci_bench.data.pipeline.core import file_digest, load_bundle, publish_bundle, verify_release


BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
SOURCE_ID = "mteb_parent_hc3_finance"
DATASET = "simpleai/HC3"
REVISION = "bf2846de7d27f17b5894f701317c824fc2192e0f"
PINNED_URL = (
    "https://modelscope.cn/api/v1/datasets/simpleai/HC3/repo"
    f"?Revision={REVISION}&FilePath=finance.jsonl"
)
TREE_URL = "https://modelscope.cn/api/v1/datasets/simpleai/HC3/repo/tree?Revision=master&Recursive=true"
OFFICIAL_REPO = "https://github.com/Hello-SimpleAI/chatgpt-comparison-detection"
MODELSCOPE_PAGE = "https://modelscope.cn/datasets/simpleai/HC3"
EXPECTED_BYTES = 9_889_186
EXPECTED_SHA256 = "72e3a439714c4693e115f8ad0b917988688abc07acba8de812198cb13cd5d72e"
EXPECTED_SOURCE_ROWS = 3_933
EXPECTED_HUMAN_ANSWERS = 3_933
EXPECTED_CHATGPT_ANSWERS = 4_503

OLD_RELEASE = BASE / "eval_data/dci_expansion_20261014_v1/canonical/LLMHC3FinanceRetrieval"
RAW_ROOT = BASE / "raw_data"
OUTPUT = BASE / "eval_data/pipeline_v1/mteb_parent_hc3_finance/expansion-v1"
RESEARCH_DIR = BASE / "eval_data/source_research_20261023/hc3_parent"
RESEARCH_PATH = RESEARCH_DIR / "research.json"
CROSSWALK_PATH = RESEARCH_DIR / "crosswalk.json"

SPEC: dict[str, Any] = {
    "id": SOURCE_ID,
    "connector": "generic_jsonl",
    "input_kind": "retrieval",
    "domain": "finance",
    "language": "en",
    "revision": REVISION,
    "license": "CC-BY-SA-4.0; preserve stricter licenses of HC3 source datasets",
    "options": {
        "dataset": DATASET,
        "genre": "human-versus-ChatGPT finance answers",
        "task": "HC3 Finance parent answer corpus with legacy MTEB qrels",
        "max_bytes": 20_000_000,
        "timeout": 60,
        "retries": 2,
        "headers": {
            "Accept": "application/octet-stream",
            "User-Agent": "dci-bench-data-pipeline",
        },
        "files": [
            {
                "role": "documents",
                "url": PINNED_URL,
                "name": "data/finance.jsonl",
                "bytes": EXPECTED_BYTES,
                "sha256": EXPECTED_SHA256,
            }
        ],
    },
    "metadata": {
        "official_repository": OFFICIAL_REPO,
        "provider_page": MODELSCOPE_PAGE,
        "source_scope": "All finance.jsonl question rows and all human/chatgpt answer strings.",
        "source_file": "finance.jsonl",
        "source_revision": REVISION,
        "source_file_bytes": EXPECTED_BYTES,
        "source_file_sha256": EXPECTED_SHA256,
        "legacy_release": str(OLD_RELEASE),
        "label_policy": "Copy legacy query IDs/qrel grades only; never derive labels from answer side.",
    },
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_exclusive_json(path: Path, payload: Any) -> str:
    """Write one new JSON artifact without replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite research artifact: {path}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def _old_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]], dict[str, Any]]:
    """Read the old release without modifying it and retain row order."""

    if not OLD_RELEASE.is_dir():
        raise FileNotFoundError(f"legacy canonical release is missing: {OLD_RELEASE}")
    corpus_path = OLD_RELEASE / "corpus/corpus.parquet"
    queries_path = OLD_RELEASE / "queries/queries.parquet"
    qrels_path = OLD_RELEASE / "data/test.parquet"
    tables = {
        "corpus": pq.read_table(corpus_path),
        "queries": pq.read_table(queries_path),
        "qrels": pq.read_table(qrels_path),
    }
    old_documents: list[dict[str, Any]] = []
    old_doc_ids: set[str] = set()
    for index, row in enumerate(tables["corpus"].to_pylist()):
        doc_id = row.get("_id")
        text = row.get("text")
        title = row.get("title", "")
        if not isinstance(doc_id, str) or not doc_id or doc_id in old_doc_ids:
            raise ValueError(f"legacy corpus has invalid/duplicate document ID at row {index}: {doc_id!r}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"legacy corpus has empty/non-string text at row {index}: {doc_id!r}")
        if not isinstance(title, str):
            raise ValueError(f"legacy corpus has non-string title at row {index}: {doc_id!r}")
        old_doc_ids.add(doc_id)
        old_documents.append({"_id": doc_id, "title": title, "text": text, "row_index": index})

    old_queries: list[dict[str, Any]] = []
    old_query_ids: set[str] = set()
    for index, row in enumerate(tables["queries"].to_pylist()):
        query_id = row.get("_id")
        text = row.get("text")
        if not isinstance(query_id, str) or not query_id or query_id in old_query_ids:
            raise ValueError(f"legacy queries have invalid/duplicate query ID at row {index}: {query_id!r}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"legacy query has empty/non-string text at row {index}: {query_id!r}")
        old_query_ids.add(query_id)
        old_queries.append({"_id": query_id, "text": text, "row_index": index})

    old_qrels: dict[str, dict[str, float]] = {}
    for index, row in enumerate(tables["qrels"].to_pylist()):
        query_id = row.get("query-id")
        doc_id = row.get("corpus-id")
        score = row.get("score")
        if not isinstance(query_id, str) or not isinstance(doc_id, str):
            raise ValueError(f"legacy qrel has invalid IDs at row {index}: {row!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"legacy qrel has invalid score at row {index}: {score!r}")
        if query_id not in old_query_ids or doc_id not in old_doc_ids:
            raise ValueError(f"legacy qrel references unknown ID at row {index}: {query_id!r}, {doc_id!r}")
        per_query = old_qrels.setdefault(query_id, {})
        if doc_id in per_query:
            raise ValueError(f"duplicate legacy qrel pair: {query_id!r}, {doc_id!r}")
        per_query[doc_id] = float(score)

    if set(old_qrels) != old_query_ids:
        missing = sorted(old_query_ids - set(old_qrels))
        extra = sorted(set(old_qrels) - old_query_ids)
        raise ValueError(f"legacy query/qrel coverage mismatch: missing={missing[:3]} extra={extra[:3]}")

    old_files = {}
    for name, path in {
        "corpus": corpus_path,
        "queries": queries_path,
        "qrels": qrels_path,
    }.items():
        old_files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_digest(path)}
    return old_documents, old_queries, old_qrels, {
        "files": old_files,
        "num_documents": len(old_documents),
        "num_queries": len(old_queries),
        "num_positive_pairs": sum(len(values) for values in old_qrels.values()),
        "max_positive_per_query": max(len(values) for values in old_qrels.values()),
    }


def _source_answers(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Parse all source answer strings and index exact text aliases."""

    text_refs: dict[str, list[dict[str, Any]]] = {}
    source_rows = 0
    side_counts = {"human": 0, "chatgpt": 0}
    answer_records = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank source line at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid source JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"source row is not an object at {path}:{line_number}")
            question = row.get("question")
            if not isinstance(question, str) or not question:
                raise ValueError(f"source question is empty/non-string at line {line_number}")
            source_row_index = source_rows
            source_rows += 1
            for side, key in (("human", "human_answers"), ("chatgpt", "chatgpt_answers")):
                answers = row.get(key)
                if not isinstance(answers, list) or not answers:
                    raise ValueError(f"source {key} must be a non-empty list at line {line_number}")
                for answer_index, answer in enumerate(answers):
                    if not isinstance(answer, str) or not answer:
                        raise ValueError(
                            f"source {key}[{answer_index}] must be a non-empty string at line {line_number}"
                        )
                    reference = {
                        "source_file": "data/finance.jsonl",
                        "source_row_index": source_row_index,
                        "source_line_number": line_number,
                        "answer_side": side,
                        "answer_index": answer_index,
                    }
                    text_refs.setdefault(answer, []).append(reference)
                    side_counts[side] += 1
                    answer_records += 1

    expected = {
        "source_rows": EXPECTED_SOURCE_ROWS,
        "human_answers": EXPECTED_HUMAN_ANSWERS,
        "chatgpt_answers": EXPECTED_CHATGPT_ANSWERS,
        "answer_records": EXPECTED_HUMAN_ANSWERS + EXPECTED_CHATGPT_ANSWERS,
    }
    actual = {
        "source_rows": source_rows,
        "human_answers": side_counts["human"],
        "chatgpt_answers": side_counts["chatgpt"],
        "answer_records": answer_records,
        "unique_answer_texts": len(text_refs),
        "exact_duplicate_text_groups": sum(1 for refs in text_refs.values() if len(refs) > 1),
        "exact_duplicate_answer_records": sum(len(refs) - 1 for refs in text_refs.values() if len(refs) > 1),
    }
    if actual != {**expected, **{key: actual[key] for key in actual if key not in expected}}:
        raise ValueError(f"source counts differ from pinned HC3 Finance facts: expected={expected}, actual={actual}")
    return text_refs, {"expected": expected, "actual": actual}


def _build_documents(
    old_documents: list[dict[str, Any]],
    text_refs: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    old_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for old_document in old_documents:
        old_by_text[old_document["text"]].append(old_document)

    old_mappings: list[dict[str, Any]] = []
    unmatched_old: list[dict[str, Any]] = []
    for old_document in old_documents:
        refs = text_refs.get(old_document["text"], [])
        mapping = {
            "old_doc_id": old_document["_id"],
            "old_row_index": old_document["row_index"],
            "old_text_sha256": _sha256_text(old_document["text"]),
            "source_ref_count": len(refs),
            "source_refs": refs,
            "exact_match": bool(refs),
            "mapped_document_ids": [old_document["_id"]] if refs else [],
        }
        old_mappings.append(mapping)
        if not refs:
            unmatched_old.append(mapping)

    # The text index is insertion ordered by source appearance, which keeps
    # output deterministic while allowing exact duplicate grouping.
    documents: list[dict[str, Any]] = []
    text_groups: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    source_refs_mapped = 0
    for text, refs in text_refs.items():
        old_for_text = old_by_text.get(text, [])
        if old_for_text:
            document_ids = [item["_id"] for item in old_for_text]
        else:
            digest = _sha256_text(text)
            document_ids = [f"hc3-answer-{digest[:32]}"]
        for document_id in document_ids:
            if document_id in used_ids:
                raise ValueError(f"document ID collision while building source corpus: {document_id}")
            used_ids.add(document_id)
        source_refs_mapped += len(refs)
        text_sha256 = _sha256_text(text)
        text_groups.append(
            {
                "text_sha256": text_sha256,
                "text_chars": len(text),
                "text_bytes": len(text.encode("utf-8")),
                "source_ref_count": len(refs),
                "source_refs": refs,
                "old_doc_ids": [item["_id"] for item in old_for_text],
                "document_ids": document_ids,
                "exact_text_deduplicated": len(refs) > 1,
            }
        )
        for document_id in document_ids:
            old_document = next((item for item in old_for_text if item["_id"] == document_id), None)
            metadata: dict[str, Any] = {
                "source_file": "data/finance.jsonl",
                "source_dataset": DATASET,
                "source_revision": REVISION,
                "source_ref_count": len(refs),
                "source_refs": refs,
                "exact_text_sha256": text_sha256,
                "exact_text_deduplicated": len(refs) > 1,
                "annotation_source": "source",
                "translation_status": "unknown",
                "answer_corpus_role": "human_or_chatgpt_answer_text",
                "legacy_doc_id": document_id if old_document else None,
                "id_derivation": "legacy_exact_text_crosswalk" if old_document else "sha256(exact_answer_text)",
            }
            documents.append(
                {
                    "_id": document_id,
                    "title": old_document["title"] if old_document else "HC3 Finance answer",
                    "text": text,
                    "language": "en",
                    "metadata": metadata,
                }
            )

    if source_refs_mapped != sum(len(refs) for refs in text_refs.values()):
        raise ValueError("not all source answer references were assigned to a document")
    if any(document["_id"] not in used_ids for document in documents):
        raise ValueError("document mapping produced an unknown document ID")
    mapping_stats = {
        "old_documents_total": len(old_documents),
        "old_documents_exactly_mapped": len(old_documents) - len(unmatched_old),
        "old_documents_unmatched": len(unmatched_old),
        "old_text_groups_with_multiple_legacy_ids": sum(
            1 for text, rows in old_by_text.items() if text in text_refs and len(rows) > 1
        ),
        "source_answer_records_mapped": source_refs_mapped,
        "source_unique_texts": len(text_refs),
        "release_documents_before_quality": len(documents),
        "exact_dedup_groups": sum(1 for refs in text_refs.values() if len(refs) > 1),
    }
    crosswalk = {
        "schema_version": "hc3-finance-crosswalk-v1",
        "source": {
            "dataset": DATASET,
            "revision": REVISION,
            "file": "data/finance.jsonl",
            "sha256": EXPECTED_SHA256,
            "bytes": EXPECTED_BYTES,
        },
        "legacy_release": str(OLD_RELEASE),
        "mapping_method": "exact UTF-8 decoded answer text equality; no normalization or fuzzy matching",
        "deduplication_method": "exact answer text only; every legacy ID retained as a separate document",
        "mapping_stats": mapping_stats,
        "old_document_mappings": old_mappings,
        "text_groups": text_groups,
    }
    return documents, crosswalk, mapping_stats


def _build_bundle(
    snapshot: Path,
    old_documents: list[dict[str, Any]],
    old_queries: list[dict[str, Any]],
    old_qrels: dict[str, dict[str, float]],
    documents: list[dict[str, Any]],
    crosswalk_sha256: str,
    mapping_stats: dict[str, Any],
    source_counts: dict[str, Any],
) -> dict[str, Any]:
    manifest = json.loads((snapshot / "source_manifest.json").read_text(encoding="utf-8"))
    bundle = _base_bundle(SPEC, manifest)
    bundle["documents"] = documents
    bundle["queries"] = [
        {
            "_id": row["_id"],
            "text": row["text"],
            "language": "en",
            "metadata": {
                "annotation_source": "source",
                "legacy_query_id": row["_id"],
                "legacy_query_row_index": row["row_index"],
                "legacy_release": str(OLD_RELEASE),
                "query_selection": "all legacy retained queries",
            },
        }
        for row in old_queries
    ]
    # Assign a fresh dict so the old read-only object can never be mutated by a
    # later quality pass or serialization step.
    bundle["qrels"] = {query_id: dict(per_query) for query_id, per_query in old_qrels.items()}
    bundle["provenance"].update(
        {
            "source_id": SOURCE_ID,
            "domain": "finance",
            "language": "en",
            "genre": "human-versus-ChatGPT finance answers",
            "task": "HC3 Finance parent answer corpus with legacy MTEB qrels",
            "source_scope": "All finance.jsonl human and ChatGPT answer strings; exact text dedup only.",
            "source_counts": source_counts,
            "mapping_stats": mapping_stats,
            "legacy_query_selection": "All legacy retained queries copied with exact IDs/text.",
            "qrels_derivation": "Legacy qrels copied verbatim; no labels generated from human/chatgpt answer side.",
            "crosswalk": {
                "schema_version": "hc3-finance-crosswalk-v1",
                "path": str(CROSSWALK_PATH),
                "sha256": crosswalk_sha256,
                "mapping_method": "exact UTF-8 decoded answer text equality",
            },
            "source_file": {
                "path": "data/finance.jsonl",
                "bytes": EXPECTED_BYTES,
                "sha256": EXPECTED_SHA256,
                "revision": REVISION,
            },
            "native_or_translated": "native English HC3 source; answer-side provenance preserved",
            "license": SPEC["license"],
        }
    )
    return bundle


def _verify_legacy_view(output: Path, old_queries: list[dict[str, Any]], old_qrels: dict[str, dict[str, float]], old_documents: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = load_bundle(output)
    actual_queries = [(row["_id"], row["text"]) for row in bundle["queries"]]
    expected_queries = [(row["_id"], row["text"]) for row in old_queries]
    if actual_queries != expected_queries:
        raise ValueError("published query IDs/texts differ from the legacy canonical query view")
    if bundle["qrels"] != old_qrels:
        raise ValueError("published qrels differ from the legacy canonical qrels, including grades")
    release_docs = {row["_id"]: row for row in bundle["documents"]}
    missing = []
    changed = []
    for old_document in old_documents:
        current = release_docs.get(old_document["_id"])
        if current is None:
            missing.append(old_document["_id"])
        elif current.get("text") != old_document["text"]:
            changed.append(old_document["_id"])
    if missing or changed:
        raise ValueError(f"legacy document preservation failed: missing={missing[:3]}, changed={changed[:3]}")
    qrel_doc_ids = {doc_id for per_query in old_qrels.values() for doc_id in per_query}
    if not qrel_doc_ids.issubset(release_docs):
        raise ValueError("published qrels reference a missing legacy document")
    return {
        "queries_exact": True,
        "qrels_exact": True,
        "legacy_documents_exact": True,
        "legacy_qrel_document_ids_present": True,
        "published_queries": len(bundle["queries"]),
        "published_positive_pairs": sum(len(values) for values in bundle["qrels"].values()),
        "published_documents": len(bundle["documents"]),
        "max_positive_per_query": max(len(values) for values in bundle["qrels"].values()),
    }


def _research_payload(
    *,
    status: str,
    snapshot: Path,
    source_counts: dict[str, Any],
    mapping_stats: dict[str, Any],
    old_stats: dict[str, Any],
    crosswalk_sha256: str,
    release_manifest: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_file = snapshot / "data/finance.jsonl"
    source_file_hash = file_digest(source_file) if source_file.is_file() else None
    source_file_bytes = source_file.stat().st_size if source_file.is_file() else None
    payload: dict[str, Any] = {
        "schema_version": "hc3-parent-research-v1",
        "acquisition_status": status,
        "sources": [
            {
                "id": SOURCE_ID,
                "dataset": DATASET,
                "repository": OFFICIAL_REPO,
                "provider_page": MODELSCOPE_PAGE,
                "revision": REVISION,
                "urls": [OFFICIAL_REPO, MODELSCOPE_PAGE, PINNED_URL, TREE_URL],
                "license": SPEC["license"],
                "acquisition_status": status,
                "raw_snapshot": str(snapshot),
                "release_path": str(OUTPUT) if release_manifest is not None else None,
                "release_state": release_manifest.get("state") if release_manifest else None,
                "source_file": {
                    "path": str(source_file),
                    "bytes": source_file_bytes,
                    "expected_bytes": EXPECTED_BYTES,
                    "sha256": source_file_hash,
                    "expected_sha256": EXPECTED_SHA256,
                    "verified": source_file_bytes == EXPECTED_BYTES and source_file_hash == EXPECTED_SHA256,
                },
                "counts": {
                    **source_counts.get("actual", {}),
                    **mapping_stats,
                    "legacy": old_stats,
                    "release": {
                        "num_documents": release_manifest.get("num_documents") if release_manifest else None,
                        "num_queries": release_manifest.get("num_queries") if release_manifest else None,
                        "num_positive_pairs": release_manifest.get("num_positive_pairs") if release_manifest else None,
                        "max_positive_per_query": verification.get("max_positive_per_query") if verification else None,
                    },
                },
                "crosswalk": {
                    "path": str(CROSSWALK_PATH),
                    "sha256": crosswalk_sha256,
                    "method": "exact answer text; source row + answer side + answer index retained",
                },
                "quality_policy": {
                    "max_positive_per_query": None,
                    "exclude_review_required": False,
                },
                "verification": verification,
                "limitations": [
                    "Answer side is provenance only; no human/chatgpt-derived qrels were generated.",
                    "Unjudged parent answers remain unjudged; copied legacy qrels are the only labels.",
                    "Exact text deduplication is lossless through source aliases; legacy IDs are never removed.",
                    "Unique answer text count is the corpus count after exact dedup, while source answer records count all list entries.",
                ],
            }
        ],
        "command": f"{sys.executable} {ROOT / 'scripts/acquire_hc3_parent_20261023.py'}",
    }
    if failure is not None:
        payload["failure"] = failure
    return payload


def main() -> None:
    if OUTPUT.exists() or RESEARCH_PATH.exists() or CROSSWALK_PATH.exists():
        raise FileExistsError("Refusing to overwrite existing HC3 parent release or research artifacts")

    snapshot = fetch_source(SPEC, RAW_ROOT)
    source_file = snapshot / "data/finance.jsonl"
    if not source_file.is_file():
        raise FileNotFoundError(f"raw connector did not materialise expected source file: {source_file}")
    source_hash = file_digest(source_file)
    source_bytes = source_file.stat().st_size
    if source_bytes != EXPECTED_BYTES or source_hash != EXPECTED_SHA256:
        raise ValueError(
            f"source hash/size mismatch: bytes={source_bytes}/{EXPECTED_BYTES}, "
            f"sha256={source_hash}/{EXPECTED_SHA256}"
        )

    old_documents, old_queries, old_qrels, old_stats = _old_inputs()
    text_refs, source_counts = _source_answers(source_file)
    documents, crosswalk, mapping_stats = _build_documents(old_documents, text_refs)
    crosswalk_json = json.dumps(crosswalk, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    crosswalk_sha256 = hashlib.sha256(crosswalk_json.encode("utf-8")).hexdigest()

    if mapping_stats["old_documents_unmatched"]:
        # Keep the complete source and an explicit discrepancy receipt, but do
        # not create a release whose old qrels would point at an unproven map.
        _write_exclusive_json(CROSSWALK_PATH, crosswalk)
        failure = {
            "reason": "exact_old_document_mapping_incomplete",
            "unmatched_old_documents": [
                item["old_doc_id"]
                for item in crosswalk["old_document_mappings"]
                if not item["exact_match"]
            ],
        }
        payload = _research_payload(
            status="blocked_mapping",
            snapshot=snapshot,
            source_counts=source_counts,
            mapping_stats=mapping_stats,
            old_stats=old_stats,
            crosswalk_sha256=crosswalk_sha256,
            release_manifest=None,
            verification=None,
            failure=failure,
        )
        _write_exclusive_json(RESEARCH_PATH, payload)
        raise RuntimeError(f"old exact mapping incomplete; see {RESEARCH_PATH}")

    bundle = _build_bundle(
        snapshot,
        old_documents,
        old_queries,
        old_qrels,
        documents,
        crosswalk_sha256,
        mapping_stats,
        source_counts,
    )
    release_manifest = publish_bundle(
        bundle,
        OUTPUT,
        specification=SPEC,
        raw_snapshot=snapshot,
        policy={"max_positive_per_query": None, "exclude_review_required": False},
    )
    verified_manifest = verify_release(OUTPUT)
    if verified_manifest != release_manifest:
        raise ValueError("verify_release returned a different release manifest")
    if verified_manifest["num_documents"] <= 1000:
        raise ValueError(f"published corpus is not larger than 1000: {verified_manifest['num_documents']}")
    verification = _verify_legacy_view(OUTPUT, old_queries, old_qrels, old_documents)
    if verification["published_documents"] != verified_manifest["num_documents"]:
        raise ValueError("legacy verification count differs from release manifest")

    _write_exclusive_json(CROSSWALK_PATH, crosswalk)
    payload = _research_payload(
        status="published_verified",
        snapshot=snapshot,
        source_counts=source_counts,
        mapping_stats=mapping_stats,
        old_stats=old_stats,
        crosswalk_sha256=crosswalk_sha256,
        release_manifest=verified_manifest,
        verification=verification,
    )
    _write_exclusive_json(RESEARCH_PATH, payload)
    print(
        json.dumps(
            {
                "status": "published_verified",
                "raw_snapshot": str(snapshot),
                "release": str(OUTPUT),
                "research": str(RESEARCH_PATH),
                "crosswalk": str(CROSSWALK_PATH),
                "source_sha256": source_hash,
                "source_bytes": source_bytes,
                "source_counts": source_counts["actual"],
                "mapping_stats": mapping_stats,
                "release_manifest": verified_manifest,
                "verification": verification,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"hc3-parent-acquire: {exc}", file=sys.stderr)
        raise SystemExit(2)
