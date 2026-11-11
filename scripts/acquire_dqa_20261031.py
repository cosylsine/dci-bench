#!/usr/bin/env python3
"""Audit and, only when justified, build a DQA-Bench Chinese retrieval release.

The public DQA-Bench repository contains examples only.  The complete package
is advertised through a Google Drive link which is not currently accessible in
this environment.  This adapter therefore has two deliberately separate
paths:

* public examples are fetched into an immutable raw snapshot and audited;
* a locally supplied, read-only extracted package may be converted, but a
  canonical release is published only for a source with explicit native
  Chinese provenance, at least 1,000 independent source documents and at least
  60 queries.

Rows from forum QA may use the published answer as the retrieval document,
which is the same source relation used by QA-to-retrieval adaptations.  The
answer is retained in query metadata and the qrel is always a positive source
relation.  Product passages use the supplied ``retrieval`` text and preserve
the original document name.  Instance/tool data is recorded as an excluded
synthetic category and is never mixed into a retrieval release.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.core import build_source, load_bundle, verify_release
from dci_bench.data.pipeline.download import download_url, file_digest


BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
RAW_ROOT = BASE / "raw_data/gapfill_dqa_zh/20261031_v1"
RELEASE_ROOT = BASE / "eval_data/pipeline_v1/gapfill_dqa_zh/20261031_v1"
RESEARCH_ROOT = BASE / "eval_data/source_research_20261031_v1/dqa"
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
MIN_DOCUMENTS = 1000
MIN_QUERIES = 60
OFFICIAL_REPOSITORY = "https://github.com/XMUDM/DQABench"
OFFICIAL_REVISION = "b2deb7a2e9267be26c35d91327e00132f4933c17"
DRIVE_URL = "https://drive.google.com/file/d/1pOn-2-HdB4exaHT2Yh0nxZzy-8B1cuvQL/view?usp=drive_link"
PAPER_URL = "https://arxiv.org/abs/2409.04475"
PAPER_REVISION = "arXiv:2409.04475v2 [cs.DB], 2024-12-06"

POLICY = {
    "max_positive_per_query": None,
    "exclude_review_required": False,
    "drop_near_duplicates": False,
    "max_document_bytes": 16 * 1024 * 1024,
    "max_query_bytes": 16 * 1024 * 1024,
}


PUBLIC_SAMPLE_FILES = (
    "general_knowledge_with_tags/zh.json",
    "ZH/general_knowledge_subset_example/train.json",
    "ZH/general_knowledge_subset_example/test.json",
    "ZH/general_knowledge_subset_example/validation.json",
    "ZH/specific_product_subset_example/GaussDB/train.json",
    "ZH/specific_product_subset_example/GaussDB/test.json",
    "ZH/specific_product_subset_example/GaussDB/validation.json",
    "ZH/specific_product_subset_example/openGauss/train.json",
    "ZH/specific_product_subset_example/openGauss/test.json",
    "ZH/specific_product_subset_example/openGauss/validation.json",
    "ZH/specific_instance_subset_example/scenarios-based/train.json",
    "ZH/specific_instance_subset_example/scenarios-based/test.json",
    "ZH/specific_instance_subset_example/scenarios-based/validation.json",
    "ZH/specific_instance_subset_example/tool-based/train.json",
    "ZH/specific_instance_subset_example/tool-based/test.json",
    "ZH/specific_instance_subset_example/tool-based/validation.json",
)

PAPER_EVIDENCE = [
    {
        "location": "Abstract",
        "summary": "The construction workflow uses language models to create and revise bilingual QA data."
    },
    {
        "location": "Section 3, Data Statistics",
        "summary": "Both translation directions are used when forming English and Chinese QA pairs."
    },
    {
        "location": "Section 3.1, Step 2",
        "summary": "GPT-4 rewrites answers selected from online question-answer communities."
    },
    {
        "location": "Section 3.2, Steps 2-3",
        "summary": "Product questions and answers are generated from manual passages; retrieval passages are selected with vector similarity."
    },
    {
        "location": "Section 3.3",
        "summary": "Instance questions are generated with language models and include tool-based scenarios."
    }
]


class SourceBlocked(RuntimeError):
    """The source cannot truthfully be advertised as a native retrieval set."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_once(path: Path, data: bytes) -> None:
    """Write an immutable artifact, accepting an identical existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def write_json_once(path: Path, value: Any) -> None:
    write_once(path, canonical_json(value))


def materialize_fixed_samples(source_root: Path, raw_root: Path = RAW_ROOT) -> dict[str, Any]:
    """Persist already-fetched fixed-commit samples without network or overwrite."""
    source_root = Path(source_root)
    destination_root = raw_root / "public_samples_reviewed_20261031"
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in PUBLIC_SAMPLE_FILES:
        source = source_root / relative
        destination = destination_root / relative
        if not source.is_file() or source.is_symlink():
            missing.append(relative)
            continue
        write_once(destination, source.read_bytes())
        digest, size = file_digest(destination, max_bytes=MAX_DOWNLOAD_BYTES)
        files.append({
            "path": str(destination),
            "relative_path": relative,
            "source_readonly_path": str(source),
            "sha256": digest,
            "bytes": size,
            "materialization": "copied from a local checkout pinned to the official commit; no network download",
        })
    manifest = {
        "schema_version": "dqa-fixed-sample-materialization-v1",
        "repository": OFFICIAL_REPOSITORY,
        "revision": OFFICIAL_REVISION,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "files": files,
        "missing": missing,
        "complete": not missing and len(files) == len(PUBLIC_SAMPLE_FILES),
        "raw_inputs_immutable": True,
    }
    write_json_once(raw_root / "public_samples_reviewed_manifest.json", manifest)
    if missing:
        raise SourceBlocked(f"fixed sample checkout is missing files: {missing}")
    return manifest


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else [value]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"all rows must be JSON objects: {path}")
    return rows


def category_for_path(path: Path) -> tuple[str | None, str | None, str | None]:
    """Return category, product/subtype and split for an official ZH path."""
    parts = [part for part in path.as_posix().split("/") if part]
    if "general_knowledge_with_tags" in parts:
        return "general_knowledge_with_tags", None, None
    try:
        zh_index = parts.index("ZH")
    except ValueError:
        return None, None, None
    tail = parts[zh_index + 1 :]
    if not tail:
        return None, None, None
    filename = Path(tail[-1]).stem
    split = filename if filename in {"train", "test", "validation"} else None
    names = set(tail)
    if "specific_product" in names or "specific_product_subset_example" in names:
        product = next((item for item in tail if item in {"GaussDB", "openGauss"}), None)
        return "specific_product", product, split
    if "specific_instance" in names or "specific_instance_subset_example" in names:
        subtype = next((item for item in tail if item in {"scenarios-based", "tool-based"}), None)
        return "specific_instance", subtype, split
    if "forum_QA" in names:
        return "forum_QA", None, split
    if "multiple_choice" in names:
        return "multiple_choice", None, split
    if "general_knowledge_with_tags" in names:
        return "general_knowledge_with_tags", None, split
    if "general_knowledge_subset_example" in names:
        return "general_knowledge_sample", None, split
    return None, None, split


def category_policy(category: str) -> dict[str, str | bool]:
    """Describe origin without promoting any unverified row to human/native."""
    if category == "forum_QA":
        return {
            "retrieval_eligible": True,
            "native_status": "unverified_translation",
            "translation_status": "unverified; paper states English/Chinese translation and row origin is not exposed",
            "annotation_source": "internet_QA_with_LLM_rewrite_claimed_by_paper",
            "document_relation": "published_answer_body",
        }
    if category == "general_knowledge_sample":
        return {
            "retrieval_eligible": True,
            "native_status": "unverified_translation",
            "translation_status": "unverified; public sample has no source-language provenance",
            "annotation_source": "public_sample_origin_unresolved",
            "document_relation": "published_answer_body",
        }
    if category == "general_knowledge_with_tags":
        return {
            "retrieval_eligible": True,
            "native_status": "unverified_translation",
            "translation_status": "unverified; tags file has no source-language provenance",
            "annotation_source": "public_tagged_QA_origin_unresolved",
            "document_relation": "published_answer_body",
        }
    if category == "specific_product":
        return {
            "retrieval_eligible": True,
            "native_status": "unverified_translation",
            "translation_status": "unverified; paper describes bilingual translation and LLM generation from manuals",
            "annotation_source": "llm_generated_from_product_manual",
            "document_relation": "supplied_retrieval_passage",
        }
    if category == "specific_instance":
        return {
            "retrieval_eligible": False,
            "native_status": "synthetic_instance",
            "translation_status": "synthetic/LLM-assisted; tool or scenario trace",
            "annotation_source": "LLM_generated_with_manual_or_expert_seed_process",
            "document_relation": "tool_trace_not_corpus_document",
        }
    if category == "multiple_choice":
        return {
            "retrieval_eligible": False,
            "native_status": "unverified_translation",
            "translation_status": "unverified; multiple-choice task has no retrieval corpus relation",
            "annotation_source": "source_origin_unresolved",
            "document_relation": "no_retrieval_document",
        }
    raise ValueError(f"unknown DQA category: {category}")


def _text(value: Any) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def _source_group(category: str, product_or_subtype: str | None, source_file: str, row: Mapping[str, Any]) -> str:
    if category == "specific_product":
        product = product_or_subtype or "unknown-product"
        document = _text(row.get("doc")).strip() or "unknown-document"
        return f"dqa:zh:specific_product:{product}:{document}"
    if category == "specific_instance":
        subtype = product_or_subtype or "unknown-instance"
        return f"dqa:zh:specific_instance:{subtype}:{source_file}"
    return f"dqa:zh:{category}:{source_file}"


def _document_id(category: str, product_or_subtype: str | None, source_key: str, text: str) -> str:
    identity = "\0".join((category, product_or_subtype or "", source_key, text))
    return "dqa-doc-" + sha256_bytes(identity.encode("utf-8"))[:32]


def _query_id(category: str, source_file: str, row_number: int) -> str:
    safe = source_file.replace("/", "!")
    return f"dqa-zh-{category}-{safe}#row{row_number}"


def convert_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    category: str,
    product_or_subtype: str | None,
    source_file: str,
    source_split: str | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    """Convert one official source file while retaining every QA relation."""
    policy = category_policy(category)
    documents: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, int]] = {}
    exclusions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, 1):
        counts["source_rows"] += 1
        question = _text(row.get("question")).strip()
        qid = _query_id(category, source_file, row_number)
        if not question:
            exclusions.append({"query_id": qid, "row": row_number, "reason": "missing_question"})
            counts["excluded_rows"] += 1
            continue
        group = _source_group(category, product_or_subtype, source_file, row)
        common_metadata: dict[str, Any] = {
            "source_family": "DQA-Bench",
            "source_category": category,
            "source_subcategory": product_or_subtype,
            "source_file": source_file,
            "source_row_1based": row_number,
            "source_split": source_split or "unknown",
            "source_group": group,
            "translation_status": policy["translation_status"],
            "native_status": policy["native_status"],
            "annotation_source": policy["annotation_source"],
            "source_question": question,
            "source_answer": row.get("answer"),
        }

        if category == "specific_instance":
            exclusions.append({
                "query_id": qid,
                "row": row_number,
                "reason": "synthetic_instance_or_tool_trace_not_retrieval_corpus",
                "source_group": group,
            })
            counts["excluded_synthetic_instance_rows"] += 1
            continue
        if category == "multiple_choice":
            exclusions.append({
                "query_id": qid,
                "row": row_number,
                "reason": "multiple_choice_has_no_source_document_relation",
                "source_group": group,
            })
            counts["excluded_non_retrieval_rows"] += 1
            continue

        if category == "specific_product":
            document_text = _text(row.get("retrieval")).strip()
            source_document_key = _text(row.get("doc")).strip()
            if not document_text or not source_document_key:
                exclusions.append({
                    "query_id": qid,
                    "row": row_number,
                    "reason": "missing_product_document_name_or_retrieval_text",
                    "source_group": group,
                })
                counts["excluded_missing_product_evidence"] += 1
                continue
            common_metadata.update({
                "source_document_name": source_document_key,
                "document_relation": "supplied_retrieval_passage",
            })
        else:
            document_text = _text(row.get("answer")).strip()
            source_document_key = "answer:" + sha256_bytes(document_text.encode("utf-8"))[:32]
            if not document_text:
                exclusions.append({
                    "query_id": qid,
                    "row": row_number,
                    "reason": "missing_or_empty_answer_document",
                    "source_group": group,
                })
                counts["excluded_missing_answer"] += 1
                continue
            common_metadata.update({
                "source_document_name": source_document_key,
                "document_relation": "published_answer_body",
            })

        document_id = _document_id(category, product_or_subtype, source_document_key, document_text)
        document = documents.get(document_id)
        alias = {
            "source_file": source_file,
            "source_row_1based": row_number,
            "source_split": source_split or "unknown",
            "source_group": group,
        }
        if document is None:
            documents[document_id] = {
                "_id": document_id,
                "title": source_document_key if category == "specific_product" else "",
                "text": document_text,
                "language": "zh",
                "metadata": {
                    "source_family": "DQA-Bench",
                    "source_category": category,
                    "source_subcategory": product_or_subtype,
                    "source_document_key": source_document_key,
                    "source_group": group,
                    "translation_status": policy["translation_status"],
                    "native_status": policy["native_status"],
                    "annotation_source": policy["annotation_source"],
                    "source_aliases": [alias],
                },
            }
        else:
            document["metadata"]["source_aliases"].append(alias)
            counts["exact_duplicate_document_aliases"] += 1

        queries.append({
            "_id": qid,
            "text": question,
            "language": "zh",
            "metadata": {**common_metadata, "positive_document_id": document_id},
        })
        qrels[qid] = {document_id: 1}
        counts["queries"] += 1
        counts["positive_pairs"] += 1

    counts["documents"] = len(documents)
    counts["unique_source_documents"] = len({
        document["metadata"]["source_document_key"] for document in documents.values()
    })
    return documents, queries, qrels, {
        "counts": dict(counts),
        "exclusions": exclusions,
        "category": category,
        "subcategory": product_or_subtype,
        "source_file": source_file,
        "translation_status": policy["translation_status"],
        "native_status": policy["native_status"],
        "annotation_source": policy["annotation_source"],
        "document_relation": policy["document_relation"],
        "qrels_policy": "Every published qrel is the original row-to-answer or row-to-supplied-retrieval relation with grade 1; no negatives are inferred.",
    }


def _merge_category(
    category: str,
    product_or_subtype: str | None,
    converted: Iterable[tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]],
) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, int]] = {}
    audits: list[dict[str, Any]] = []
    for source_documents, source_queries, source_qrels, audit in converted:
        for document_id, document in source_documents.items():
            if document_id in documents:
                documents[document_id]["metadata"]["source_aliases"].extend(document["metadata"]["source_aliases"])
            else:
                documents[document_id] = document
        queries.extend(source_queries)
        for query_id, positives in source_qrels.items():
            if query_id in qrels:
                raise ValueError(f"duplicate query ID across DQA source files: {query_id}")
            qrels[query_id] = positives
        audits.append(audit)
    counts = Counter()
    for audit in audits:
        counts.update(audit["counts"])
    counts["documents"] = len(documents)
    counts["queries"] = len(queries)
    counts["positive_pairs"] = sum(len(positives) for positives in qrels.values())
    counts["unique_source_documents"] = len({
        document["metadata"]["source_document_key"] for document in documents.values()
    })
    policy = category_policy(category)
    return {
        "category": category,
        "subcategory": product_or_subtype,
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "audits": audits,
        "counts": dict(counts),
        "native_status": policy["native_status"],
        "translation_status": policy["translation_status"],
        "annotation_source": policy["annotation_source"],
        "retrieval_eligible": policy["retrieval_eligible"],
    }


def discover_package_files(package_root: Path) -> list[Path]:
    if not package_root.exists() or not package_root.is_dir():
        raise FileNotFoundError(f"DQA package root is not a directory: {package_root}")
    output = []
    for path in sorted(package_root.rglob("*.json")):
        category, _, _ = category_for_path(path.relative_to(package_root))
        if category is not None:
            output.append(path)
    return output


def convert_package(package_root: Path) -> dict[str, dict[str, Any]]:
    """Parse only official ZH category JSON files from a read-only package."""
    grouped: dict[tuple[str, str | None], list[tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]]] = defaultdict(list)
    for path in discover_package_files(package_root):
        relative = path.relative_to(package_root).as_posix()
        category, product_or_subtype, split = category_for_path(Path(relative))
        if category is None:
            continue
        rows = read_json_records(path)
        grouped[(category, product_or_subtype)].append(
            convert_rows(
                rows,
                category=category,
                product_or_subtype=product_or_subtype,
                source_file=relative,
                source_split=split,
            )
        )
    return {
        category if product_or_subtype is None else f"{category}:{product_or_subtype}":
        _merge_category(category, product_or_subtype, converted)
        for (category, product_or_subtype), converted in sorted(grouped.items())
    }


def fetch_public_samples(raw_root: Path = RAW_ROOT) -> dict[str, Any]:
    """Fetch the small, pinned public examples with a 512 MiB hard limit."""
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    sample_root = raw_root / "public_samples"
    downloaded_total = 0
    for relative in PUBLIC_SAMPLE_FILES:
        url = f"https://raw.githubusercontent.com/XMUDM/DQABench/{OFFICIAL_REVISION}/1_Dataset_of_Benchmark_DQA/{relative}"
        destination = sample_root / relative
        try:
            partial = destination.with_name(destination.name + ".partial")
            if partial.exists() and not destination.exists():
                raise SourceBlocked(
                    f"retained partial download refuses an implicit retry: {partial}"
                )
            if destination.exists():
                digest, size = file_digest(destination, max_bytes=MAX_DOWNLOAD_BYTES)
                reused = True
            else:
                remaining = MAX_DOWNLOAD_BYTES - downloaded_total
                if remaining <= 0:
                    raise SourceBlocked(
                        f"cumulative public-sample download limit reached ({MAX_DOWNLOAD_BYTES} bytes)"
                    )
                digest, size = download_url(
                    url,
                    destination,
                    timeout=60,
                    retries=1,
                    max_bytes=remaining,
                    overwrite=False,
                )
                reused = False
                downloaded_total += size
            manifest_rows.append({
                "path": relative,
                "url": url,
                "sha256": digest,
                "bytes": size,
                "reused_existing": reused,
            })
        except Exception as exc:
            failures.append({"path": relative, "url": url, "error": f"{type(exc).__name__}: {exc}"})
    manifest = {
        "schema_version": "dqa-public-sample-manifest-v1",
        "repository": OFFICIAL_REPOSITORY,
        "revision": OFFICIAL_REVISION,
        "max_download_bytes": MAX_DOWNLOAD_BYTES,
        "downloaded_bytes_this_run": downloaded_total,
        "files": manifest_rows,
        "failures": failures,
        "complete": not failures and len(manifest_rows) == len(PUBLIC_SAMPLE_FILES),
    }
    write_json_once(raw_root / "public_samples_manifest.json", manifest)
    return manifest


def probe_drive(url: str = DRIVE_URL) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=20) as response:
            return {
                "url": url,
                "status": "accessible_metadata",
                "http_status": int(response.status),
                "content_length": response.headers.get("Content-Length"),
                "content_type": response.headers.get("Content-Type"),
            }
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": "unavailable", "http_status": exc.code, "error": str(exc)}
    except urllib.error.URLError as exc:
        return {"url": url, "status": "unavailable", "error": str(exc.reason)}


def _write_canonical_input(group: Mapping[str, Any], root: Path) -> dict[str, Path]:
    category = str(group["category"])
    subcategory = group.get("subcategory")
    slug = category if not subcategory else f"{category}_{subcategory}"
    canonical = root / "canonical_input" / slug
    documents_path = canonical / "documents.jsonl"
    queries_path = canonical / "queries.jsonl"
    qrels_path = canonical / "qrels.json"
    write_once(
        documents_path,
        b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
                 for row in group["documents"].values()),
    )
    write_once(
        queries_path,
        b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
                 for row in group["queries"]),
    )
    write_json_once(qrels_path, group["qrels"])
    return {"documents": documents_path, "queries": queries_path, "qrels": qrels_path}


def _validate_positive_qrels(group: Mapping[str, Any]) -> None:
    queries = {query["_id"] for query in group["queries"]}
    documents = set(group["documents"])
    for query_id, positives in group["qrels"].items():
        if query_id not in queries or not positives:
            raise ValueError(f"qrel query is missing or has no positives: {query_id}")
        for document_id, score in positives.items():
            if document_id not in documents or not isinstance(score, (int, float)) or score <= 0:
                raise ValueError(f"qrel is not a strict positive source relation: {query_id}/{document_id}")


def eligibility(group: Mapping[str, Any]) -> tuple[bool, list[str]]:
    counts = group["counts"]
    reasons: list[str] = []
    if group["native_status"] != "verified_native":
        reasons.append(f"native Chinese provenance is {group['native_status']}")
    if not group["retrieval_eligible"]:
        reasons.append("category is not a retrieval corpus")
    if counts.get("documents", 0) < MIN_DOCUMENTS:
        reasons.append(f"only {counts.get('documents', 0)} distinct retrieval documents; need {MIN_DOCUMENTS}")
    if counts.get("unique_source_documents", 0) < MIN_DOCUMENTS:
        reasons.append(
            f"only {counts.get('unique_source_documents', 0)} independent source documents; need {MIN_DOCUMENTS}"
        )
    if counts.get("queries", 0) < MIN_QUERIES:
        reasons.append(f"only {counts.get('queries', 0)} queries; need {MIN_QUERIES}")
    return not reasons, reasons


def _spec_for_group(group: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    category = str(group["category"])
    subcategory = group.get("subcategory")
    id_suffix = category if not subcategory else f"{category}_{subcategory}"
    source_id = f"gapfill_dqa_zh_{id_suffix}_20261031"
    return {
        "id": source_id,
        "connector": "generic_jsonl",
        "input_kind": "retrieval",
        "domain": "software_database",
        "genre": "DQA-Bench Chinese QA-to-retrieval adaptation",
        "language": "zh",
        "license": "DQA-Bench source terms unknown; retain official terms",
        "revision": f"DQA-Bench@{OFFICIAL_REVISION}",
        "metadata": {
            "source_family": "DQA-Bench",
            "source_category": category,
            "source_subcategory": subcategory,
            "source_repository": OFFICIAL_REPOSITORY,
            "translation_status": group["translation_status"],
            "native_status": group["native_status"],
            "annotation_source": group["annotation_source"],
            "grouping": "source_group is preserved in query/document metadata; categories and product/subtype releases are never merged",
            "qrels": "strict positive original row-to-answer or row-to-retrieval relation; no inferred negatives",
        },
        "options": {
            "dataset": "XMUDM/DQABench",
            "max_bytes": MAX_DOWNLOAD_BYTES,
            "files": {
                "documents": {"path": str(paths["documents"]), "name": "data/documents.jsonl"},
                "queries": {"path": str(paths["queries"]), "name": "data/queries.jsonl"},
                "qrels": {"path": str(paths["qrels"]), "name": "data/qrels.json"},
            },
        },
    }


def build_verified_release(group: Mapping[str, Any], *, raw_root: Path = RAW_ROOT, release_root: Path = RELEASE_ROOT) -> dict[str, Any]:
    """Build and re-load one eligible category through the generic pipeline."""
    _validate_positive_qrels(group)
    paths = _write_canonical_input(group, raw_root)
    spec = _spec_for_group(group, paths)
    category = str(group["category"])
    subcategory = group.get("subcategory")
    slug = category if not subcategory else f"{category}_{subcategory}"
    output = release_root / slug
    if output.exists():
        manifest = verify_release(output)
        bundle = load_bundle(output)
    else:
        manifest = build_source(spec, raw_root, output, policy=POLICY)
        verify_release(output)
        bundle = load_bundle(output)
    if {row["_id"] for row in bundle["documents"]} != set(group["documents"]):
        raise ValueError("published documents differ from conversion")
    if {row["_id"] for row in bundle["queries"]} != {row["_id"] for row in group["queries"]}:
        raise ValueError("published queries differ from conversion")
    if bundle["qrels"] != group["qrels"]:
        raise ValueError("published qrels differ from conversion")
    return {
        "release_path": str(output),
        "release_state": manifest["state"],
        "manifest": str(output / "manifest.json"),
        "bundle_hash": manifest["bundle_hash"],
        "verification": [
            "generic_jsonl build_source completed",
            "verify_release completed",
            "load_bundle round-trip IDs/text/qrels matched conversion",
            "all qrel grades were strict positive source relations",
        ],
    }


def _input_manifest(package_root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    output = []
    for path in files:
        digest, size = file_digest(path, max_bytes=MAX_DOWNLOAD_BYTES)
        output.append({
            "path": str(path),
            "sha256": digest,
            "bytes": size,
            "read_only_source": True,
        })
    return output


def run(
    *,
    package_root: Path | None = None,
    fetch_samples: bool = False,
    probe_drive_url: str = DRIVE_URL,
    raw_root: Path = RAW_ROOT,
    release_root: Path = RELEASE_ROOT,
    research_root: Path = RESEARCH_ROOT,
    receipt_filename: str | None = None,
) -> dict[str, Any]:
    raw_root.mkdir(parents=True, exist_ok=True)
    research_root.mkdir(parents=True, exist_ok=True)
    drive = probe_drive(probe_drive_url)
    public_manifest = fetch_public_samples(raw_root) if fetch_samples else None

    selected_root = package_root
    if selected_root is None and public_manifest and public_manifest["complete"]:
        selected_root = raw_root / "public_samples"
    groups: dict[str, dict[str, Any]] = {}
    package_files: list[Path] = []
    conversion_error: str | None = None
    if selected_root is not None:
        try:
            groups = convert_package(selected_root)
            package_files = discover_package_files(selected_root)
        except Exception as exc:
            conversion_error = f"{type(exc).__name__}: {exc}"

    category_receipts: list[dict[str, Any]] = []
    for category, group in sorted(groups.items()):
        ready, reasons = eligibility(group)
        receipt: dict[str, Any] = {
            "category": category,
            "subcategory": group.get("subcategory"),
            "status": "candidate_ready" if ready else "blocked",
            "counts": group["counts"],
            "native_status": group["native_status"],
            "translation_status": group["translation_status"],
            "annotation_source": group["annotation_source"],
            "retrieval_eligible_category": group["retrieval_eligible"],
            "audits": group["audits"],
            "blocking_reasons": reasons,
        }
        if ready:
            receipt["release"] = build_verified_release(group, raw_root=raw_root, release_root=release_root)
        category_receipts.append(receipt)

    overall_reasons: list[str] = []
    if drive.get("status") != "accessible_metadata":
        overall_reasons.append(f"official complete-package Drive link unavailable: HTTP {drive.get('http_status', 'error')}")
    if not groups:
        overall_reasons.append("no readable official ZH package or public sample files were supplied")
    if conversion_error:
        overall_reasons.append(f"package conversion failed: {conversion_error}")
    if not any(receipt["status"] == "candidate_ready" for receipt in category_receipts):
        overall_reasons.append("no category passed native provenance, independent-document, and query thresholds")

    receipt = {
        "schema_version": "dqa-acquisition-receipt-v1",
        "status": "ready" if not overall_reasons else "blocked",
        "dataset": "DQA-Bench",
        "language": "zh",
        "domain": "software_database",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_revision": OFFICIAL_REVISION,
        "official_drive_url": DRIVE_URL,
        "paper_url": PAPER_URL,
        "paper_revision": PAPER_REVISION,
        "paper_evidence": PAPER_EVIDENCE,
        "drive_probe": drive,
        "download_policy": {
            "first_download_limit_bytes": MAX_DOWNLOAD_BYTES,
            "raw_inputs_immutable": True,
            "repeated_downloads": "existing files are hashed and reused; differing bytes fail closed",
        },
        "raw_root": str(raw_root),
        "release_root": str(release_root),
        "research_root": str(research_root),
        "package_root": str(selected_root) if selected_root else None,
        "input_files": _input_manifest(selected_root, package_files) if selected_root and not conversion_error else [],
        "public_samples_manifest": str(raw_root / "public_samples_manifest.json") if public_manifest else None,
        "categories": category_receipts,
        "overall_blocking_reasons": overall_reasons,
        "source_evidence": {
            "directory_categories": ["forum_QA", "specific_product", "specific_instance", "multiple_choice"],
            "paper_language_boundary": "DQA paper describes bilingual pairs and English/Chinese translation; Chinese row-level native provenance is not exposed",
            "paper_generation_boundary": "general QA uses accepted/high-upvote online answers with GPT-4 rewriting; product and instance data use LLM generation; instance rows include tool/scenario traces",
            "public_sample_boundary": "product examples expose doc/retrieval; general and instance examples do not expose an independent source corpus relation",
        },
        "grouping_policy": "Each category and product/subtype is a separate candidate. source_group, source file, split and row are retained in query metadata; identical answer/passages keep aliases. No language, category, or duplicated QA padding is used.",
        "raw_to_release_checks": [
            "source row numbers and split are stored in query metadata",
            "source answers are retained in query metadata",
            "qrels use only strict positive row-to-source relations",
            "eligible releases use generic_jsonl build_source, verify_release and load_bundle round-trip checks",
        ],
    }
    receipt_name = receipt_filename or ("acquisition_receipt.json" if receipt["status"] == "ready" else "blocked_receipt.json")
    receipt["receipt_path"] = str(research_root / receipt_name)
    write_json_once(research_root / receipt_name, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, help="Read-only extracted DQA-Bench package root")
    parser.add_argument(
        "--materialize-fixed-samples-from",
        type=Path,
        help="Copy public sample files from a read-only fixed-commit checkout into the reviewed raw namespace",
    )
    parser.add_argument("--fetch-public-samples", action="store_true", help="Fetch only the pinned public ZH examples")
    parser.add_argument("--drive-url", default=DRIVE_URL, help=argparse.SUPPRESS)
    args = parser.parse_args()
    receipt_name = None
    package_root = args.package_root
    if args.materialize_fixed_samples_from is not None:
        materialized = materialize_fixed_samples(args.materialize_fixed_samples_from)
        package_root = Path(materialized["destination_root"])
        receipt_name = "reviewed_receipt.json"
    receipt = run(
        package_root=package_root,
        fetch_samples=args.fetch_public_samples,
        probe_drive_url=args.drive_url,
        receipt_filename=receipt_name,
    )
    print(json.dumps({
        "status": receipt["status"],
        "categories": [
            {"category": item["category"], "status": item["status"], "counts": item["counts"]}
            for item in receipt["categories"]
        ],
        "receipt": receipt["receipt_path"],
    }, ensure_ascii=False), flush=True)
    if receipt["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
