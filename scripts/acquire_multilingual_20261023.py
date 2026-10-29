#!/usr/bin/env python3
"""Acquire bounded multilingual retrieval releases without model inference.

The script deliberately downloads only the fixed resources in the companion
JSON configuration.  MLDR's dev files contain positive passages rather than a
separate qrels file; the qrels emitted here are a lossless, auditable
normalisation of those official positive passage IDs with score 1.0.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.connectors import fetch_source
from dci_bench.data.pipeline.core import build_source, code_identity, verify_release
from dci_bench.data.pipeline.download import download_url, file_digest


PYTHON = "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python"
DEFAULT_CONFIG = ROOT / "configs/data_pipeline/multilingual_20261023.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path: Path, value: Any) -> None:
    payload = canonical_json(value) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to overwrite existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"partial artifact already exists; inspect before retry: {temporary}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    if path.exists():
        digest, size = file_digest(path)
        return {"path": str(path), "sha256": digest, "bytes": size, "reused": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"partial artifact already exists; inspect before retry: {temporary}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest, size = file_digest(path)
    return {"path": str(path), "sha256": digest, "bytes": size, "rows": count, "reused": False}


def download_resources(source: Mapping[str, Any], source_inputs: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for resource in source["resources"]:
        name = Path(str(resource["url"]).split("?", 1)[0]).name
        destination = source_inputs / name
        digest, size = download_url(
            resource["url"],
            destination,
            timeout=120,
            retries=4,
            max_bytes=int(resource["bytes"]) + 1024 * 1024,
            expected_sha256=resource["sha256"],
            expected_bytes=int(resource["bytes"]),
            headers=resource.get("headers"),
            overwrite=False,
        )
        receipts.append({
            "role": resource["role"],
            "url": resource["url"],
            "path": str(destination),
            "bytes": size,
            "sha256": digest,
            "expected_bytes": int(resource["bytes"]),
            "expected_sha256": resource["sha256"],
        })
    return receipts


def stable_alias(receipts: list[dict[str, Any]], source_inputs: Path, role: str, name: str) -> Path:
    """Copy one downloaded source byte-for-byte to a stable raw-data name."""

    source_file = next(item["path"] for item in receipts if item["role"] == role)
    target = source_inputs / name
    expected = next(item for item in receipts if item["role"] == role)
    if target.exists():
        digest, size = file_digest(target)
        if digest != expected["sha256"] or size != expected["bytes"]:
            raise RuntimeError(f"stable source alias differs from downloaded resource: {target}")
    else:
        shutil.copyfile(source_file, target)
    return target


def iter_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield value


def _metadata(source: Mapping[str, Any], *, split: str, origin: str) -> dict[str, Any]:
    return {
        "translation_status": source["translation_status"],
        "native": source["translation_status"] == "native",
        "query_or_document_origin": origin,
        "source_split": split,
        "document_granularity": source["document_granularity"],
    }


def normalise_mldr(source: Mapping[str, Any], inputs: Path, canonical: Path) -> dict[str, Any]:
    language = source["language"]
    corpus_path = inputs / "corpus.jsonl.gz"
    dev_path = inputs / "dev.jsonl.gz"
    corpus_rows = list(iter_gzip_jsonl(corpus_path))
    dev_rows = list(iter_gzip_jsonl(dev_path))
    documents: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    for row in corpus_rows:
        docid = str(row.get("docid", ""))
        text = str(row.get("text", ""))
        if not docid or not text or docid in document_ids:
            raise ValueError(f"invalid or duplicate MLDR corpus row: {row!r}")
        document_ids.add(docid)
        documents.append({
            "_id": docid,
            "title": "",
            "text": text,
            "language": language,
            "metadata": _metadata(source, split="corpus", origin="official MLDR corpus row"),
        })
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, float]] = {}
    query_ids: set[str] = set()
    for row in dev_rows:
        query_id = str(row.get("query_id", ""))
        query = str(row.get("query", ""))
        if not query_id or not query or query_id in query_ids:
            raise ValueError(f"invalid or duplicate MLDR dev query: {row!r}")
        query_ids.add(query_id)
        positives = row.get("positive_passages") or []
        if not isinstance(positives, list) or not positives:
            raise ValueError(f"MLDR dev query has no positive_passages: {query_id}")
        labels: dict[str, float] = {}
        for positive in positives:
            docid = str(positive.get("docid", ""))
            if docid not in document_ids:
                raise ValueError(f"MLDR positive doc is absent from corpus: {query_id} -> {docid}")
            if docid in labels:
                raise ValueError(f"duplicate MLDR positive doc: {query_id} -> {docid}")
            labels[docid] = 1.0
        qrels[query_id] = labels
        queries.append({
            "_id": query_id,
            "text": query,
            "language": language,
            "metadata": _metadata(source, split="dev", origin=source["query_origin"]),
        })
    doc_receipt = write_jsonl_once(canonical / "documents.jsonl", documents)
    query_receipt = write_jsonl_once(canonical / "queries.jsonl", queries)
    qrel_path = canonical / "qrels.json"
    write_json_once(qrel_path, qrels)
    qrel_digest, qrel_bytes = file_digest(qrel_path)
    write_json_once(canonical / "normalization.json", {
        "schema_version": "dci-multilingual-normalization-v1",
        "source_id": source["id"],
        "source_format": "MLDR corpus.jsonl.gz + dev.jsonl.gz",
        "qrels_derivation": "one score-1.0 qrel for each official dev positive_passages.docid; no negative labels invented",
        "counts": {
            "documents": len(documents), "queries": len(queries),
            "qrels": sum(len(v) for v in qrels.values()),
        },
        "inputs": {"corpus": str(inputs / "corpus.jsonl.gz"), "dev": str(inputs / "dev.jsonl.gz")},
    })
    return {
        "documents": len(documents), "queries": len(queries),
        "qrels": sum(len(v) for v in qrels.values()),
        "canonical_files": [doc_receipt, query_receipt, {"path": str(qrel_path), "sha256": qrel_digest, "bytes": qrel_bytes}],
    }


def normalise_duretrieval(source: Mapping[str, Any], inputs: Path, canonical: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("DuRetrieval normalization requires the preinstalled pyarrow in dci-bench") from exc
    documents_table = parquet.read_table(inputs / "corpus.parquet")
    queries_table = parquet.read_table(inputs / "queries.parquet")
    qrels_table = parquet.read_table(inputs / "qrels.parquet")
    documents_rows = documents_table.to_pylist()
    queries_rows = queries_table.to_pylist()
    qrels_rows = qrels_table.to_pylist()
    document_ids = {str(row["id"]) for row in documents_rows}
    query_ids = {str(row["id"]) for row in queries_rows}
    if len(document_ids) != len(documents_rows) or len(query_ids) != len(queries_rows):
        raise ValueError("DuRetrieval contains duplicate document or query IDs")
    documents = [{
        "_id": str(row["id"]), "title": "", "text": str(row["text"]), "language": "zh",
        "metadata": _metadata(source, split="corpus", origin="C-MTEB/DuRetrieval parquet corpus"),
    } for row in documents_rows]
    queries = [{
        "_id": str(row["id"]), "text": str(row["text"]), "language": "zh",
        "metadata": _metadata(source, split="dev", origin=source["query_origin"]),
    } for row in queries_rows]
    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_rows:
        qid = str(row["qid"])
        pid = str(row["pid"])
        if qid not in query_ids or pid not in document_ids:
            raise ValueError(f"DuRetrieval qrel references missing ID: {qid} -> {pid}")
        if pid in qrels.setdefault(qid, {}):
            raise ValueError(f"duplicate DuRetrieval qrel: {qid} -> {pid}")
        qrels[qid][pid] = float(row["score"])
    doc_receipt = write_jsonl_once(canonical / "documents.jsonl", documents)
    query_receipt = write_jsonl_once(canonical / "queries.jsonl", queries)
    qrel_path = canonical / "qrels.json"
    write_json_once(qrel_path, qrels)
    qrel_digest, qrel_bytes = file_digest(qrel_path)
    write_json_once(canonical / "normalization.json", {
        "schema_version": "dci-multilingual-normalization-v1",
        "source_id": source["id"],
        "source_format": "C-MTEB/DuRetrieval parquet + C-MTEB/DuRetrieval-qrels parquet",
        "qrels_derivation": "copied qid/pid/score values from the official qrels parquet; scores are not replaced by retrieval scores",
        "counts": {
            "documents": len(documents), "queries": len(queries),
            "qrels": sum(len(v) for v in qrels.values()),
        },
        "inputs": {"corpus": str(inputs / "corpus.parquet"), "queries": str(inputs / "queries.parquet"), "qrels": str(inputs / "qrels.parquet")},
    })
    return {
        "documents": len(documents), "queries": len(queries),
        "qrels": sum(len(v) for v in qrels.values()),
        "canonical_files": [doc_receipt, query_receipt, {"path": str(qrel_path), "sha256": qrel_digest, "bytes": qrel_bytes}],
    }


def normalise_korquad(source: Mapping[str, Any], train_path: Path, dev_path: Path,
                      canonical: Path) -> dict[str, Any]:
    """Convert official KorQuAD v1 SQuAD JSON into an auditable retrieval bundle.

    The complete train and public dev paragraphs form the corpus.  Only the
    official public dev questions are queries (the hidden evaluation set is not
    available in the GitHub release).  A question remains linked to its source
    paragraph even when an optional answer span is malformed; such spans are
    retained in ``span_audit.json`` and omitted from evidence rather than
    dropping the question or inventing a negative label.
    """

    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, float]] = {}
    evidence: list[dict[str, Any]] = []
    span_audit: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    seen_document_ids: set[str] = set()
    split_counts: dict[str, dict[str, int]] = {}

    for split, path in (("train", train_path), ("dev", dev_path)):
        payload = read_json(path)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            raise ValueError(f"KorQuAD {split} JSON must contain a data list: {path}")
        split_document_count = 0
        split_query_count = 0
        split_span_count = 0
        for article_index, article in enumerate(payload["data"]):
            if not isinstance(article, Mapping):
                raise ValueError(f"KorQuAD {split} article {article_index} is not an object")
            title = str(article.get("title", ""))
            paragraphs = article.get("paragraphs")
            if not isinstance(paragraphs, list):
                raise ValueError(f"KorQuAD {split} article {article_index} has no paragraphs list")
            for paragraph_index, paragraph in enumerate(paragraphs):
                if not isinstance(paragraph, Mapping):
                    raise ValueError(f"KorQuAD {split} paragraph {article_index}/{paragraph_index} is not an object")
                context = paragraph.get("context")
                if not isinstance(context, str) or not context.strip():
                    raise ValueError(f"KorQuAD {split} paragraph {article_index}/{paragraph_index} is empty")
                document_id = f"korquad-v1:{split}:{article_index:05d}:{paragraph_index:04d}"
                if document_id in seen_document_ids:
                    raise ValueError(f"duplicate derived KorQuAD document ID: {document_id}")
                seen_document_ids.add(document_id)
                documents.append({
                    "_id": document_id,
                    "title": title,
                    "text": context,
                    "language": source["language"],
                    "metadata": {
                        "source_split": split,
                        "source_article_index": article_index,
                        "source_paragraph_index": paragraph_index,
                        "source_title": title,
                        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                        "annotation_source": "source_human",
                        "translation_status": source["translation_status"],
                        "native": True,
                        "document_granularity": source["document_granularity"],
                    },
                })
                split_document_count += 1
                qas = paragraph.get("qas", [])
                if not isinstance(qas, list):
                    raise ValueError(f"KorQuAD {split} paragraph {article_index}/{paragraph_index} qas is not a list")
                for qa_index, qa in enumerate(qas):
                    if not isinstance(qa, Mapping):
                        raise ValueError(f"KorQuAD {split} QA {article_index}/{paragraph_index}/{qa_index} is not an object")
                    query_id = qa.get("id")
                    question = qa.get("question")
                    if not isinstance(query_id, str) or not query_id or query_id in seen_query_ids:
                        raise ValueError(f"invalid or duplicate KorQuAD query ID: {query_id!r}")
                    seen_query_ids.add(query_id)
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError(f"KorQuAD query has empty question: {query_id}")
                    if split == "dev":
                        queries.append({
                            "_id": query_id,
                            "text": question,
                            "language": source["language"],
                            "metadata": {
                                "source_split": "dev",
                                "evaluation_split": "official_public_dev",
                                "source_article_index": article_index,
                                "source_paragraph_index": paragraph_index,
                                "source_title": title,
                                "annotation_source": "source_human",
                                "translation_status": source["translation_status"],
                                "native": True,
                            },
                        })
                        qrels[query_id] = {document_id: 1.0}
                        split_query_count += 1
                    answers = qa.get("answers", [])
                    if answers is None:
                        answers = []
                    if not isinstance(answers, list):
                        raise ValueError(f"KorQuAD answers is not a list: {query_id}")
                    for answer_index, answer in enumerate(answers):
                        if not isinstance(answer, Mapping):
                            span_audit.append({
                                "query_id": query_id, "source_split": split,
                                "article_index": article_index, "paragraph_index": paragraph_index,
                                "answer_index": answer_index, "reason": "answer_not_object",
                            })
                            continue
                        quote = answer.get("text")
                        start = answer.get("answer_start")
                        valid = (
                            isinstance(quote, str) and bool(quote)
                            and isinstance(start, int) and not isinstance(start, bool)
                            and start >= 0 and context[start:start + len(quote)] == quote
                        )
                        if not valid:
                            span_audit.append({
                                "query_id": query_id, "source_split": split,
                                "article_index": article_index, "paragraph_index": paragraph_index,
                                "answer_index": answer_index, "answer": dict(answer),
                                "reason": "offset_or_quote_mismatch",
                            })
                            continue
                        span_counted = split == "dev"
                        if span_counted:
                            evidence.append({
                                "query_id": query_id, "doc_id": document_id,
                                "quote": quote, "start": start, "end": start + len(quote),
                                "annotation_source": "source_human",
                                "offset_unit": "unicode_codepoints",
                                "source_split": split,
                            })
                            split_span_count += 1

        split_counts[split] = {
            "documents": split_document_count,
            "queries": split_query_count,
            "valid_dev_evidence": split_span_count,
        }

    doc_receipt = write_jsonl_once(canonical / "documents.jsonl", documents)
    query_receipt = write_jsonl_once(canonical / "queries.jsonl", queries)
    qrel_path = canonical / "qrels.json"
    write_json_once(qrel_path, qrels)
    qrel_digest, qrel_bytes = file_digest(qrel_path)
    evidence_receipt = write_jsonl_once(canonical / "evidence.jsonl", evidence)
    write_json_once(canonical / "span_audit.json", {
        "schema_version": "korquad-span-audit-v1",
        "source": "official KorQuAD v1 SQuAD answers",
        "algorithm": "validate answer_start and exact Unicode-codepoint context slice; retain invalid rows here and omit them from evidence",
        "invalid_span_count": len(span_audit),
        "invalid_spans": span_audit,
    })
    write_json_once(canonical / "normalization.json", {
        "schema_version": "dci-multilingual-normalization-v1",
        "source_id": source["id"],
        "source_format": "official KorQuAD v1 SQuAD train + public dev JSON",
        "corpus_policy": "all train and public dev paragraphs; no truncation",
        "query_policy": "official public dev questions only; hidden test split is not present in the GitHub release",
        "qrels_derivation": "one score-1.0 relation from each retained official dev question to its source paragraph; no negative labels invented",
        "counts": {
            "documents": len(documents), "queries": len(queries),
            "qrels": sum(len(v) for v in qrels.values()),
            "valid_evidence": len(evidence), "invalid_spans": len(span_audit),
        },
        "split_counts": split_counts,
        "inputs": {"train": str(train_path), "dev": str(dev_path)},
    })
    return {
        "documents": len(documents), "queries": len(queries),
        "qrels": sum(len(v) for v in qrels.values()),
        "valid_evidence": len(evidence), "invalid_spans": len(span_audit),
        "split_counts": split_counts,
        "canonical_files": [doc_receipt, query_receipt,
                            {"path": str(qrel_path), "sha256": qrel_digest, "bytes": qrel_bytes},
                            evidence_receipt],
    }


def prepare_cmrc_squad(source_path: Path, target_path: Path) -> dict[str, Any]:
    """Validate CMRC spans and write a deterministic adapter input.

    The official SQuAD-style file has 21 answer offsets that do not point at
    their answer text (including a few one-character annotation slips).  Keep
    the downloaded file immutable and correct only those offsets in a derived
    adapter input so the existing squad connector can validate every span.
    """

    data = read_json(source_path)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("CMRC SQuAD source must contain a data list")
    corrections: list[dict[str, Any]] = []
    for article_index, article in enumerate(data["data"]):
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                for answer_index, answer in enumerate(qa.get("answers", [])):
                    quote = answer.get("text", "")
                    declared = answer.get("answer_start")
                    actual = context.find(quote)
                    if actual < 0:
                        raise ValueError(f"CMRC answer text is absent from context: {qa.get('id')}")
                    if declared != actual:
                        corrections.append({
                            "query_id": qa.get("id"), "article_index": article_index,
                            "paragraph_index": paragraph_index, "answer_index": answer_index,
                            "declared_start": declared, "corrected_start": actual,
                        })
                        answer["answer_start"] = actual
    write_json_once(target_path, data)
    correction_path = target_path.with_name("cmrc2018_offset_corrections.json")
    write_json_once(correction_path, {
        "schema_version": "cmrc2018-offset-correction-v1",
        "source": str(source_path), "adapter_input": str(target_path),
        "algorithm": "context.find(answer.text); reject absent text; preserve IDs/text",
        "corrections": corrections,
        "count": len(corrections),
    })
    digest, size = file_digest(target_path)
    return {"path": str(target_path), "sha256": digest, "bytes": size, "corrections": len(corrections)}


def source_spec(source: Mapping[str, Any], canonical: Path) -> dict[str, Any]:
    source_format = source.get("format")
    if source_format == "korquad_squad":
        return {
            "id": source["id"], "connector": "generic_jsonl", "input_kind": "evidence_qa",
            "domain": "encyclopedia", "language": source["language"],
            "revision": source["revision"], "license": source["license"],
            "options": {
                "dataset": source["dataset"],
                "files": {
                    "documents": {"path": str(canonical / "documents.jsonl"), "name": "documents.jsonl"},
                    "queries": {"path": str(canonical / "queries.jsonl"), "name": "queries.jsonl"},
                    "qrels": {"path": str(canonical / "qrels.json"), "name": "qrels.json"},
                    "evidence": {"path": str(canonical / "evidence.jsonl"), "name": "evidence.jsonl"},
                },
                "genre": "wikipedia_paragraph_qa_retrieval",
                "document_granularity": source["document_granularity"],
                "translation_status": source["translation_status"],
                "split": "official_public_dev",
            },
            "metadata": {
                "source_scope": source["query_origin"],
                "document_granularity": source["document_granularity"],
                "task_adaptation": "qa-derived; official dev questions to source-paragraph seed qrels; unjudged passages are not negatives",
                "normalization": "full train+dev corpus with official public dev queries; malformed optional spans audited separately",
            },
        }
    if source_format == "cmrc_squad":
        squad_path = Path(str(source["adapter_path"]))
        return {
            "id": source["id"], "connector": "squad", "input_kind": "evidence_qa",
            "domain": "encyclopedia", "language": source["language"],
            "revision": source["revision"], "license": source["license"],
            "options": {
                "files": [{
                    "path": str(squad_path),
                    "name": "data/cmrc2018_train.json",
                    "sha256": source["adapter_sha256"],
                    "bytes": source["adapter_bytes"],
                }],
                "genre": "wikipedia_paragraph_qa",
                "document_granularity": source["document_granularity"],
                "translation_status": source["translation_status"],
                "task_adaptation": "QA evidence to seed passage-retrieval qrels",
            },
            "metadata": {
                "source_scope": source["query_origin"],
                "document_granularity": source["document_granularity"],
                "task_adaptation": "qa-derived; seed positives only; unjudged passages are not negatives",
                "normalization": "official SQuAD-style JSON consumed by the existing squad adapter",
                "offset_corrections": source.get("offset_corrections", 0),
            },
        }
    if source_format == "vihealthqa_beir":
        archive_path = Path(str(source["archive_path"]))
        return {
            "id": source["id"], "connector": "beir", "input_kind": "retrieval",
            "domain": "health", "language": source["language"],
            "revision": source["revision"], "license": source["license"],
            "options": {
                "archive": str(archive_path), "archive_sha256": source["archive_sha256"],
                "archive_bytes": source["archive_bytes"], "split": source["split"],
                "genre": "health_answer_passage_retrieval",
                "document_granularity": source["document_granularity"],
                "translation_status": source["translation_status"],
            },
            "metadata": {
                "source_scope": source["query_origin"],
                "document_granularity": source["document_granularity"],
                "normalization": "official BEIR JSONL/TSV archive consumed by the existing beir adapter",
            },
        }
    return {
        "id": source["id"], "connector": "generic_jsonl", "input_kind": "retrieval",
        "domain": "web_search" if source["id"].endswith("duretrieval_zh") else "encyclopaedic",
        "language": source["language"], "revision": source["revision"], "license": source["license"],
        "options": {
            "files": {
                "documents": {"path": str(canonical / "documents.jsonl"), "name": "documents.jsonl"},
                "queries": {"path": str(canonical / "queries.jsonl"), "name": "queries.jsonl"},
                "qrels": {"path": str(canonical / "qrels.json"), "name": "qrels.json"},
            },
            "genre": "passage_retrieval",
            "document_granularity": source["document_granularity"],
            "translation_status": source["translation_status"],
        },
        "metadata": {
            "source_scope": source["query_origin"],
            "document_granularity": source["document_granularity"],
            "normalization": "canonical JSONL/qrels generated by acquire_multilingual_20261023.py",
        },
    }


def acquire_one(source: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    raw_root = Path(config["raw_root"])
    eval_root = Path(config["eval_root"])
    source_root = raw_root / source["id"]
    inputs = source_root / "source_inputs"
    canonical = source_root / "canonical_input"
    inputs.mkdir(parents=True, exist_ok=True)
    canonical.mkdir(parents=True, exist_ok=True)
    receipts = download_resources(source, inputs)
    source_format = source.get("format")
    if source_format == "cmrc_squad":
        squad_path = stable_alias(receipts, inputs, "squad", "cmrc2018_train.json")
        adapter_path = inputs / "cmrc2018_train_adapter.json"
        adapter_receipt = prepare_cmrc_squad(squad_path, adapter_path)
        source_for_spec = dict(source)
        source_for_spec["squad_path"] = squad_path
        source_for_spec["adapter_path"] = adapter_path
        source_for_spec["adapter_sha256"] = adapter_receipt["sha256"]
        source_for_spec["adapter_bytes"] = adapter_receipt["bytes"]
        source_for_spec["offset_corrections"] = adapter_receipt["corrections"]
        # Keep the unmodified CMRC list-form file beside the SQuAD-style file.
        # It contains the official context_id values that the adapter's
        # content-derived passage IDs are mapped back to in the research record.
        native_path = stable_alias(receipts, inputs, "native", "cmrc2018_train_native.json")
        source_for_spec["native_path"] = native_path
        counts = {
            "documents": int(source["counts"]["documents"]),
            "queries": int(source["counts"]["queries"]),
            "qrels": int(source["counts"]["qrels"]),
            "native_context_ids": int(source["counts"]["documents"]),
        }
    elif source_format == "korquad_squad":
        train_path = stable_alias(receipts, inputs, "train", "KorQuAD_v1.0_train.json")
        dev_path = stable_alias(receipts, inputs, "dev", "KorQuAD_v1.0_dev.json")
        counts = normalise_korquad(source, train_path, dev_path, canonical)
        source_for_spec = dict(source)
    elif source_format == "vihealthqa_beir":
        archive_path = stable_alias(receipts, inputs, "archive", "vihealthqa.zip")
        source_for_spec = dict(source)
        source_for_spec["archive_path"] = archive_path
        archive_digest, archive_bytes = file_digest(archive_path)
        source_for_spec["archive_sha256"] = archive_digest
        source_for_spec["archive_bytes"] = archive_bytes
        counts = {
            "documents": int(source["counts"]["documents"]),
            "queries": int(source["counts"]["queries"]),
            "qrels": int(source["counts"]["qrels"]),
        }
    elif source["id"].endswith("duretrieval_zh"):
        # Keep stable local names independent of the HF parquet shard names.
        for role, name in (("documents", "corpus.parquet"), ("queries", "queries.parquet"), ("qrels", "qrels.parquet")):
            source_file = next(item["path"] for item in receipts if item["role"] == role)
            target = inputs / name
            if target.exists():
                digest, size = file_digest(target)
                expected = next(item for item in receipts if item["role"] == role)
                if digest != expected["sha256"] or size != expected["bytes"]:
                    raise RuntimeError(f"stable source alias differs from downloaded resource: {target}")
            else:
                shutil.copyfile(source_file, target)
        counts = normalise_duretrieval(source, inputs, canonical)
    else:
        # Stable aliases make the normalizer independent of the language path.
        for role, name in (("documents", "corpus.jsonl.gz"), ("queries", "dev.jsonl.gz")):
            source_file = next(item["path"] for item in receipts if item["role"] == role)
            target = inputs / name
            if target.exists():
                if file_digest(target) != file_digest(Path(source_file)):
                    raise RuntimeError(f"stable source alias differs from downloaded resource: {target}")
            else:
                shutil.copyfile(source_file, target)
        counts = normalise_mldr(source, inputs, canonical)
    spec = source_spec(source_for_spec if source_format in {"cmrc_squad", "korquad_squad", "vihealthqa_beir"} else source, canonical)
    snapshot = fetch_source(spec, raw_root)
    release = eval_root / "pipeline_v1" / source["id"] / source["release_name"]
    if release.exists():
        manifest = verify_release(release)
    else:
        build_source(spec, raw_root, release, snapshot=snapshot, policy=source.get("quality_policy"))
        manifest = verify_release(release)
    if manifest["state"] not in {"labels_ready", "review_required"}:
        raise RuntimeError(f"release did not become a verifiable labeled/review release: {release} ({manifest['state']})")
    return {
        "id": source["id"], "language": source["language"], "status": "fetched_verified",
        "revision": source["revision"], "license": source["license"],
        "translation_status": source["translation_status"],
        "query_origin": source["query_origin"], "document_granularity": source["document_granularity"],
        "counts": {key: manifest[key] for key in ("num_documents", "num_queries", "num_positive_pairs")},
        "source_declared_counts": source.get("counts", {}),
        "limitations": source.get("limitations", ""),
        "normalization": (
            {"adapter_input": str(source_for_spec.get("adapter_path")),
             "offset_corrections": source_for_spec.get("offset_corrections", 0)}
            if source_format == "cmrc_squad" else
            {"canonical_input": str(canonical),
             "invalid_spans_audit": str(canonical / "span_audit.json"),
             "query_split": "official_public_dev"}
            if source_format == "korquad_squad" else {}
        ),
        "normalization_counts": counts,
        "raw_source_inputs": str(inputs), "raw_snapshot": str(snapshot), "release_path": str(release),
        "release_manifest_sha256": sha256_json(manifest), "release_state": manifest["state"],
        "verification": {
            "standard_verify_release": True,
            "release_manifest_bundle_hash": manifest["bundle_hash"],
            "validation_boundary": manifest.get("validation_boundary"),
        },
        "source_files": receipts,
        "task_adaptation": source.get("task_adaptation", "native retrieval"),
        "id_preservation": (
            "query IDs and raw context_id file preserved; release passage IDs are deterministic content hashes"
            if source_format == "cmrc_squad" else
            "official KorQuAD QA IDs preserved; document IDs deterministically encode source split/article/paragraph"
            if source_format == "korquad_squad" else "official corpus/query/qrels IDs copied by BEIR adapter"
        ),
        "pipeline_code_identity": code_identity(),
    }


def build_research(config: Mapping[str, Any], released: list[dict[str, Any]]) -> dict[str, Any]:
    raw_bytes = sum(sum(int(item["bytes"]) for item in row["source_files"]) for row in released)
    source_ids = " ".join(str(row["id"]) for row in released)
    final_research_path = Path(config["research_root"]) / "research_final.json"
    return {
        "schema_version": "dci-multilingual-research-v1",
        "created_at": "2026-10-23",
        "reproduction": {
            "python": PYTHON,
            "command": f"{PYTHON} scripts/acquire_multilingual_20261023.py --config configs/data_pipeline/multilingual_20261023.json --source {source_ids} --research-output {final_research_path}",
            "network": "dci_bench.data.pipeline.download.download_url direct opener; no proxy and no model/GPU",
            "verify_command": "{python} scripts/data_pipeline.py --config configs/data_pipeline/multilingual_20261023.json verify <release_path>",
            "verify_commands": [
                f"{PYTHON} scripts/data_pipeline.py verify {row['release_path']}"
                for row in released
            ],
        },
        "budget": {
            "limit_bytes": config["download_budget_bytes"],
            "released_source_bytes": raw_bytes,
            "released_source_mib": round(raw_bytes / (1024 * 1024), 3),
            "within_limit": raw_bytes <= int(config["download_budget_bytes"]),
            "per_source_limit_bytes": config["per_source_budget_bytes"],
        },
        "sources": released + list(config.get("researched_not_downloaded", [])),
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", nargs="*", help="subset of configured source IDs; default is all downloadable sources")
    parser.add_argument("--research-only", action="store_true", help="write a candidate research record without downloading")
    parser.add_argument("--research-output", type=Path,
                        help="write a separate research record (for example research_final.json) without replacing research.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = read_json(args.config)
    selected = set(args.source or [source["id"] for source in config["sources"]])
    known = {source["id"] for source in config["sources"]}
    unknown = selected - known
    if unknown:
        raise SystemExit(f"unknown sources: {sorted(unknown)}")
    released: list[dict[str, Any]] = []
    if not args.research_only:
        for source in config["sources"]:
            if source["id"] in selected:
                released.append(acquire_one(source, config))
    research_path = args.research_output or (Path(config["research_root"]) / "research.json")
    research = build_research(config, released)
    if research_path.exists():
        existing = read_json(research_path)
        if existing != research:
            raise RuntimeError(f"refusing to overwrite existing research record: {research_path}")
    else:
        write_json_once(research_path, research)
    print(json.dumps({"research": str(research_path), "released": released}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
