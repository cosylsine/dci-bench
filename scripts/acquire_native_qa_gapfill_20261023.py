#!/usr/bin/env python3
"""Acquire native-language QA sources and publish paragraph-retrieval views.

The source datasets are native extractive-QA resources, rather than existing
information-retrieval benchmarks.  This adapter keeps the complete source
train+dev context collection (exact-text deduplication with aliases), retains
only official dev questions as held-out queries, and derives one positive
qrel from the source question-to-context association.  Answer spans are
optional evidence: malformed offsets are audited and never used to remove a
question or to invent a label.

This script owns only the ``gapfill_nativeqa_*`` namespace.  Generic source
snapshotting, bundle validation, quality auditing, and immutable release
publication remain owned by the data pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Any, Mapping
import urllib.request
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dci_bench.data.pipeline.connectors import fetch_source  # noqa: E402
from dci_bench.data.pipeline.core import build_source, verify_release  # noqa: E402
from dci_bench.data.pipeline.download import download_url, file_digest  # noqa: E402


BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
RAW_ROOT = BASE / "raw_data"
EVAL_ROOT = BASE / "eval_data"
RESEARCH_ROOT = EVAL_ROOT / "source_research_20261023_round3" / "native_qa"

QUALITY_POLICY = {
    "max_positive_per_query": None,
    "exclude_review_required": False,
    "drop_near_duplicates": False,
}

LANGUAGES = ("ru", "th", "tr")

RU_ARCHIVE_URL = "https://files.deeppavlov.ai/datasets/sber_squad_clean-v1.1.tar.gz"
RU_ARCHIVE_BYTES = 22_766_184
RU_ARCHIVE_SHA256 = "5f5ba3e1a9344e72a60b701f8c30b52536df860810654a899dcf441714d0d40a"
RU_REVISION = f"sber_squad_clean-v1.1.tar.gz#sha256:{RU_ARCHIVE_SHA256}"
RU_SOURCE_FILES = ("train-v1.1.json", "dev-v1.1.json")

TH_REPO = "PyThaiNLP/thaiqa_squad"
TH_REVISION = "61a4c4f36babcad721bb6bf9561fff4416a775ae"
TH_ARCHIVE_URL = f"https://raw.githubusercontent.com/{TH_REPO}/{TH_REVISION}/data.zip"
TH_ARCHIVE_BYTES = 10_003_354
TH_ARCHIVE_SHA256 = "6aa58c8a9b52862de166618b9b682d7094f56a1587f9681f3dd617bfd78f95c8"
TH_SOURCE_FILES = {
    "train": "data/train.jsonl",
    "dev": "data/dev.jsonl",
}

TR_REPO = "TQuad/turkish-nlp-qa-dataset"
TR_REVISION = "3bece60f62569182476fe00683793745ba92c32e"
TR_FILES = {
    "train": {
        "name": "train-v0.1.json",
        "url": f"https://raw.githubusercontent.com/{TR_REPO}/{TR_REVISION}/train-v0.1.json",
        "bytes": 3_078_146,
        "sha256": "a1033c237665c680cdf1e2071b39c9c1ca3d4287e2de460530abe519c5f70ddc",
    },
    "dev": {
        "name": "dev-v0.1.json",
        "url": f"https://raw.githubusercontent.com/{TR_REPO}/{TR_REVISION}/dev-v0.1.json",
        "bytes": 458_494,
        "sha256": "1a42c8b00222db18331cbaa5b20d3f21c29196c46892e9602cc9a2a9b6b819b2",
    },
}

SOURCE_INFO: dict[str, dict[str, Any]] = {
    "ru": {
        "source_family": "SberQuAD",
        "dataset": "DeepPavlov/SberQuADClean",
        "revision": RU_REVISION,
        "language_name": "Russian",
        "license": "unknown; the accessed DeepPavlov configuration and SberQuAD archive do not state a dataset license",
        "official_sources": {
            "archive": RU_ARCHIVE_URL,
            "deeppavlov_config": "https://github.com/deeppavlov/DeepPavlov/blob/5f9fbed0c7191466bc7621e604b810f66f254c03/deeppavlov/configs/squad/squad_ru_bert.json",
            "documentation": "https://docs.deeppavlov.ai/en/master/features/models/SQuAD.html",
            "paper": "https://arxiv.org/abs/1912.09723",
        },
        "native_evidence": "SberQuAD is a Russian reading-comprehension dataset; questions and contexts are retained in source Russian text.",
    },
    "th": {
        "source_family": "ThaiQA",
        "dataset": TH_REPO,
        "revision": TH_REVISION,
        "language_name": "Thai",
        "license": "CC-BY-NC-SA-3.0",
        "official_sources": {
            "repository": f"https://github.com/{TH_REPO}",
            "readme_raw": f"https://raw.githubusercontent.com/{TH_REPO}/{TH_REVISION}/README.md",
            "archive": TH_ARCHIVE_URL,
            "original_nectec": "https://aiforthai.in.th/corpus.php",
            "thai_qa_task": "http://copycatch.in.th/thai-qa-task.html",
            "nectec": "https://www.nectec.or.th/en/",
        },
        "native_evidence": "The pinned dataset card states that NECTEC created Thai questions and answer annotations from Thai Wikipedia contexts; PyThaiNLP adapted the source to SQuAD format.",
    },
    "tr": {
        "source_family": "TQuAD",
        "dataset": TR_REPO,
        "revision": TR_REVISION,
        "language_name": "Turkish",
        "license": "unknown; the pinned repository has no explicit data license file",
        "official_sources": {
            "repository": f"https://github.com/{TR_REPO}",
            "readme_raw": f"https://raw.githubusercontent.com/{TR_REPO}/{TR_REVISION}/README.md",
            "train": TR_FILES["train"]["url"],
            "dev": TR_FILES["dev"]["url"],
        },
        "native_evidence": "The pinned repository describes a Turkish question-answer dataset created for the Teknofest 2018 Turkish and Islamic science history task; source questions and contexts are retained in Turkish.",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _write_bytes_once(path: Path, payload: bytes) -> str:
    """Write an immutable regular file and preserve conflicting partials."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"immutable destination is not a regular file: {path}")
        if path.read_bytes() != payload:
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
    return _write_bytes_once(Path(path), _canonical_json(value).encode("utf-8"))


def _write_jsonl_once(path: Path, rows: list[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    return _write_bytes_once(Path(path), payload)


def _download_once(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Download a pinned source with proxy-aware bounded I/O.

    The pipeline's downloader keeps a ``.partial`` recovery artifact.  The
    opener intentionally follows the configured environment proxy because the
    source hosts are reachable through that path in this deployment.
    """

    if destination.exists():
        digest, size = file_digest(destination, max_bytes=max_bytes)
        if digest != expected_sha256 or size != expected_bytes:
            raise RuntimeError(
                f"existing source failed pinned integrity: {destination} "
                f"sha256={digest} bytes={size}"
            )
        status = "reused_verified"
    else:
        digest, size = download_url(
            url,
            destination,
            timeout=60,
            retries=2,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
            overwrite=False,
            opener=urllib.request.build_opener(),
        )
        status = "fetched_verified"
    return {
        "url": url,
        "path": str(destination),
        "status": status,
        "bytes": expected_bytes,
        "sha256": expected_sha256,
    }


def _read_squad_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid SQuAD JSON: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        raise RuntimeError(f"SQuAD source has no data list: {path}")
    return value


def _source_alias(
    *,
    source_file: str,
    split: str,
    article_index: int,
    paragraph_index: int,
    title: str,
    source_context_id: Any = None,
) -> dict[str, Any]:
    alias = {
        "source_file": source_file,
        "source_split": split,
        "source_article_index": article_index,
        "source_paragraph_index": paragraph_index,
        "source_title": title,
    }
    if source_context_id is not None:
        alias["source_context_id"] = str(source_context_id)
    return alias


def _document_for_context(
    *,
    language: str,
    context: str,
    title: str,
    family: str,
    alias: dict[str, Any],
    source_context_id: Any = None,
    documents_by_key: dict[str, dict[str, Any]],
) -> tuple[str, bool]:
    """Insert one exact context and return ``(document_id, was_new)``."""

    text_key = _sha256_text(language + "\0" + context)
    document_id = f"nativeqa-{language}-context-{text_key[:32]}"
    existing = documents_by_key.get(text_key)
    if existing is None:
        metadata = {
            "source_family": family,
            "source_context_id": str(source_context_id) if source_context_id is not None else None,
            "source_aliases": [alias],
            "source_split": alias["source_split"],
            "source_file": alias["source_file"],
            "source_title": title,
            "source_language": language,
            "corpus_language": language,
            "corpus_origin": "Wikipedia",
            "corpus_granularity": "source QA context treated as a retrieval paragraph",
            "translation_status": "native",
            "id_derivation": "sha256(language + NUL + exact_source_context) first 32 hex",
            "exact_text_deduplication": "only byte-identical Unicode text; all source aliases retained",
        }
        documents_by_key[text_key] = {
            "_id": document_id,
            "title": title,
            "text": context,
            "language": language,
            "metadata": metadata,
        }
        return document_id, True
    existing_aliases = existing["metadata"].setdefault("source_aliases", [])
    if alias not in existing_aliases:
        existing_aliases.append(alias)
    return document_id, False


def _validate_squad_answer(context: str, answer: Mapping[str, Any]) -> tuple[bool, int | None, str | None, str]:
    text = answer.get("text")
    raw_start = answer.get("answer_start")
    start: int | None
    if isinstance(raw_start, int) and not isinstance(raw_start, bool):
        start = raw_start
    elif isinstance(raw_start, str) and raw_start.strip().isdigit():
        # TQuAD stores answer_start as a decimal string.  The original value is
        # retained in span_audit; conversion is only for optional validation.
        start = int(raw_start.strip())
    else:
        start = None
    valid = (
        isinstance(text, str)
        and bool(text)
        and start is not None
        and start >= 0
        and start + len(text) <= len(context)
        and context[start : start + len(text)] == text
    )
    if valid:
        return True, start, text, "official_answer_span_validated"
    return False, start, text if isinstance(text, str) else None, "invalid_original_optional_answer_offset"


def _parse_squad_language(
    language: str,
    *,
    family: str,
    source_dir: Path,
    split_files: Mapping[str, str],
) -> dict[str, Any]:
    parsed = {split: _read_squad_json(source_dir / name) for split, name in split_files.items()}
    documents_by_key: dict[str, dict[str, Any]] = {}
    positions: dict[tuple[str, int, int], str] = {}
    source_paragraphs = 0
    split_stats: dict[str, dict[str, int]] = {}

    for split in ("train", "dev"):
        data = parsed[split].get("data", [])
        split_stats[split] = {"articles": len(data), "source_paragraphs": 0, "source_queries": 0}
        source_file = split_files[split]
        for article_index, article in enumerate(data):
            if not isinstance(article, dict) or not isinstance(article.get("paragraphs"), list):
                raise RuntimeError(f"invalid article/paragraph list in {source_dir / source_file}:{article_index}")
            title = article.get("title", "")
            if not isinstance(title, str):
                raise RuntimeError(f"non-string title in {source_dir / source_file}:{article_index}")
            for paragraph_index, paragraph in enumerate(article["paragraphs"]):
                if not isinstance(paragraph, dict) or not isinstance(paragraph.get("context"), str):
                    raise RuntimeError(
                        f"invalid context in {source_dir / source_file}:{article_index}:{paragraph_index}"
                    )
                context = paragraph["context"]
                if not context:
                    raise RuntimeError(f"empty context in {source_dir / source_file}:{article_index}:{paragraph_index}")
                alias = _source_alias(
                    source_file=source_file,
                    split=split,
                    article_index=article_index,
                    paragraph_index=paragraph_index,
                    title=title,
                )
                doc_id, _ = _document_for_context(
                    language=language,
                    context=context,
                    title=title,
                    family=family,
                    alias=alias,
                    documents_by_key=documents_by_key,
                )
                positions[(split, article_index, paragraph_index)] = doc_id
                source_paragraphs += 1
                split_stats[split]["source_paragraphs"] += 1

                qas = paragraph.get("qas", [])
                if not isinstance(qas, list):
                    raise RuntimeError(f"paragraph qas must be a list in {source_dir / source_file}:{article_index}:{paragraph_index}")
                split_stats[split]["source_queries"] += len(qas)

    documents = sorted(documents_by_key.values(), key=lambda row: row["_id"])
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, float]] = {}
    evidence: list[dict[str, Any]] = []
    span_audit: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    seen_evidence: set[tuple[str, str, str, int, int]] = set()
    answers_total = 0
    valid_answers = 0
    duplicate_answers = 0

    # Only dev is held out.  Training questions are counted as source QA but
    # never emitted as retrieval queries.
    for article_index, article in enumerate(parsed["dev"]["data"]):
        title = str(article.get("title") or "")
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            context = paragraph["context"]
            doc_id = positions[("dev", article_index, paragraph_index)]
            for qa_index, qa in enumerate(paragraph.get("qas", [])):
                if not isinstance(qa, dict):
                    raise RuntimeError(f"invalid dev QA at {source_dir / split_files['dev']}:{article_index}:{paragraph_index}:{qa_index}")
                raw_qid = qa.get("id")
                qid = str(raw_qid) if raw_qid is not None else ""
                question = qa.get("question")
                if not qid or not isinstance(question, str) or not question:
                    raise RuntimeError(f"dev QA missing ID/question at {source_dir / split_files['dev']}:{article_index}:{paragraph_index}:{qa_index}")
                if qid in seen_queries:
                    raise RuntimeError(f"duplicate official dev query ID {qid!r} in {source_dir / split_files['dev']}")
                seen_queries.add(qid)
                query_metadata = {
                    "source_family": family,
                    "source_scope": "complete train+dev source contexts; official dev queries only",
                    "task_origin": "source extractive QA converted to paragraph retrieval",
                    "source_query_id": qid,
                    "source_query_id_raw": raw_qid,
                    "source_split": "dev",
                    "source_file": split_files["dev"],
                    "source_article_index": article_index,
                    "source_paragraph_index": paragraph_index,
                    "source_title": title,
                    "source_context_id": doc_id,
                    "source_question_language": language,
                    "source_context_language": language,
                    "translation_status": "native",
                    "translation_status_evidence": "source repository/dataset describes native-language QA; no translation performed by this adapter",
                    "qa_granularity": "source QA context / retrieval paragraph",
                    "annotation_source": "source_provided",
                }
                queries.append({"_id": qid, "text": question, "language": language, "metadata": query_metadata})
                qrels[qid] = {doc_id: 1.0}

                answers = qa.get("answers", [])
                if not isinstance(answers, list):
                    span_audit.append({
                        "query_id": qid,
                        "doc_id": doc_id,
                        "status": "answers_not_a_list",
                        "source_file": split_files["dev"],
                    })
                    continue
                if not answers:
                    span_audit.append({
                        "query_id": qid,
                        "doc_id": doc_id,
                        "status": "missing_answer",
                        "source_file": split_files["dev"],
                    })
                for answer_index, answer in enumerate(answers):
                    answers_total += 1
                    if not isinstance(answer, Mapping):
                        span_audit.append({
                            "query_id": qid,
                            "doc_id": doc_id,
                            "answer_index": answer_index,
                            "status": "malformed_answer_object",
                            "original_answer": answer,
                        })
                        continue
                    valid, start, quote, status = _validate_squad_answer(context, answer)
                    if not valid or quote is None or start is None:
                        span_audit.append({
                            "query_id": qid,
                            "doc_id": doc_id,
                            "answer_index": answer_index,
                            "status": status,
                            "original_answer": dict(answer),
                        })
                        continue
                    valid_answers += 1
                    key = (qid, doc_id, quote, start, start + len(quote))
                    if key in seen_evidence:
                        duplicate_answers += 1
                        span_audit.append({
                            "query_id": qid,
                            "doc_id": doc_id,
                            "answer_index": answer_index,
                            "status": "duplicate_valid_answer_annotation",
                            "original_answer": dict(answer),
                        })
                        continue
                    seen_evidence.add(key)
                    evidence.append({
                        "query_id": qid,
                        "doc_id": doc_id,
                        "quote": quote,
                        "start": start,
                        "end": start + len(quote),
                        "metadata": {
                            "source_family": family,
                            "source_query_id": qid,
                            "source_answer_index": answer_index,
                            "offset_units": "Python Unicode code points as in original JSON",
                            "span_status": "official_answer_span_validated",
                        },
                    })

    counts = {
        "source_paragraphs_train_plus_dev": source_paragraphs,
        "deduplicated_context_documents": len(documents),
        "duplicate_context_rows_merged": source_paragraphs - len(documents),
        "official_dev_queries": len(queries),
        "qrels_queries": len(qrels),
        "positive_pairs": sum(len(row) for row in qrels.values()),
        "evidence_rows_valid_unique": len(evidence),
        "answers_total_in_official_dev": answers_total,
        "answer_spans_valid": valid_answers,
        "answer_spans_review_records": len(span_audit),
        "duplicate_valid_answer_annotations": duplicate_answers,
        "max_positive_per_query": max((len(row) for row in qrels.values()), default=0),
        "split_stats": split_stats,
    }
    return {
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "evidence": evidence,
        "span_audit": {
            "source_family": family,
            "language": language,
            "query_split": "dev",
            "qrels_retained_for_all_official_dev_queries": True,
            "counts": counts,
            "records": span_audit,
        },
        "counts": counts,
        "source_files": {
            split: {
                "path": str(source_dir / name),
                "sha256": file_digest(source_dir / name)[0],
                "bytes": file_digest(source_dir / name)[1],
            }
            for split, name in split_files.items()
        },
    }


_DOC_TAG_RE = re.compile(r"<doc\b(?P<attrs>[^>]*)>", re.IGNORECASE | re.DOTALL)
_TITLE_ATTR_RE = re.compile(r"\btitle=(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)


def _thai_title(context: str) -> str:
    match = _DOC_TAG_RE.search(context)
    if not match:
        return ""
    title = _TITLE_ATTR_RE.search(match.group("attrs"))
    return html.unescape(title.group(2)) if title else ""


def _parse_thai(source_dir: Path) -> dict[str, Any]:
    documents_by_key: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, filename in TH_SOURCE_FILES.items():
        path = source_dir / filename.rsplit("/", 1)[-1]
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid ThaiQA JSONL at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise RuntimeError(f"ThaiQA row is not an object at {path}:{line_number}")
                rows.append(row)
        rows_by_split[split] = rows
        source_counts[split] = {"rows": len(rows), "unique_article_ids": 0, "unique_contexts": 0}

        article_ids: set[str] = set()
        context_keys: set[str] = set()
        for row_index, row in enumerate(rows):
            context = row.get("context")
            if not isinstance(context, str) or not context:
                raise RuntimeError(f"ThaiQA context is empty/non-string at {path}:{row_index + 1}")
            article_id = row.get("article_id")
            article_id_text = str(article_id) if article_id is not None else ""
            article_ids.add(article_id_text)
            key = _sha256_text("th\0" + context)
            context_keys.add(key)
            alias = {
                "source_file": path.name,
                "source_split": split,
                "source_row_index": row_index,
                "source_article_id": article_id_text,
                "source_title": _thai_title(context),
            }
            _document_for_context(
                language="th",
                context=context,
                title=_thai_title(context),
                family="ThaiQA",
                alias=alias,
                source_context_id=article_id_text or None,
                documents_by_key=documents_by_key,
            )
        source_counts[split]["unique_article_ids"] = len(article_ids)
        source_counts[split]["unique_contexts"] = len(context_keys)

    documents = sorted(documents_by_key.values(), key=lambda row: row["_id"])
    doc_by_text = {_sha256_text("th\0" + row["text"]): row["_id"] for row in documents}
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, float]] = {}
    evidence: list[dict[str, Any]] = []
    span_audit: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    seen_evidence: set[tuple[str, str, str, int, int]] = set()
    answers_total = 0
    valid_answers = 0
    for row_index, row in enumerate(rows_by_split["dev"]):
        qid_raw = row.get("question_id")
        qid = str(qid_raw) if qid_raw is not None else ""
        question = row.get("question")
        context = row.get("context")
        article_id = row.get("article_id")
        article_id_text = str(article_id) if article_id is not None else ""
        if not qid or not isinstance(question, str) or not question:
            raise RuntimeError(f"ThaiQA dev row has missing question_id/question at row {row_index}")
        if not isinstance(context, str) or not context:
            raise RuntimeError(f"ThaiQA dev row has missing context at row {row_index}")
        if qid in seen_queries:
            raise RuntimeError(f"duplicate official ThaiQA dev question_id {qid!r}")
        seen_queries.add(qid)
        doc_id = doc_by_text[_sha256_text("th\0" + context)]
        queries.append({
            "_id": qid,
            "text": question,
            "language": "th",
            "metadata": {
                "source_family": "ThaiQA",
                "source_scope": "complete train+dev source contexts; official dev queries only",
                "task_origin": "source extractive QA converted to paragraph retrieval",
                "source_query_id": qid,
                "source_query_id_raw": qid_raw,
                "source_split": "dev",
                "source_file": Path(TH_SOURCE_FILES["dev"]).name,
                "source_row_index": row_index,
                "source_article_id": article_id_text,
                "source_context_id": doc_id,
                "source_question_language": "th",
                "source_context_language": "th",
                "translation_status": "native",
                "translation_status_evidence": "pinned ThaiQA README attributes questions and answer annotations to NECTEC; no translation performed by this adapter",
                "qa_granularity": "source QA context / retrieval paragraph",
                "annotation_source": "source_provided",
            },
        })
        qrels[qid] = {doc_id: 1.0}
        answer = row.get("answer")
        start = row.get("answer_begin_position")
        end = row.get("answer_end_position")
        answers_total += 1
        valid = (
            isinstance(answer, str)
            and bool(answer)
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start >= 0
            and end == start + len(answer)
            and context[start:end] == answer
        )
        if not valid:
            span_audit.append({
                "query_id": qid,
                "doc_id": doc_id,
                "source_file": Path(TH_SOURCE_FILES["dev"]).name,
                "source_row_index": row_index,
                "status": "invalid_original_optional_answer_offset",
                "original_answer": {
                    "answer": answer,
                    "answer_begin_position": start,
                    "answer_end_position": end,
                },
            })
            continue
        valid_answers += 1
        key = (qid, doc_id, answer, start, end)
        if key in seen_evidence:
            span_audit.append({
                "query_id": qid,
                "doc_id": doc_id,
                "source_row_index": row_index,
                "status": "duplicate_valid_answer_annotation",
            })
            continue
        seen_evidence.add(key)
        evidence.append({
            "query_id": qid,
            "doc_id": doc_id,
            "quote": answer,
            "start": start,
            "end": end,
            "metadata": {
                "source_family": "ThaiQA",
                "source_query_id": qid,
                "source_row_index": row_index,
                "offset_units": "Python Unicode code points as in source JSONL",
                "span_status": "official_answer_span_validated",
            },
        })

    counts = {
        "source_rows_train_plus_dev": sum(item["rows"] for item in source_counts.values()),
        "source_unique_contexts_train_plus_dev": len(documents),
        "deduplicated_context_documents": len(documents),
        "duplicate_context_rows_merged": sum(item["rows"] for item in source_counts.values()) - len(documents),
        "official_dev_queries": len(queries),
        "qrels_queries": len(qrels),
        "positive_pairs": sum(len(row) for row in qrels.values()),
        "evidence_rows_valid_unique": len(evidence),
        "answers_total_in_official_dev": answers_total,
        "answer_spans_valid": valid_answers,
        "answer_spans_review_records": len(span_audit),
        "max_positive_per_query": max((len(row) for row in qrels.values()), default=0),
        "split_stats": source_counts,
        "source_context_format": "literal <doc ...>...</doc> context retained without stripping or paragraph splitting",
    }
    return {
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "evidence": evidence,
        "span_audit": {
            "source_family": "ThaiQA",
            "language": "th",
            "query_split": "dev",
            "qrels_retained_for_all_official_dev_queries": True,
            "counts": counts,
            "records": span_audit,
        },
        "counts": counts,
        "source_files": {
            split: {
                "path": str(source_dir / Path(name).name),
                "sha256": file_digest(source_dir / Path(name).name)[0],
                "bytes": file_digest(source_dir / Path(name).name)[1],
            }
            for split, name in TH_SOURCE_FILES.items()
        },
    }


def _materialize(language: str) -> dict[str, Any]:
    info = SOURCE_INFO[language]
    raw_dir = RAW_ROOT / f"gapfill_nativeqa_{language}"
    source_dir = raw_dir / "source_inputs"
    canonical_dir = raw_dir / "canonical_input"
    source_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    downloads: dict[str, Any] = {}

    if language == "ru":
        archive_path = source_dir / "sber_squad_clean-v1.1.tar.gz"
        downloads["archive"] = _download_once(
            RU_ARCHIVE_URL,
            archive_path,
            expected_bytes=RU_ARCHIVE_BYTES,
            expected_sha256=RU_ARCHIVE_SHA256,
            max_bytes=200 * 1024 * 1024,
        )
        with tarfile.open(archive_path, "r:gz") as archive:
            member_names = {member.name for member in archive.getmembers() if member.isfile()}
            for name in RU_SOURCE_FILES:
                if name not in member_names:
                    raise RuntimeError(f"SberQuAD archive member missing: {name}")
                member = archive.getmember(name)
                if member.size > 150 * 1024 * 1024:
                    raise RuntimeError(f"SberQuAD member too large: {name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(f"SberQuAD member is unreadable: {name}")
                _write_bytes_once(source_dir / name, handle.read())
        split_files = {"train": "train-v1.1.json", "dev": "dev-v1.1.json"}
        converted = _parse_squad_language(
            "ru", family="SberQuAD", source_dir=source_dir, split_files=split_files
        )
        source_files = converted["source_files"]
    elif language == "th":
        archive_path = source_dir / "data.zip"
        downloads["archive"] = _download_once(
            TH_ARCHIVE_URL,
            archive_path,
            expected_bytes=TH_ARCHIVE_BYTES,
            expected_sha256=TH_ARCHIVE_SHA256,
            max_bytes=200 * 1024 * 1024,
        )
        with zipfile.ZipFile(archive_path) as archive:
            for split, member_name in TH_SOURCE_FILES.items():
                try:
                    member = archive.getinfo(member_name)
                except KeyError as exc:
                    raise RuntimeError(f"ThaiQA archive member missing: {member_name}") from exc
                if member.is_dir() or member.file_size > 150 * 1024 * 1024:
                    raise RuntimeError(f"ThaiQA member unsafe or too large: {member_name}")
                _write_bytes_once(source_dir / Path(member_name).name, archive.read(member))
        converted = _parse_thai(source_dir)
        source_files = converted["source_files"]
    elif language == "tr":
        for split, descriptor in TR_FILES.items():
            path = source_dir / descriptor["name"]
            downloads[split] = _download_once(
                descriptor["url"],
                path,
                expected_bytes=descriptor["bytes"],
                expected_sha256=descriptor["sha256"],
                max_bytes=20 * 1024 * 1024,
            )
        converted = _parse_squad_language(
            "tr",
            family="TQuAD",
            source_dir=source_dir,
            split_files={"train": TR_FILES["train"]["name"], "dev": TR_FILES["dev"]["name"]},
        )
        source_files = converted["source_files"]
    else:  # pragma: no cover - argparse and run() guard this
        raise ValueError(f"unsupported language: {language}")

    canonical_paths = {
        "documents": canonical_dir / "documents.jsonl",
        "queries": canonical_dir / "queries.jsonl",
        "qrels": canonical_dir / "qrels.json",
        "evidence": canonical_dir / "evidence.jsonl",
    }
    _write_jsonl_once(canonical_paths["documents"], converted["documents"])
    _write_jsonl_once(canonical_paths["queries"], converted["queries"])
    _write_json_once(canonical_paths["qrels"], converted["qrels"])
    _write_jsonl_once(canonical_paths["evidence"], converted["evidence"])
    _write_json_once(raw_dir / "span_audit.json", converted["span_audit"])

    conversion = {
        "schema_version": "native-qa-gapfill-conversion-v1",
        "source_family": info["source_family"],
        "language": language,
        "language_name": info["language_name"],
        "dataset": info["dataset"],
        "revision": info["revision"],
        "license": info["license"],
        "official_sources": info["official_sources"],
        "native_evidence": info["native_evidence"],
        "query_split": "dev only",
        "corpus_splits": ["train", "dev"],
        "corpus_granularity": "source QA context treated as retrieval paragraph; exact text preserved",
        "document_id_strategy": "sha256(language + NUL + exact_source_context) first 32 hex; all duplicate aliases retained",
        "quality_policy": QUALITY_POLICY,
        "downloads": downloads,
        "source_files": source_files,
        "counts": converted["counts"],
        "canonical_files": {
            role: {"path": str(path), "sha256": file_digest(path)[0], "bytes": file_digest(path)[1]}
            for role, path in canonical_paths.items()
        },
        "raw_paths": {
            "source_inputs": str(source_dir),
            "canonical_input": str(canonical_dir),
            "span_audit": str(raw_dir / "span_audit.json"),
        },
    }
    # A second invocation verifies and reuses the same source bytes.  Keep the
    # first invocation's transient fetched/reused status in the immutable
    # conversion manifest so reruns do not attempt to rewrite history.
    previous_conversion_path = raw_dir / "conversion_manifest.json"
    if previous_conversion_path.exists():
        try:
            previous_conversion = json.loads(previous_conversion_path.read_text(encoding="utf-8"))
            previous_downloads = previous_conversion.get("downloads", {})
            for key, current in conversion["downloads"].items():
                previous = previous_downloads.get(key, {})
                if (
                    isinstance(previous, Mapping)
                    and previous.get("url") == current.get("url")
                    and previous.get("bytes") == current.get("bytes")
                    and previous.get("sha256") == current.get("sha256")
                    and previous.get("path") == current.get("path")
                ):
                    current["status"] = previous.get("status", current["status"])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
    _write_json_once(raw_dir / "conversion_manifest.json", conversion)
    return {
        "language": language,
        "source_family": info["source_family"],
        "raw_dir": str(raw_dir),
        "canonical_dir": str(canonical_dir),
        "canonical_paths": canonical_paths,
        "source_files": source_files,
        "downloads": downloads,
        "documents": converted["documents"],
        "queries": converted["queries"],
        "qrels": converted["qrels"],
        "evidence": converted["evidence"],
        "counts": converted["counts"],
        "conversion": conversion,
    }


def _source_spec(language: str, materialized: Mapping[str, Any]) -> dict[str, Any]:
    info = SOURCE_INFO[language]
    canonical = materialized["canonical_paths"]
    return {
        "id": f"gapfill_nativeqa_{language}",
        "connector": "generic_jsonl",
        "input_kind": "evidence_qa",
        "domain": "encyclopedia",
        "genre": "native QA-derived paragraph retrieval",
        "language": language,
        "license": info["license"],
        "revision": info["revision"],
        "metadata": {
            "source_family": info["source_family"],
            "source_scope": "complete train+dev source contexts; official dev queries only",
            "task_origin": "source extractive QA converted to paragraph retrieval",
            "query_translation_status": "native",
            "corpus_language_policy": "same-language source contexts only; no translation or cross-language padding",
            "source_context_policy": "exact text retained; exact duplicate contexts merge only with source_aliases",
        },
        "options": {
            "dataset": info["dataset"],
            "source_family": info["source_family"],
            "source_urls": info["official_sources"],
            "files": {
                "documents": {"path": str(canonical["documents"]), "name": "data/documents.jsonl"},
                "queries": {"path": str(canonical["queries"]), "name": "data/queries.jsonl"},
                "qrels": {"path": str(canonical["qrels"]), "name": "data/qrels.json"},
                "evidence": {"path": str(canonical["evidence"]), "name": "data/evidence.jsonl"},
            },
        },
        "selection": {"max_queries": None, "seed": 20261023},
    }


def _publish(language: str, materialized: Mapping[str, Any]) -> dict[str, Any]:
    spec = _source_spec(language, materialized)
    release = EVAL_ROOT / "pipeline_v1" / f"gapfill_nativeqa_{language}" / "v1"
    if release.exists():
        manifest = verify_release(release)
        if manifest.get("specification") != spec:
            raise RuntimeError(f"existing native QA release has a different specification: {release}")
        # Preserve the original acquisition status in an existing research
        # record so a verified rerun is byte-stable.  The release manifest is
        # the authoritative reuse check; this only avoids changing a prose
        # status field from fetched_verified to reused_verified.
        status = "reused_verified"
        research_path = RESEARCH_ROOT / "research.json"
        if research_path.exists():
            try:
                previous = json.loads(research_path.read_text(encoding="utf-8"))
                for previous_source in previous.get("sources", []):
                    if previous_source.get("id") == f"gapfill_nativeqa_{language}":
                        status = previous_source.get("status", status)
                        break
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                pass
        return {
            "spec": spec,
            "snapshot": manifest.get("raw_snapshot"),
            "release": str(release),
            "manifest": manifest,
            "status": status,
        }
    snapshot = fetch_source(spec, RAW_ROOT)
    manifest = build_source(spec, RAW_ROOT, release, snapshot=snapshot, policy=QUALITY_POLICY)
    verified = verify_release(release)
    for key in ("bundle_hash", "num_documents", "num_queries", "num_positive_pairs", "state"):
        if verified.get(key) != manifest.get(key):
            raise RuntimeError(f"published native QA manifest changed during verification: {key}")
    return {
        "spec": spec,
        "snapshot": str(snapshot),
        "release": str(release),
        "manifest": verified,
        "status": "fetched_verified",
    }


def _blocked_candidates() -> list[dict[str, Any]]:
    return [
        {
            "id": "FQuAD_fr",
            "language": "fr",
            "status": "blocked_candidate",
            "official_urls": [
                "https://storage.googleapis.com/illuin/fquad/train.json.zip",
                "https://storage.googleapis.com/illuin/fquad/valid.json.zip",
                "https://illuin-tech.github.io/FQuAD-explorer/",
            ],
            "license": "non-commercial/research terms stated by the FQuAD dataset page; redistribution restrictions require separate review",
            "reason": "bounded HEAD request to both legacy storage URLs returned HTTP 403; no long retry and no unverified mirror used",
        },
        {
            "id": "PIAF_fr",
            "language": "fr",
            "status": "candidate_not_downloaded",
            "official_urls": [
                "https://etalab-ia.github.io/knowledge-base/piaf/index.html",
                "https://arxiv.org/abs/2007.00968",
            ],
            "license": "not resolved for a complete source release in this bounded pass",
            "reason": "no stable, size-bounded official train+dev source file was identified before the ru/th/tr builds",
        },
    ]


def _research_entry(language: str, materialized: Mapping[str, Any], published: Mapping[str, Any]) -> dict[str, Any]:
    manifest = published["manifest"]
    return {
        "id": f"gapfill_nativeqa_{language}",
        "source_family": materialized["source_family"],
        "language": language,
        "language_name": SOURCE_INFO[language]["language_name"],
        "dataset": SOURCE_INFO[language]["dataset"],
        "revision": SOURCE_INFO[language]["revision"],
        "official_sources": SOURCE_INFO[language]["official_sources"],
        "license": SOURCE_INFO[language]["license"],
        "native_evidence": SOURCE_INFO[language]["native_evidence"],
        "status": published["status"],
        "query_policy": "official dev questions only; train questions are source-counted but never emitted as held-out queries",
        "corpus_policy": "all train+dev source contexts, exact-text deduplication with aliases; no cross-language padding",
        "context_policy": "source context text is retained byte-for-byte; ThaiQA literal <doc> tags are retained and not split",
        "translation_status": "native",
        "counts": materialized["counts"],
        "source_files": materialized["source_files"],
        "raw_path": materialized["raw_dir"],
        "snapshot_path": published.get("snapshot"),
        "release_path": published["release"],
        "state": manifest.get("state"),
        "bundle_hash": manifest.get("bundle_hash"),
        "raw_manifest_sha256": manifest.get("raw_manifest_sha256"),
        "quality_policy": QUALITY_POLICY,
        "validation_boundary": manifest.get("validation_boundary"),
        "limitations": [
            "The qrel is the source QA question-to-context association and is not an exhaustive IR judgment pool.",
            "Answer evidence is optional; every invalid/missing span remains in raw span_audit while its qrel is retained.",
            "License is unknown for ru/tr and is recorded as such; research access does not imply redistribution permission.",
        ],
    }


def _write_initial_progress() -> None:
    """Create a small progress marker without overwriting a parent marker."""

    path = RESEARCH_ROOT / "progress.json"
    if path.exists():
        return
    _write_json_once(
        path,
        {
            "schema_version": "native-qa-gapfill-progress-v1",
            "updated_at": "2026-10-23",
            "status": "materialization_started",
            "languages_requested": list(LANGUAGES),
            "completed": [],
            "blocked_candidates": ["FQuAD_fr"],
        },
    )


def run(languages: tuple[str, ...]) -> dict[str, Any]:
    unknown = sorted(set(languages) - set(LANGUAGES))
    if unknown:
        raise ValueError(f"unsupported native QA languages: {unknown}")
    RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    _write_initial_progress()
    entries: list[dict[str, Any]] = []
    materialized: dict[str, dict[str, Any]] = {}
    published: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for language in languages:
        try:
            current = _materialize(language)
            current_published = _publish(language, current)
            materialized[language] = current
            published[language] = current_published
            entries.append(_research_entry(language, current, current_published))
        except Exception as exc:
            failures.append({
                "language": language,
                "status": "failed_preserved",
                "error": f"{type(exc).__name__}: {exc}",
            })
            break

    research = {
        "schema_version": "native-qa-gapfill-research-v1",
        "created_at": "2026-10-23",
        "source_family": "native-language QA-derived paragraph retrieval",
        "official_sources": {
            language: SOURCE_INFO[language]["official_sources"] for language in languages
        },
        "sources": entries,
        "acquired_verified": [entry["id"] for entry in entries if entry.get("status") in {"fetched_verified", "reused_verified"}],
        "blocked_candidates": _blocked_candidates(),
        "quality_policy": QUALITY_POLICY,
        "matrix_boundary": {
            "domain": "encyclopedia",
            "languages": list(languages),
            "query_to_document_contract": "official source QA question -> its source context document",
            "query_translation_status": "native only",
            "corpus_contract": "complete source train+dev contexts per family; exact duplicate text merged with aliases",
            "heldout_contract": "dev only; no training question is emitted as a retrieval query",
        },
        "reproduction": {
            "python": "/mnt/afs2/202609/miniconda/envs/dci-bench/bin/python",
            "command": f"/mnt/afs2/202609/miniconda/envs/dci-bench/bin/python {REPO_ROOT / 'scripts/acquire_native_qa_gapfill_20261023.py'} --languages {' '.join(languages)}",
            "verify_commands": [
                f"/mnt/afs2/202609/miniconda/envs/dci-bench/bin/python scripts/data_pipeline.py verify {EVAL_ROOT / 'pipeline_v1' / 'gapfill_nativeqa_<lang>' / 'v1'}",
            ],
            "network": "bounded pipeline download_url with configured proxy-aware opener; each source is pinned by bytes and sha256",
            "gpu_or_cascade": "not run",
        },
        "failures": failures,
    }
    _write_json_once(RESEARCH_ROOT / "research.json", research)
    if failures:
        raise RuntimeError(f"native QA acquisition stopped after preserved failure: {failures[0]}")
    return research


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    args = parser.parse_args(argv)
    research = run(tuple(args.languages))
    print(json.dumps({
        "status": "ok",
        "acquired_verified": research["acquired_verified"],
        "sources": [
            {
                "id": source["id"],
                "status": source["status"],
                "release_path": source.get("release_path"),
                "state": source.get("state"),
                "counts": source.get("counts"),
            }
            for source in research["sources"]
        ],
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
