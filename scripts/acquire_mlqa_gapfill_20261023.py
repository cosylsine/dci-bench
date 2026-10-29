#!/usr/bin/env python3
"""Acquire and publish bounded, language-separated MLQA retrieval views.

The upstream MLQA archive is SQuAD-shaped rather than a retrieval corpus.  This
adapter keeps the complete same-language dev+test paragraph collection, uses
only same-language test questions, and derives one score-1 paragraph qrel from
each official answer association.  It never uses another language to pad a
view and never drops a query because an answer span is malformed.

This script intentionally owns only the MLQA gap-fill namespace.  The generic
pipeline still owns snapshot manifests, canonical validation, quality audits,
and immutable release publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dci_bench.data.pipeline.connectors import fetch_source  # noqa: E402
from dci_bench.data.pipeline.core import build_source, verify_release  # noqa: E402
from dci_bench.data.pipeline.download import download_url, file_digest  # noqa: E402


BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
RAW_ROOT = BASE / "raw_data"
EVAL_ROOT = BASE / "eval_data"
RESEARCH_ROOT = EVAL_ROOT / "source_research_20261023_round2" / "mlqa"
ARCHIVE_RAW = RAW_ROOT / "gapfill_mlqa_archive" / "source_inputs"
ARCHIVE_PATH = ARCHIVE_RAW / "MLQA_V1.zip"

ARCHIVE_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"
README_URL = "https://github.com/facebookresearch/MLQA/blob/main/README.md"
README_RAW_URL = "https://raw.githubusercontent.com/facebookresearch/MLQA/main/README.md"
PAPER_URL = "https://arxiv.org/abs/1910.07475"
PAPER_HTML_URL = "https://arxiv.org/html/1910.07475v3"
ARCHIVE_BYTES = 75_719_050
ARCHIVE_SHA256 = "246e8089933d13007fe80684d5c5c0713d6834cf8b3b4a0ec7c66f0a0d2baac8"
ARCHIVE_LAST_MODIFIED = "2019-10-16T21:33:54Z"
ARCHIVE_VERSION = f"MLQA_V1.zip#sha256:{ARCHIVE_SHA256}"

LANGUAGES = ("en", "ar", "de", "es", "hi", "vi", "zh")
LANGUAGE_NAMES = {
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "zh": "Simplified Chinese",
}
README_REPORTED_QA_COUNTS = {
    "en": {"dev": 1148, "test": 11590},
    "de": {"dev": 512, "test": 4517},
    "es": {"dev": 500, "test": 5254},
    "ar": {"dev": 517, "test": 5335},
    "zh": {"dev": 504, "test": 5137},
    "vi": {"dev": 511, "test": 5495},
    "hi": {"dev": 507, "test": 4918},
}

# These values are required for this family.  In particular, no query is
# removed merely because an equivalent paragraph is retained in the corpus.
QUALITY_POLICY = {
    "max_positive_per_query": None,
    "exclude_review_required": False,
    "drop_near_duplicates": False,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _write_bytes_once(path: Path, payload: bytes) -> str:
    """Write an immutable regular file, preserving a conflicting partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(payload)
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
    return _write_bytes_once(Path(path), _canonical_json(value).encode("utf-8"))


def _write_jsonl_once(path: Path, rows: list[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    return _write_bytes_once(Path(path), payload)


def _ensure_archive() -> dict[str, Any]:
    """Fetch the one permitted archive with direct, bounded pipeline I/O."""
    ARCHIVE_RAW.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        digest, size = file_digest(ARCHIVE_PATH, max_bytes=150 * 1024 * 1024)
        if digest != ARCHIVE_SHA256 or size != ARCHIVE_BYTES:
            raise RuntimeError(
                f"existing MLQA archive failed pinned integrity: {ARCHIVE_PATH} "
                f"sha256={digest} bytes={size}"
            )
        status = "reused_verified"
    else:
        # download_url uses a ProxyHandler({}) opener, keeps .partial on
        # failure, resumes Range downloads, and permits at most one retry.
        digest, size = download_url(
            ARCHIVE_URL,
            ARCHIVE_PATH,
            timeout=30,
            retries=1,
            max_bytes=150 * 1024 * 1024,
            expected_sha256=ARCHIVE_SHA256,
            expected_bytes=ARCHIVE_BYTES,
            overwrite=False,
        )
        status = "fetched_verified"
    return {
        "url": ARCHIVE_URL,
        "path": str(ARCHIVE_PATH),
        "status": status,
        "revision": ARCHIVE_VERSION,
        "last_modified": ARCHIVE_LAST_MODIFIED,
        "sha256": ARCHIVE_SHA256,
        "bytes": ARCHIVE_BYTES,
        "license": "CC-BY-SA 3.0",
        "archive_integrity": "verified",
    }


def _member_name(split: str, language: str) -> str:
    return f"MLQA_V1/{split}/{split}-context-{language}-question-{language}.json"


def _read_member(archive: zipfile.ZipFile, member: str) -> tuple[bytes, dict[str, Any]]:
    try:
        info = archive.getinfo(member)
    except KeyError as exc:
        raise RuntimeError(f"MLQA archive member missing: {member}") from exc
    if info.is_dir() or info.file_size > 150 * 1024 * 1024:
        raise RuntimeError(f"unsafe or oversized MLQA member: {member}")
    payload = archive.read(member)
    return payload, {
        "archive_member": member,
        "compressed_bytes": info.compress_size,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _load_payload(payload: bytes, member: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid MLQA JSON member: {member}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        raise RuntimeError(f"MLQA member is not SQuAD-shaped: {member}")
    return value


def _source_position(split: str, data_index: int, paragraph_index: int, title: str) -> dict[str, Any]:
    return {
        "split": split,
        "data_index": data_index,
        "paragraph_index": paragraph_index,
        "title": title,
    }


def _materialize_language(language: str, archive_path: Path) -> dict[str, Any]:
    raw_dir = RAW_ROOT / f"gapfill_mlqa_{language}"
    source_dir = raw_dir / "source_inputs"
    canonical_dir = raw_dir / "canonical_input"
    source_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        parsed: dict[str, dict[str, Any]] = {}
        members: dict[str, dict[str, Any]] = {}
        for split in ("dev", "test"):
            member = _member_name(split, language)
            payload, member_meta = _read_member(archive, member)
            source_name = f"{split}-context-{language}-question-{language}.json"
            _write_bytes_once(source_dir / source_name, payload)
            parsed[split] = _load_payload(payload, member)
            members[split] = member_meta

    # Deduplicate exact paragraph text without losing source locations.  The
    # archive has no paragraph IDs, so each ID is a stable hash plus language;
    # all source positions remain in metadata.source_aliases.
    docs_by_key: dict[str, dict[str, Any]] = {}
    position_to_doc: dict[tuple[str, int, int], str] = {}
    source_paragraphs = 0
    for split in ("dev", "test"):
        data = parsed[split]["data"]
        for data_index, article in enumerate(data):
            if not isinstance(article, dict):
                raise RuntimeError(f"MLQA {language} {split} data[{data_index}] is not an object")
            title = str(article.get("title") or "")
            paragraphs = article.get("paragraphs")
            if not isinstance(paragraphs, list):
                raise RuntimeError(f"MLQA {language} {split} article {data_index} has no paragraphs")
            for paragraph_index, paragraph in enumerate(paragraphs):
                if not isinstance(paragraph, dict) or not isinstance(paragraph.get("context"), str):
                    raise RuntimeError(
                        f"MLQA {language} {split} paragraph {data_index}/{paragraph_index} is invalid"
                    )
                text = paragraph["context"]
                if not text:
                    raise RuntimeError(
                        f"MLQA {language} {split} paragraph {data_index}/{paragraph_index} is empty"
                    )
                source_paragraphs += 1
                text_key = _sha256_text(language + "\0" + text)
                doc_id = f"mlqa-{language}-paragraph-{text_key[:32]}"
                position = _source_position(split, data_index, paragraph_index, title)
                if text_key not in docs_by_key:
                    docs_by_key[text_key] = {
                        "_id": doc_id,
                        "title": title,
                        "text": text,
                        "language": language,
                        "metadata": {
                            "source_family": "MLQA",
                            "source_id_available": False,
                            "id_strategy": "sha256(language + NUL + exact_context) first 32 hex",
                            "source_context_id": None,
                            "source_aliases": [position],
                            "source_split_members": [members[split]["archive_member"]],
                            "source_language": language,
                            "corpus_language": language,
                            "corpus_origin": "Wikipedia",
                            "corpus_granularity": "paragraph",
                            "corpus_translation_status": "natural Wikipedia language corpus; article translation history not audited",
                            "parallel_document_family": f"mlqa-wikipedia-paragraph:{text_key}",
                        },
                    }
                    docs_by_key[text_key]["metadata"]["source_context_id"] = doc_id
                else:
                    existing = docs_by_key[text_key]
                    existing["metadata"]["source_aliases"].append(position)
                    if members[split]["archive_member"] not in existing["metadata"]["source_split_members"]:
                        existing["metadata"]["source_split_members"].append(members[split]["archive_member"])
                position_to_doc[(split, data_index, paragraph_index)] = doc_id

    documents = list(docs_by_key.values())
    documents.sort(key=lambda row: row["_id"])
    document_ids = {row["_id"] for row in documents}

    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, float]] = {}
    evidence: list[dict[str, Any]] = []
    span_audit: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    test_qas = 0
    answer_count = 0
    valid_answer_count = 0
    malformed_answer_count = 0
    for data_index, article in enumerate(parsed["test"]["data"]):
        title = str(article.get("title") or "")
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            text = paragraph["context"]
            doc_id = position_to_doc[("test", data_index, paragraph_index)]
            for qa_index, qa in enumerate(paragraph.get("qas", [])):
                if not isinstance(qa, dict):
                    raise RuntimeError(f"MLQA {language} test QA is not an object")
                qid = str(qa.get("id") or "")
                question = qa.get("question")
                if not qid or not isinstance(question, str) or not question:
                    raise RuntimeError(f"MLQA {language} test QA missing id/question at {data_index}/{paragraph_index}/{qa_index}")
                if qid in seen_query_ids:
                    raise RuntimeError(f"duplicate MLQA {language} official test query ID: {qid}")
                seen_query_ids.add(qid)
                test_qas += 1
                translation_status = "non_translated" if language == "en" else "human_translated"
                query_metadata = {
                    "source_family": "MLQA",
                    "source_query_id": qid,
                    "source_split": "test",
                    "source_member": members["test"]["archive_member"],
                    "source_data_index": data_index,
                    "source_paragraph_index": paragraph_index,
                    "source_title": title,
                    "source_context_id": doc_id,
                    "source_question_language": language,
                    "source_context_language": language,
                    "parallel_query_family": f"mlqa-test-qid:{qid}",
                    "parallel_query_family_source": "official MLQA question ID; language views are shared parallel instances",
                    "translation_status": translation_status,
                    "translation_source_language": None if language == "en" else "en",
                    "translation_process": "English questions annotated first; target questions translated by professional translators" if language != "en" else "English crowd-sourced question annotation",
                    "qa_granularity": "answerable Wikipedia paragraph",
                }
                queries.append({"_id": qid, "text": question, "language": language, "metadata": query_metadata})
                # One qrel preserves the official QA paragraph association. It
                # is a seed retrieval label, not an exhaustive IR judgment.
                qrels[qid] = {doc_id: 1.0}
                answers = qa.get("answers", [])
                if not isinstance(answers, list):
                    answers = []
                if not answers:
                    span_audit.append({"query_id": qid, "doc_id": doc_id, "status": "missing_answer", "qa_index": qa_index})
                for answer_index, answer in enumerate(answers):
                    answer_count += 1
                    if not isinstance(answer, dict):
                        malformed_answer_count += 1
                        span_audit.append({"query_id": qid, "doc_id": doc_id, "answer_index": answer_index, "status": "malformed_answer_object"})
                        continue
                    answer_text = answer.get("text")
                    start = answer.get("answer_start")
                    valid = (
                        isinstance(answer_text, str)
                        and bool(answer_text)
                        and isinstance(start, int)
                        and not isinstance(start, bool)
                        and start >= 0
                        and start + len(answer_text) <= len(text)
                        and text[start:start + len(answer_text)] == answer_text
                    )
                    if not valid:
                        malformed_answer_count += 1
                        span_audit.append({
                            "query_id": qid,
                            "doc_id": doc_id,
                            "answer_index": answer_index,
                            "status": "invalid_offset_or_quote",
                            "answer_text": answer_text,
                            "answer_start": start,
                        })
                        continue
                    valid_answer_count += 1
                    evidence.append({
                        "query_id": qid,
                        "doc_id": doc_id,
                        "quote": answer_text,
                        "start": start,
                        "end": start + len(answer_text),
                        "metadata": {
                            "source_family": "MLQA",
                            "source_query_id": qid,
                            "source_answer_index": answer_index,
                            "offset_units": "Python Unicode code points as in original JSON",
                            "span_status": "official_answer_span_validated",
                        },
                    })

    documents_path = canonical_dir / "documents.jsonl"
    queries_path = canonical_dir / "queries.jsonl"
    qrels_path = canonical_dir / "qrels.json"
    evidence_path = canonical_dir / "evidence.jsonl"
    _write_jsonl_once(documents_path, documents)
    _write_jsonl_once(queries_path, queries)
    _write_json_once(qrels_path, qrels)
    _write_jsonl_once(evidence_path, evidence)
    _write_json_once(
        raw_dir / "span_audit.json",
        {
            "source_family": "MLQA",
            "language": language,
            "query_split": "test",
            "answers_total": answer_count,
            "valid_answer_spans": valid_answer_count,
            "malformed_or_missing_answer_records": len(span_audit),
            "malformed_answer_count": malformed_answer_count,
            "records": span_audit,
            "qrels_retained_for_all_official_test_queries": True,
        },
    )
    conversion = {
        "source_family": "MLQA",
        "archive": {
            "url": ARCHIVE_URL,
            "revision": ARCHIVE_VERSION,
            "sha256": ARCHIVE_SHA256,
            "bytes": ARCHIVE_BYTES,
        },
        "language": language,
        "language_name": LANGUAGE_NAMES[language],
        "query_split": "test only",
        "corpus_splits": ["dev", "test"],
        "corpus_granularity": "Wikipedia paragraph",
        "source_ids_available": False,
        "document_id_strategy": "stable sha256(language + NUL + exact_context), preserving source_aliases",
        "quality_policy": QUALITY_POLICY,
        "source_members": members,
        "counts": {
            "source_paragraphs_dev_plus_test": source_paragraphs,
            "deduplicated_paragraph_documents": len(documents),
            "duplicate_context_rows_merged": source_paragraphs - len(documents),
            "official_test_queries": len(queries),
            "qrels_queries": len(qrels),
            "positive_pairs": sum(len(row) for row in qrels.values()),
            "evidence_rows_valid": len(evidence),
            "answers_total": answer_count,
            "answer_spans_valid": valid_answer_count,
            "answer_spans_malformed_or_missing": len(span_audit),
            "max_positive_per_query": max((len(row) for row in qrels.values()), default=0),
        },
        "canonical_files": {
            "documents": {"path": str(documents_path), "sha256": file_digest(documents_path)[0]},
            "queries": {"path": str(queries_path), "sha256": file_digest(queries_path)[0]},
            "qrels": {"path": str(qrels_path), "sha256": file_digest(qrels_path)[0]},
            "evidence": {"path": str(evidence_path), "sha256": file_digest(evidence_path)[0]},
        },
    }
    _write_json_once(raw_dir / "conversion_manifest.json", conversion)
    return {
        "language": language,
        "raw_dir": str(raw_dir),
        "canonical_dir": str(canonical_dir),
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "evidence": evidence,
        "counts": conversion["counts"],
        "conversion": conversion,
    }


def _source_spec(language: str, canonical_dir: Path) -> dict[str, Any]:
    return {
        "id": f"gapfill_mlqa_{language}",
        "connector": "generic_jsonl",
        "input_kind": "evidence_qa",
        "domain": "Wikipedia",
        "genre": "extractive QA / paragraph retrieval",
        "language": language,
        "license": "CC-BY-SA 3.0",
        "revision": ARCHIVE_VERSION,
        "metadata": {
            "source_family": "MLQA",
            "parallel_query_family": "MLQA official test IDs",
            "query_translation_policy": "English non_translated; target-language questions human_translated by professional translators",
            "corpus_language_policy": "same-language Wikipedia paragraphs only; document-level native history not audited",
        },
        "options": {
            "dataset": "facebookresearch/MLQA",
            "archive_url": ARCHIVE_URL,
            "archive_member_view": f"test-context-{language}-question-{language} plus dev/test same-language members",
            "files": {
                "documents": {"path": str(canonical_dir / "documents.jsonl"), "name": "data/documents.jsonl"},
                "queries": {"path": str(canonical_dir / "queries.jsonl"), "name": "data/queries.jsonl"},
                "qrels": {"path": str(canonical_dir / "qrels.json"), "name": "data/qrels.json"},
                "evidence": {"path": str(canonical_dir / "evidence.jsonl"), "name": "data/evidence.jsonl"},
            },
        },
        "selection": {"max_queries": None, "seed": 20261023},
    }


def _publish_language(language: str, materialized: dict[str, Any]) -> dict[str, Any]:
    canonical_dir = Path(materialized["canonical_dir"])
    spec = _source_spec(language, canonical_dir)
    raw_source_root = RAW_ROOT
    release = EVAL_ROOT / "pipeline_v1" / f"gapfill_mlqa_{language}" / "v1"
    if release.exists():
        manifest = verify_release(release)
        if manifest.get("specification") != spec:
            raise RuntimeError(f"existing MLQA release has a different specification: {release}")
        return {"spec": spec, "snapshot": manifest.get("raw_snapshot"), "release": str(release), "manifest": manifest, "status": "reused_verified"}
    snapshot = fetch_source(spec, raw_source_root)
    manifest = build_source(spec, raw_source_root, release, snapshot=snapshot, policy=QUALITY_POLICY)
    verified = verify_release(release)
    if verified != manifest:
        # Verification returns the persisted manifest.  Compare the key
        # integrity fields instead of relying on object identity/timestamps.
        for key in ("bundle_hash", "num_documents", "num_queries", "num_positive_pairs", "state"):
            if verified.get(key) != manifest.get(key):
                raise RuntimeError(f"published MLQA manifest changed during verification: {key}")
    return {"spec": spec, "snapshot": str(snapshot), "release": str(release), "manifest": verified, "status": "fetched_verified"}


def _research_entry_archive(archive_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "MLQA_V1_archive",
        "source_family": "MLQA",
        "url": ARCHIVE_URL,
        "official_repository": README_URL,
        "status": archive_info["status"],
        "revision": ARCHIVE_VERSION,
        "license": "CC-BY-SA 3.0",
        "native_or_translated": "mixed: Wikipedia language corpora; non-English questions translated from English by professional translators",
        "granularity": "SQuAD QA over Wikipedia paragraphs",
        "counts": {
            "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": ARCHIVE_SHA256,
            "official_language_views": 7,
        },
        "archive_path": str(ARCHIVE_PATH),
        "release_path": None,
        "limitations": [
            "MLQA is an extractive QA benchmark; these qrels are one official answer-paragraph association per test question, not exhaustive IR judgments.",
            "The source archive has no paragraph IDs; stable IDs and source_aliases preserve positions and exact source text.",
        ],
    }


def _research_entry_language(language: str, materialized: dict[str, Any], published: dict[str, Any]) -> dict[str, Any]:
    counts = materialized["counts"]
    translation = "non_translated" if language == "en" else "human_translated"
    return {
        "id": f"gapfill_mlqa_{language}",
        "source_family": "MLQA",
        "url": ARCHIVE_URL,
        "official_repository": README_URL,
        "status": published["status"],
        "revision": ARCHIVE_VERSION,
        "license": "CC-BY-SA 3.0",
        "language": language,
        "language_name": LANGUAGE_NAMES[language],
        "native_or_translated": {
            "questions": translation,
            "contexts": "same-language Wikipedia paragraphs; source article translation history not audited",
        },
        "granularity": "paragraph",
        "query_policy": "official same-language test questions only",
        "corpus_policy": "all same-language dev+test paragraphs, exact duplicate merge with aliases",
        "counts": counts,
        "upstream_readme_reported_qa_counts": README_REPORTED_QA_COUNTS[language],
        "count_reconciliation": {
            "same_language_test_delta_vs_readme": counts["official_test_queries"] - README_REPORTED_QA_COUNTS[language]["test"],
            "note": "The official README table reports 5,254 Spanish test instances; MLQA_V1/test/test-context-es-question-es.json contains 5,253 QAs, which is retained as the direct archive count.",
        },
        "release_path": published["release"],
        "raw_path": materialized["raw_dir"],
        "snapshot_path": published.get("snapshot"),
        "state": published["manifest"].get("state"),
        "bundle_hash": published["manifest"].get("bundle_hash"),
        "quality_policy": QUALITY_POLICY,
        "limitations": [
            "QA-derived score-1 qrels identify the source paragraph for each official answer and are not a complete retrieval relevance pool.",
            "Parallel query family is shared across language views by official MLQA question ID; the seven releases are one source family, not independent benchmarks.",
            "No paragraph source ID is present in the SQuAD JSON; deterministic stable IDs plus all split/article/paragraph positions are retained.",
        ],
    }


def _save_research_documents() -> dict[str, str]:
    """Copy official textual research evidence if available in /tmp.

    These copies are convenience evidence only; URLs remain authoritative and
    the acquisition does not depend on a transient /tmp file.
    """
    copied: dict[str, str] = {}
    for name, source in (("official_README.md", Path("/tmp/mlqa_readme_main.md")), ("official_paper.html", Path("/tmp/mlqa_paper.html"))):
        if source.exists() and source.is_file():
            target = RESEARCH_ROOT / name
            _write_bytes_once(target, source.read_bytes())
            copied[name] = str(target)
    return copied


def run(languages: tuple[str, ...]) -> dict[str, Any]:
    RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    archive_info = _ensure_archive()
    archive_research = _research_entry_archive(archive_info)
    materialized: dict[str, dict[str, Any]] = {}
    published: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = [archive_research]
    failures: list[dict[str, Any]] = []
    for language in languages:
        try:
            m = _materialize_language(language, ARCHIVE_PATH)
            p = _publish_language(language, m)
            materialized[language] = m
            published[language] = p
            entries.append(_research_entry_language(language, m, p))
        except Exception as exc:
            failures.append({"language": language, "status": "failed_preserved", "error": f"{type(exc).__name__}: {exc}"})
            break
    copied_evidence = _save_research_documents()
    query_id_sets = {
        language: set(materialized[language]["qrels"])
        for language in languages
        if language in materialized
    }
    parallel_query_summary = {
        "family_key": "official MLQA test question ID",
        "source_family": "MLQA",
        "language_views": list(query_id_sets),
        "union_query_families": len(set().union(*query_id_sets.values())) if query_id_sets else 0,
        "all_language_intersection": len(set.intersection(*query_id_sets.values())) if query_id_sets else 0,
        "pairwise_overlap": {
            left: {right: len(query_id_sets[left] & query_id_sets[right]) for right in query_id_sets}
            for left in query_id_sets
        },
        "interpretation": "Language releases share official MLQA question-ID families; per-language counts differ because the aligned benchmark is only parallel for a subset of languages.",
    }
    research = {
        "schema_version": "mlqa-gapfill-research-v1",
        "created_at": "2026-10-23",
        "source_family": "MLQA",
        "official_sources": {
            "repository": README_URL,
            "readme_raw": README_RAW_URL,
            "archive": ARCHIVE_URL,
            "paper": PAPER_URL,
            "paper_html": PAPER_HTML_URL,
        },
        "archive": archive_info,
        "sources": entries,
        "acquired_verified": [entry["id"] for entry in entries if entry.get("status") in {"fetched_verified", "reused_verified"}],
        "candidates_not_downloaded": [
            {
                "id": "MLQA_cross_language_context_question_views",
                "status": "candidate_not_downloaded",
                "reason": "This gap-fill task publishes only same-language context/question views; cross-language combinations remain in the original archive and are not separate releases.",
                "url": ARCHIVE_URL,
            },
            {
                "id": "MLQA_translate_train_and_translate_test_packages",
                "status": "candidate_not_downloaded",
                "reason": "Official README lists machine-translated auxiliary packages, but this task is restricted to MLQA_V1.zip.",
                "url": README_URL,
            },
        ],
        "research_evidence_copies": copied_evidence,
        "translation_process": {
            "paper": PAPER_HTML_URL,
            "summary": "The paper describes English questions crowd-sourced on aligned English paragraphs, then target-language questions translated by professional translators; answer spans are annotated in aligned target contexts.",
            "context_provenance": "Wikipedia paragraphs are naturally sourced in each language; the paper warns that some articles may be volunteer translations, so releases do not claim every context is independently native.",
        },
        "count_reconciliation": {
            "official_readme_table": README_URL,
            "es_test_note": "README reports 5,254 Spanish test instances; direct same-language archive member has 5,253 QAs. The release keeps the direct member count.",
            "other_languages": "Direct archive member counts match the README table.",
        },
        "parallel_query_families": parallel_query_summary,
        "reproduction": {
            "python": "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python",
            "command": f"/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python {REPO_ROOT / 'scripts/acquire_mlqa_gapfill_20261023.py'} --languages {' '.join(languages)}",
            "verify_commands": [
                f"/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python scripts/data_pipeline.py verify {EVAL_ROOT / 'pipeline_v1' / 'gapfill_mlqa_<lang>' / 'v1'}",
            ],
            "network": "direct urllib opener via pipeline download_url; no proxy; timeout=30s; retries=1",
            "archive_integrity": {"bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256, "last_modified": ARCHIVE_LAST_MODIFIED},
        },
        "quality_policy": QUALITY_POLICY,
        "failures": failures,
    }
    _write_json_once(RESEARCH_ROOT / "research.json", research)
    if failures:
        raise RuntimeError(f"MLQA acquisition stopped after preserved failure: {failures[0]}")
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
                "counts": source.get("counts"),
            }
            for source in research["sources"]
        ],
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
