#!/usr/bin/env python3
"""Acquire pinned JSQuAD train/valid and publish valid-QA passage retrieval.

The source is the native Japanese JSQuAD release in yahoojapan/JGLUE.  The
train and valid passages form the corpus; only valid questions are retained as
queries.  This is a QA-derived passage retrieval adaptation and does not claim
to reproduce the JSQuAD reading-comprehension leaderboard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.connectors import _base_bundle, fetch_source
from dci_bench.data.pipeline.core import publish_bundle, verify_release


BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
REVISION = "6f071c09316baae89c3d083a90985b4b1cb9968c"
REPO = "yahoojapan/JGLUE"
API_BASE = f"https://api.github.com/repos/{REPO}/contents/datasets/jsquad-v1.3"
FILES: dict[str, tuple[int, str]] = {
    "train-v1.3.json": (21737422, "87b42821ac5487246164483d625b928bf8904344"),
    "valid-v1.3.json": (1976638, "bcde414e0b7131607bb1fa2c4542dd888f68771a"),
}

SPEC: dict[str, Any] = {
    "id": "qa_jsquad_ja",
    "connector": "squad",
    "input_kind": "evidence_qa",
    "domain": "encyclopedia",
    "language": "ja",
    "revision": REVISION,
    "license": "CC-BY-SA-4.0; see yahoojapan/JGLUE README",
    "options": {
        "dataset": REPO,
        "translation_status": "native",
        "genre": "Japanese Wikipedia passage",
        "task": "JSQuAD v1.3 QA-derived passage retrieval",
        "max_bytes": 30000000,
        "timeout": 60,
        "retries": 2,
        "headers": {
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": "dci-bench-data-pipeline",
        },
        "files": [
            {
                "url": f"{API_BASE}/{name}?ref={REVISION}",
                "name": f"data/{name}",
                "bytes": size,
            }
            for name, (size, _) in FILES.items()
        ],
    },
    "metadata": {
        "source_scope": "Full train+valid paragraph corpus; valid questions only; QA-derived passage retrieval.",
        "query_split": "valid",
        "corpus_splits": ["train", "valid"],
        "granularity": "Japanese Wikipedia paragraph",
        "official_url": f"https://github.com/{REPO}",
        "readme_url": f"https://raw.githubusercontent.com/{REPO}/{REVISION}/README.md",
        "expected_git_blobs": {name: blob for name, (_, blob) in FILES.items()},
        "native_or_translated": "native",
    },
}


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def stable_paragraph_id(*, split: str, article_index: int, paragraph_index: int, context: str) -> str:
    """Derive an ID that retains source position when paragraphs have no ID."""

    position = f"{split}\0{article_index}\0{paragraph_index}"
    digest = hashlib.sha256((position + "\0" + context).encode("utf-8")).hexdigest()
    return f"passage-{split}-a{article_index}-p{paragraph_index}-{digest}"


def _validate_squad_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"JSQuAD file must contain a data list: {path}")
    return payload


def build_bundle(snapshot: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((snapshot / "source_manifest.json").read_text(encoding="utf-8"))
    bundle = _base_bundle(SPEC, manifest)
    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, int]] = {}
    evidence: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    seen_queries: set[str] = set()
    invalid_spans: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}

    for filename in FILES:
        split = "valid" if filename.startswith("valid-") else "train"
        path = snapshot / "data" / filename
        payload = _validate_squad_file(path)
        article_count = 0
        paragraph_count = 0
        qa_count = 0
        for article_index, article in enumerate(payload["data"]):
            if not isinstance(article, dict) or not isinstance(article.get("paragraphs"), list):
                raise ValueError(f"invalid article/paragraph list in {path}:{article_index}")
            title = article.get("title", "")
            if not isinstance(title, str):
                raise ValueError(f"non-string article title in {path}:{article_index}")
            article_count += 1
            for paragraph_index, paragraph in enumerate(article["paragraphs"]):
                if not isinstance(paragraph, dict):
                    raise ValueError(f"invalid paragraph in {path}:{article_index}:{paragraph_index}")
                context = paragraph.get("context")
                if not isinstance(context, str) or not context.strip():
                    raise ValueError(f"empty/non-string context in {path}:{article_index}:{paragraph_index}")
                paragraph_count += 1
                original_id = paragraph.get("id")
                if original_id is not None and (not isinstance(original_id, str) or not original_id):
                    raise ValueError(f"invalid paragraph id in {path}:{article_index}:{paragraph_index}")
                document_id = (
                    original_id
                    if isinstance(original_id, str)
                    else stable_paragraph_id(
                        split=split,
                        article_index=article_index,
                        paragraph_index=paragraph_index,
                        context=context,
                    )
                )
                if document_id in seen_documents:
                    raise ValueError(f"duplicate paragraph/document ID: {document_id}")
                seen_documents.add(document_id)
                metadata: dict[str, Any] = {
                    "source_file": f"data/{filename}",
                    "original_split": split,
                    "source_article_index": article_index,
                    "source_paragraph_index": paragraph_index,
                    "source_title": title,
                    "translation_status": "native",
                    "id_derivation": "original_paragraph_id" if original_id else "sha256(split,NUL,article_index,NUL,paragraph_index,NUL,context)",
                    "source_position": {
                        "file": filename,
                        "article_index": article_index,
                        "paragraph_index": paragraph_index,
                    },
                }
                if original_id:
                    metadata["original_paragraph_id"] = original_id
                documents.append(
                    {
                        "_id": document_id,
                        "title": title,
                        "text": context,
                        "language": "ja",
                        "metadata": metadata,
                    }
                )

                if split != "valid":
                    continue
                qas = paragraph.get("qas")
                if not isinstance(qas, list):
                    raise ValueError(f"paragraph qas must be a list in {path}:{article_index}:{paragraph_index}")
                for qa_index, qa in enumerate(qas):
                    if not isinstance(qa, dict):
                        raise ValueError(f"invalid QA in {path}:{article_index}:{paragraph_index}:{qa_index}")
                    query_id = qa.get("id")
                    question = qa.get("question")
                    if not isinstance(query_id, str) or not query_id or query_id in seen_queries:
                        raise ValueError(f"duplicate or invalid original query ID: {query_id!r}")
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError(f"empty/non-string question: {query_id}")
                    seen_queries.add(query_id)
                    qa_count += 1
                    queries.append(
                        {
                            "_id": query_id,
                            "text": question,
                            "language": "ja",
                            "metadata": {
                                "annotation_source": "source_human",
                                "original_split": "valid",
                                "translation_status": "native",
                                "source_file": f"data/{filename}",
                                "source_article_index": article_index,
                                "source_paragraph_index": paragraph_index,
                                "source_paragraph_id": document_id,
                                "source_title": title,
                            },
                        }
                    )
                    # Preserve the original QA-to-paragraph association even
                    # if an optional answer span is malformed.
                    qrels[query_id] = {document_id: 1}
                    answers = qa.get("answers", [])
                    if not isinstance(answers, list):
                        answers = []
                        invalid_spans.append(
                            {
                                "query_id": query_id,
                                "reason": "answers_not_a_list",
                                "source_file": filename,
                                "source_article_index": article_index,
                                "source_paragraph_index": paragraph_index,
                            }
                        )
                    for answer_index, answer in enumerate(answers):
                        if not isinstance(answer, dict):
                            invalid_spans.append(
                                {
                                    "query_id": query_id,
                                    "answer_index": answer_index,
                                    "reason": "answer_not_an_object",
                                    "original_answer": answer,
                                }
                            )
                            continue
                        start = answer.get("answer_start")
                        quote = answer.get("text")
                        valid_span = (
                            isinstance(start, int)
                            and not isinstance(start, bool)
                            and start >= 0
                            and isinstance(quote, str)
                            and bool(quote)
                            and context[start : start + len(quote)] == quote
                        )
                        if not valid_span:
                            invalid_spans.append(
                                {
                                    "query_id": query_id,
                                    "answer_index": answer_index,
                                    "reason": "invalid_original_optional_answer_offset",
                                    "original_answer": answer,
                                    "source_file": filename,
                                    "source_article_index": article_index,
                                    "source_paragraph_index": paragraph_index,
                                }
                            )
                            continue
                        row = {
                            "query_id": query_id,
                            "doc_id": document_id,
                            "quote": quote,
                            "start": start,
                            "end": start + len(quote),
                            "annotation_source": "source_human",
                            "offset_unit": "unicode_codepoints",
                        }
                        if row not in evidence:
                            evidence.append(row)
        source_counts[split] = {
            "articles": article_count,
            "paragraphs": paragraph_count,
            "queries": qa_count,
        }

    if len(documents) <= 1000:
        raise ValueError(f"JSQuAD train+valid corpus must exceed 1000 documents, got {len(documents)}")
    if len(queries) != len(qrels):
        raise ValueError(f"query/qrel count mismatch: {len(queries)} != {len(qrels)}")
    query_ids = {query["_id"] for query in queries}
    if set(qrels) != query_ids:
        raise ValueError("qrels do not cover exactly the retained valid queries")
    document_ids = {document["_id"] for document in documents}
    if any(doc_id not in document_ids for per_query in qrels.values() for doc_id in per_query):
        raise ValueError("qrels reference a missing document")

    bundle["documents"] = documents
    bundle["queries"] = queries
    bundle["qrels"] = qrels
    bundle["evidence"] = evidence
    bundle["provenance"].update(
        {
            "query_split": "valid",
            "corpus_splits": ["train", "valid"],
            "query_selection": "all original valid QA IDs; no train queries and no test access",
            "original_valid_queries": len(queries),
            "qrels_derivation": "Original valid question-to-paragraph association; one grade-1 paragraph positive per question.",
            "optional_evidence_policy": "Keep exact original offsets only; malformed offsets are retained in provenance and never remove the query/qrel.",
            "ignored_optional_spans": invalid_spans,
            "source_counts": source_counts,
            "native_or_translated": "native",
            "license": SPEC["license"],
        }
    )
    return bundle, {
        "source_counts": source_counts,
        "invalid_optional_spans": len(invalid_spans),
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_positive_pairs": sum(len(per_query) for per_query in qrels.values()),
        "num_evidence": len(evidence),
    }


def main() -> None:
    output = BASE / "eval_data/pipeline_v1/qa_jsquad_ja/expansion-v1"
    receipt = BASE / "eval_data/source_research_20261023/jsquad/research.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("Refusing to overwrite an existing JSQuAD release or research receipt")

    snapshot = fetch_source(SPEC, BASE / "raw_data")
    git_blob_receipts: dict[str, dict[str, Any]] = {}
    for name, (expected_bytes, expected_blob) in FILES.items():
        path = snapshot / "data" / name
        actual_bytes = path.stat().st_size
        actual_blob = git_blob_sha1(path)
        if actual_bytes != expected_bytes or actual_blob != expected_blob:
            raise ValueError(
                f"Pinned Git blob mismatch for {name}: bytes={actual_bytes}/{expected_bytes}, "
                f"blob={actual_blob}/{expected_blob}"
            )
        git_blob_receipts[name] = {
            "bytes": actual_bytes,
            "expected_bytes": expected_bytes,
            "git_blob_sha1": actual_blob,
            "expected_git_blob_sha1": expected_blob,
            "verified": True,
        }

    bundle, counts = build_bundle(snapshot)
    manifest = publish_bundle(
        bundle,
        output,
        specification=SPEC,
        raw_snapshot=snapshot,
        policy={"max_positive_per_query": None, "exclude_review_required": False},
    )
    verified = verify_release(output)
    if verified != manifest:
        raise ValueError("verify_release returned a different manifest")
    if verified["num_documents"] <= 1000:
        raise ValueError("verified release corpus is not larger than 1000")
    if verified["num_queries"] != counts["num_queries"] or verified["num_positive_pairs"] != counts["num_positive_pairs"]:
        raise ValueError("verified release counts differ from source counts")

    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mteb-origin-research-v1",
        "sources": [
            {
                "id": SPEC["id"],
                "repository": REPO,
                "revision": REVISION,
                "acquisition_status": "published_verified",
                "release_path": str(output),
                "raw_snapshot": str(snapshot),
                "urls": [
                    f"https://github.com/{REPO}/tree/{REVISION}/datasets/jsquad-v1.3",
                    f"https://raw.githubusercontent.com/{REPO}/{REVISION}/README.md",
                    *[f"{API_BASE}/{name}?ref={REVISION}" for name in FILES],
                ],
                "counts": {
                    **counts,
                    "release_num_documents": verified["num_documents"],
                    "release_num_queries": verified["num_queries"],
                    "release_num_positive_pairs": verified["num_positive_pairs"],
                    "release_state": verified["state"],
                },
                "git_blob_verification": git_blob_receipts,
                "language": "ja",
                "native_or_translated": "native",
                "license": SPEC["license"],
                "quality_policy": {
                    "max_positive_per_query": None,
                    "exclude_review_required": False,
                },
                "limitations": [
                    "QA-derived seed positives, not exhaustive relevance judgments over the corpus.",
                    "Train passages are corpus-only; valid questions and their original paragraph qrels are retained.",
                    "Test was intentionally not downloaded or used because this release does not require test answers.",
                    "Malformed optional answer offsets are recorded in provenance and do not remove the associated valid query/qrel.",
                ],
            }
        ],
        "command": f"{sys.executable} {ROOT / 'scripts/acquire_jsquad_20261023.py'}",
        "official_readme": {
            "url": f"https://raw.githubusercontent.com/{REPO}/{REVISION}/README.md",
            "claims_used": [
                "JGLUE is constructed from scratch without translation.",
                "JSQuAD is Japanese SQuAD 1.1-style QA over Japanese Wikipedia contexts.",
                "JGLUE is licensed CC BY-SA 4.0.",
            ],
        },
    }
    if receipt.exists():
        raise FileExistsError(f"research receipt appeared unexpectedly: {receipt}")
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
