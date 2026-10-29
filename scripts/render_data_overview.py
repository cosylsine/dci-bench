#!/usr/bin/env python3
"""Render the combined pipeline and frozen legacy domain/language matrix.

The input is deliberately a small source matrix rather than a corpus reader.  The
renderer only consumes source-level counts and metadata, so it cannot silently
turn a candidate-pool size into a unique-document count or a document qrel into
span evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


EXPECTED_SOURCE_VIEWS = 31
EXPECTED_QUERY_INSTANCES = 16254

# This is an inclusion list, not a metric fixture.  Counts are always read from
# document_tasks.json, which keeps the renderer valid when a frozen source is
# repacked with a new manifest.
LEGACY_TASKS = {
    "LLMAILAStatutes": {
        "domain": "legal",
        "language": "en",
        "language_basis": "inferred",
        "language_note": "English inferred from the task/query records; corpus language is not declared in the index.",
        "domain_basis": "legacy_mapping",
        "translation": "unknown",
        "translation_note": "No translation field in the frozen metadata.",
    },
    "LLMFQuADRetrieval": {
        "domain": "general_qa",
        "language": "fr",
        "language_basis": "task_instruction",
        "language_note": "French stated in the frozen task instruction.",
        "domain_basis": "legacy_mapping",
        "translation": "unknown",
        "translation_note": "No translation field in the frozen metadata.",
    },
    "LLMHC3FinanceRetrieval": {
        "domain": "finance",
        "language": "en",
        "language_basis": "inferred",
        "language_note": "English inferred from the task/query records; corpus language is not declared in the index.",
        "domain_basis": "legacy_mapping",
        "translation": "unknown",
        "translation_note": "No translation field in the frozen metadata.",
    },
    "LLMLegalBenchConsumerContractsQA": {
        "domain": "legal",
        "language": "en",
        "language_basis": "inferred",
        "language_note": "English inferred from the task/query records; corpus language is not declared in the index.",
        "domain_basis": "legacy_mapping",
        "translation": "unknown",
        "translation_note": "No translation field in the frozen metadata.",
    },
    "LLMPublicHealthQA": {
        "domain": "public_health",
        "language": "en",
        "language_basis": "inferred",
        "language_note": "English inferred from the task/query records; corpus language is not declared in the index.",
        "domain_basis": "legacy_mapping",
        "translation": "unknown",
        "translation_note": "No translation field in the frozen metadata.",
    },
    "LLMTwitterHjerneRetrieval": {
        "domain": "social_media",
        "language": "da",
        "language_basis": "task_instruction",
        "language_note": "Danish stated in the frozen task instruction.",
        "domain_basis": "legacy_mapping",
        "translation": "unknown",
        "translation_note": "No translation field in the frozen metadata.",
    },
    "browsecomp_plus": {
        "domain": "web",
        "language": "en",
        "language_basis": "inferred",
        "language_note": "English inferred for the task; the BrowseComp corpus language is explicitly unknown.",
        "corpus_language_unknown": True,
        "domain_basis": "legacy_mapping",
        "translation": "unknown",
        "translation_note": "No translation field in the frozen metadata.",
    },
    "enterprise_rag": {
        "domain": "enterprise",
        "language": "en",
        "language_basis": "inferred",
        "language_note": "English inferred from the task/query records; corpus language is not declared in the index.",
        "domain_basis": "legacy_mapping",
        "translation": "unknown",
        "translation_note": "No translation field in the frozen metadata.",
    },
}

PREFERRED_DOMAINS = [
    "scientific",
    "enterprise",
    "software",
    "encyclopedia",
    "legal",
    "general_qa",
    "finance",
    "public_health",
    "social_media",
    "web",
]
PREFERRED_LANGUAGES = [
    "en",
    "zh",
    "ar",
    "de",
    "el",
    "es",
    "hi",
    "ru",
    "th",
    "tr",
    "vi",
    "fr",
    "ja",
    "da",
]


def slug(value: Any) -> str:
    """Make an identity key without changing the displayed source name."""

    if value is None:
        return ""
    text = str(value).strip().casefold().replace("\\", "/")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if re.fullmatch(r"[-+]?\d+", cleaned):
            return int(cleaned)
    return default


def scalar(value: Any, default: Any = None) -> Any:
    """Return a usable scalar from common count/metadata wrappers."""

    if value is None:
        return default
    if isinstance(value, dict):
        for key in ("value", "name", "id", "label", "status", "state"):
            if key in value and not isinstance(value[key], (dict, list)):
                return value[key]
        return default
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value


def first(mapping: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def count_value(mapping: dict[str, Any], keys: Iterable[str], default: int = 0) -> int:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, dict):
            value = first(value, ("count", "total", "num", "value", "size"), default)
        elif isinstance(value, list):
            value = len(value)
        parsed = as_int(value, default)
        if value is not None:
            return parsed
    return default


def load_source_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("sources", "records", "entries", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None and isinstance(payload.get("matrix"), dict):
            candidate = payload["matrix"].get("sources")
            if isinstance(candidate, list):
                records = candidate
        if records is None:
            raise ValueError(f"{path}: expected a source list under sources/records/entries/items")
    else:
        raise ValueError(f"{path}: expected a JSON list or source-list object")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: every source record must be an object")
    return records


def source_values(record: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    source = record.get("source")
    if isinstance(source, dict):
        values.extend(source.get(key) for key in ("id", "source_id", "name", "task", "dataset", "path", "uri"))
    else:
        values.append(source)
    values.extend(record.get(key) for key in ("source_id", "id", "name", "task", "dataset", "path", "uri"))
    return [value for value in values if value not in (None, "")]


def source_aliases(record: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for value in source_values(record):
        key = slug(value)
        if not key:
            continue
        aliases.add(key)
        # A path/dataset often has a useful final component; include it to catch
        # a frozen task represented once by a dataset and once by a short name.
        for part in str(value).replace("\\", "/").split("/"):
            part_key = slug(part)
            if part_key:
                aliases.add(part_key)
    return aliases


def canonical_source_id(record: dict[str, Any]) -> str:
    values = source_values(record)
    for value in values:
        key = slug(value)
        if key:
            return key
    raise ValueError("source record has no id/source/name/task/dataset/path")


def nested_field(record: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    value = first(record, keys, default)
    if value is not None:
        return value
    source = record.get("source")
    if isinstance(source, dict):
        return first(source, keys, default)
    return default


def normalize_language(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = "+".join(str(item) for item in value)
    text = str(scalar(value, "unknown") or "unknown").strip().casefold().replace("_", "-")
    aliases = {"english": "en", "french": "fr", "danish": "da", "unknown": "unknown"}
    return aliases.get(text, text)


def normalize_domain(value: Any) -> str:
    text = str(scalar(value, "unknown") or "unknown").strip().casefold().replace(" ", "_")
    return text or "unknown"


def translation_value(raw: Any) -> tuple[str, str]:
    if raw is None:
        return "unknown", "No translation status declared."
    if isinstance(raw, dict):
        status = first(raw, ("status", "translation_status", "label", "type"), None)
        if status is None:
            # The pipeline matrix uses a compact histogram such as
            # {"unknown": 48}; retain that status instead of collapsing it to
            # the uninformative label "declared".
            statuses = [(str(key), value) for key, value in raw.items() if key not in {"note", "details", "source"}]
            if len(statuses) == 1:
                status = statuses[0][0]
                details = f"{statuses[0][1]} query records"
            elif statuses:
                status = "mixed"
                details = ", ".join(f"{key}={value}" for key, value in statuses)
            else:
                status = "declared"
                details = ""
        else:
            details = first(raw, ("note", "details", "source"), "")
        return str(status), str(details)
    if isinstance(raw, bool):
        return ("translated" if raw else "native"), "Boolean translation field."
    return str(raw), ""


def parse_evidence(record: dict[str, Any], queries: int) -> dict[str, Any]:
    raw = record.get("evidence")
    evidence_count = 0
    evidence_queries: int | None = None
    evidence_scope = str(first(record, ("evidence_scope", "evidence_kind"), "span_or_explicit"))
    evidence_note = str(first(record, ("evidence_note", "evidence_notes"), ""))

    if isinstance(raw, dict):
        evidence_count = count_value(raw, ("count", "total", "items", "num_evidence", "evidence_count", "pairs"), 0)
        evidence_queries = count_value(raw, ("queries", "query_count", "covered_queries", "evidence_queries", "with_evidence"), -1)
        if evidence_queries < 0:
            evidence_queries = None
        evidence_scope = str(first(raw, ("scope", "kind", "evidence_scope"), evidence_scope))
        evidence_note = str(first(raw, ("note", "notes"), evidence_note))
    elif isinstance(raw, list):
        evidence_count = len(raw)
    elif raw is not None:
        evidence_count = as_int(raw, 0)

    explicit_query_count = first(
        record,
        ("evidence_queries", "evidence_query_count", "covered_evidence_queries", "queries_with_evidence"),
        None,
    )
    if explicit_query_count is not None:
        evidence_queries = as_int(explicit_query_count, 0)

    explicit_pending = first(
        record,
        ("pending_review", "pending_evidence_review", "evidence_pending", "review_required", "review_queries"),
        None,
    )
    if isinstance(explicit_pending, bool):
        explicit_pending = queries if explicit_pending else 0
    # A missing evidence file is a Q+R source state, not an evidence review
    # queue.  Only an explicit review_queries/pending_review field contributes
    # to the pending count.
    pending = as_int(explicit_pending, 0) if explicit_pending is not None else 0

    # A numeric evidence count is an item/pair count, not automatically a query
    # coverage count.  Unknown coverage remains marked for review instead of
    # being presented as complete.
    evidence_complete = record.get("evidence_complete")
    if evidence_complete is True and evidence_queries is None:
        evidence_queries = queries
    if evidence_count <= 0:
        evidence_star = False
    elif evidence_queries is None:
        evidence_star = queries > 0
    else:
        evidence_queries = max(0, min(queries, evidence_queries))
        evidence_star = evidence_queries != queries
    return {
        "evidence": evidence_count,
        "evidence_queries": evidence_queries if evidence_queries is not None else 0,
        "evidence_coverage_known": evidence_queries is not None,
        "evidence_star": bool(evidence_star),
        "evidence_pending_review": max(0, pending),
        "evidence_scope": evidence_scope,
        "evidence_note": evidence_note,
    }


def display_state(
    documents: int,
    queries: int,
    positive_pairs: int,
    evidence: int,
    evidence_star: bool,
) -> str:
    """State shown in the matrix, derived from the available source counts."""

    if evidence > 0:
        return "Q+R+E*" if evidence_star else "Q+R+E"
    if queries > 0:
        return "Q+R"
    if documents > 0:
        return "C"
    return "unknown"


def source_display(record: dict[str, Any], source_id: str) -> tuple[str, str]:
    source = record.get("source")
    if isinstance(source, dict):
        display = first(source, ("name", "id", "source_id", "task", "dataset", "path"), source_id)
        path = first(source, ("path", "uri", "dataset", "canonical", "release"), display)
    else:
        display = source or first(record, ("name", "task", "dataset"), source_id)
        path = first(record, ("source_path", "path", "uri", "dataset", "canonical", "release", "release_path"), display)
    return str(display), str(path)


def normalize_pipeline_record(record: dict[str, Any]) -> dict[str, Any]:
    source_id = canonical_source_id(record)
    display, source_path = source_display(record, source_id)
    queries = count_value(record, ("queries", "num_queries", "query_count"), 0)
    evidence = parse_evidence(record, queries)
    translation, translation_note = translation_value(record.get("translation"))
    language = normalize_language(nested_field(record, ("language", "lang"), "unknown"))
    domain = normalize_domain(nested_field(record, ("domain",), "unknown"))
    language_basis = str(first(record, ("language_basis", "language_source"), "declared"))
    language_inferred = bool(record.get("language_inferred", language_basis == "inferred"))
    corpus_unknown = bool(record.get("corpus_language_unknown", False))
    documents = count_value(record, ("documents", "num_documents", "document_count", "corpus"), 0)
    positive_pairs = count_value(record, ("positive_pairs", "num_positive_pairs", "qrels", "relevant_pairs", "num_qrels"), 0)
    raw_state = str(first(record, ("state", "status"), "unknown"))
    return {
        "origin": "pipeline",
        "source_id": source_id,
        "source": display,
        "display_name": str(first(record, ("name",), display)),
        "source_path": source_path,
        "dataset": str(first(record, ("dataset", "dataset_id"), "")),
        "domain": domain,
        "language": language,
        "domain_basis": str(first(record, ("domain_basis",), "pipeline_record")),
        "language_basis": language_basis,
        "language_inferred": language_inferred,
        "language_note": str(first(record, ("language_note",), "")),
        "corpus_language_unknown": corpus_unknown,
        "translation": translation,
        "translation_note": translation_note,
        "documents": documents,
        "queries": queries,
        "positive_pairs": positive_pairs,
        **evidence,
        "source_state": raw_state,
        "state": display_state(documents, queries, positive_pairs, evidence["evidence"], evidence["evidence_star"]),
        "release_state": str(first(record, ("release_state", "release_status"), "unknown")),
        "counts_basis": "pipeline source record",
        "full_counts_from_record": True,
    }


def normalize_legacy_record(record: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    task_name = str(record["task"])
    spec = LEGACY_TASKS[task_name]
    queries = as_int(record.get("num_queries"), 0)
    # Legacy metadata has document-level qrels only.  In particular, the
    # BrowseComp evidence qrels are not span evidence and therefore contribute
    # zero to E.  Keep their count separately for transparent reporting.
    is_browse = task_name == "browsecomp_plus"
    qrels = as_int(record.get("num_qrels"), 0)
    evidence_note = (
        "Document-level evidence qrels; not span evidence E. "
        "BrowseComp corpus language is unknown."
        if is_browse
        else "No explicit evidence/span file in the frozen task metadata."
    )
    return {
        "origin": "legacy_frozen_only",
        "source_id": slug(task_name),
        "source": task_name,
        "display_name": {
            "LLMAILAStatutes": "AILA",
            "LLMFQuADRetrieval": "FQuAD",
            "LLMHC3FinanceRetrieval": "HC3 finance",
            "LLMLegalBenchConsumerContractsQA": "LegalBench",
            "LLMPublicHealthQA": "PublicHealth",
            "LLMTwitterHjerneRetrieval": "Twitter Hjerne",
            "browsecomp_plus": "BrowseComp+",
            "enterprise_rag": "EnterpriseRAG",
        }[task_name],
        "source_path": str(record.get("canonical", record.get("metadata", task_name))),
        "dataset": str(record.get("dataset", "")),
        "domain": spec["domain"],
        "language": spec["language"],
        "domain_basis": spec["domain_basis"],
        "language_basis": spec["language_basis"],
        "language_inferred": spec["language_basis"] == "inferred",
        "language_note": spec["language_note"],
        "corpus_language_unknown": bool(spec.get("corpus_language_unknown", False)),
        "translation": spec["translation"],
        "translation_note": spec["translation_note"],
        "documents": as_int(record.get("num_docs"), 0),
        "queries": queries,
        "positive_pairs": qrels,
        "evidence": 0,
        "evidence_queries": 0,
        "evidence_coverage_known": True,
        "evidence_star": False,
        "evidence_pending_review": 0,
        "evidence_scope": "document_qrels" if is_browse else "none",
        "evidence_note": evidence_note,
        "document_evidence_qrels": qrels if is_browse else 0,
        "source_state": "frozen",
        "state": display_state(as_int(record.get("num_docs"), 0), queries, qrels, 0, False),
        "release_state": "legacy_frozen_only",
        "counts_basis": "legacy document_tasks.json record",
        "full_counts_from_record": True,
        "revision": str(record.get("revision", "")),
        "legacy_task_instruction": str(record.get("task_instruction", "")),
    }


def format_int(value: int) -> str:
    return f"{int(value):,}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_plot_source(row: dict[str, Any]) -> str:
    """Use explicit, readable aliases without truncating source identities."""

    source = row["source"]
    language = row["language"]
    if source.startswith("xquad_"):
        translated = " (T)" if row["translation"] == "translated" or language != "en" else ""
        return f"XQuAD {language}{translated}"
    if source.startswith("github_kubernetes_website_"):
        return f"Kubernetes {language}"
    aliases = {
        "beir_scifact_test": "SciFact",
        "qasper_dev": "QASPER",
        "wixqa_expertwritten": "WixQA expert",
        "wixqa_simulated": "WixQA simulated",
        "worksurface_backend_developer": "WS backend",
        "worksurface_logistics_manager": "WS logistics",
        "worksurface_operations_manager": "WS operations",
        "LLMAILAStatutes": "AILA",
        "LLMFQuADRetrieval": "FQuAD",
        "LLMHC3FinanceRetrieval": "HC3 finance",
        "LLMLegalBenchConsumerContractsQA": "LegalBench",
        "LLMPublicHealthQA": "PublicHealth",
        "LLMTwitterHjerneRetrieval": "Twitter Hjerne",
        "browsecomp_plus": "BrowseComp+",
        "enterprise_rag": "EnterpriseRAG",
    }
    return aliases.get(source, row.get("display_name", source))


def metric_text(row: dict[str, Any]) -> str:
    pending = f" · 待复核 {format_int(row['evidence_pending_review'])}" if row["evidence_pending_review"] else ""
    return (
        f"{row['state']} · {format_int(row['documents'])} 文档 · "
        f"{format_int(row['queries'])} queries · {format_int(row['positive_pairs'])} 正例关系 · "
        f"{format_int(row['evidence'])} 证据{pending}"
    )


def cell_records(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["domain"], row["language"])].append(row)
    return dict(cells)


def ordered_values(preferred: list[str], observed: Iterable[str]) -> list[str]:
    observed_set = set(observed)
    return [item for item in preferred if item in observed_set] + sorted(observed_set - set(preferred))


def html_options(values: Iterable[str]) -> str:
    return "".join(f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>' for value in values)


def render_html(output: Path, rows: list[dict[str, Any]], domains: list[str], languages: list[str], payload: dict[str, Any]) -> None:
    cells = cell_records(rows)
    origin_values = sorted({row["origin"] for row in rows})
    state_values = sorted({row["state"] for row in rows})
    release_values = sorted({row["release_state"] for row in rows})
    translation_values = sorted({row["translation"] for row in rows})

    matrix_rows: list[str] = []
    for domain in domains:
        cells_html: list[str] = []
        for language in languages:
            items = cells.get((domain, language), [])
            if not items:
                cells_html.append('<td class="empty-cell">—<small>no source</small></td>')
                continue
            cards = []
            for row in items:
                attrs = " ".join(
                    f'data-{key}="{html.escape(str(value), quote=True)}"'
                    for key, value in (
                        ("source", row["source"]),
                        ("origin", row["origin"]),
                        ("domain", row["domain"]),
                        ("language", row["language"]),
                        ("state", row["state"]),
                        ("release", row["release_state"]),
                        ("translation", row["translation"]),
                    )
                )
                notes = [row["language_note"], row["translation_note"], row["evidence_note"]]
                if row["corpus_language_unknown"]:
                    notes.append("corpus_language_unknown=true")
                notes = [note for note in notes if note]
                note_html = " ".join(html.escape(note) for note in notes)
                document_qrel = (
                    f'<p class="warning">document-level qrels: {format_int(row.get("document_evidence_qrels", 0))}; '
                    "not span evidence E</p>"
                    if row.get("document_evidence_qrels", 0)
                    else ""
                )
                cards.append(
                    f'<article class="source-card" {attrs}>'
                    f'<strong>{html.escape(row["source"])}</strong>'
                    f'<span class="origin">{html.escape(row["origin"])}</span>'
                    f'<p>{html.escape(metric_text(row))}</p>'
                    f'<p class="meta">release={html.escape(row["release_state"])} · source_state={html.escape(row.get("source_state", row["state"]))} · '
                    f'translation={html.escape(row["translation"])}</p>'
                    f'{document_qrel}'
                    f'<details><summary>source and basis</summary><code>{html.escape(row["source_path"])}</code>'
                    f'<p>{note_html}</p><p>counts: {html.escape(row["counts_basis"])}</p></details>'
                    "</article>"
                )
            cells_html.append(f'<td data-cell="{html.escape(domain)}|{html.escape(language)}">{"".join(cards)}</td>')
        matrix_rows.append(f'<tr><th class="domain-label">{html.escape(domain)}</th>{"".join(cells_html)}</tr>')

    source_rows = []
    for row in rows:
        source_rows.append(
            "<tr data-source-row=" + html.escape(row["source"], quote=True) + ">"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    row["origin"], row["source"], row["domain"], row["language"], row["translation"],
                    row["state"], row["release_state"], format_int(row["documents"]), format_int(row["queries"]),
                    format_int(row["positive_pairs"]), format_int(row["evidence"]),
                    format_int(row["evidence_pending_review"]), row["source_path"],
                )
            )
            + "</tr>"
        )

    counts = payload["counts"]
    validation = payload["validation"]
    notes = [
        "Each source card shows its own state, document/query/positive/evidence counts. Shared candidate documents are never added into a unique-document claim.",
        "A trailing * is shown only for a source with evidence that covers fewer than all queries; the card then shows its explicit pending-review count when present.",
        "BrowseComp+ evidence is document-level qrels and is shown separately; it is never counted as span evidence E.",
        "English inferred fields and BrowseComp corpus_language_unknown are explicit metadata flags. A frozen/current release state does not mean exhaustive annotation or completed model evaluation.",
    ]
    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCI-Bench 数据矩阵总览</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;padding:26px;background:#f5f7fb;color:#17283d;font:14px system-ui,-apple-system,Segoe UI,sans-serif}}
h1{{margin:0 0 8px;font-size:28px}} h2{{margin-top:30px}} .sub{{color:#566d87;line-height:1.6}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}} .stats span{{background:#fff;border:1px solid #d8e1eb;border-radius:9px;padding:10px 14px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:end;background:#fff;padding:14px;border:1px solid #d8e1eb;border-radius:10px;margin:16px 0}}
.controls label{{display:flex;flex-direction:column;gap:4px;color:#536a84;font-size:12px}} select,input{{padding:7px;border:1px solid #b9c8d9;border-radius:6px;background:#fff;color:#17283d}}
.wrap{{overflow:auto;border:1px solid #d8e1eb;border-radius:10px;background:#fff}} table{{border-collapse:separate;border-spacing:0;min-width:1700px;width:100%}}
th,td{{border-right:1px solid #e3e9f0;border-bottom:1px solid #e3e9f0;padding:10px;vertical-align:top}} thead th{{position:sticky;top:0;background:#1a3554;color:white;text-align:left;z-index:3}}
.domain-label{{position:sticky;left:0;background:#edf3f9;z-index:2;min-width:145px;text-align:left}} thead th:first-child{{z-index:4}}
.source-card{{border:1px solid #b6c6d9;border-radius:8px;padding:9px;margin:0 0 8px;background:#f8fbff;min-width:245px}} .source-card:last-child{{margin-bottom:0}}
.source-card strong{{display:block;font-size:13px;margin-bottom:4px;word-break:break-word}} .source-card p{{margin:5px 0;line-height:1.4;font-size:12px}}
.source-card .origin{{display:inline-block;background:#e5edf7;border-radius:4px;padding:2px 5px;font:11px ui-monospace,monospace}} .meta{{color:#536a84}}
.warning{{color:#a23a27;font-weight:600}} .empty-cell{{min-width:245px;text-align:center;color:#9aa8b8;padding-top:28px}} small{{display:block;color:#72849a;margin-top:4px}}
details{{font-size:11px;margin-top:8px}} code{{word-break:break-all;color:#344d68}} .notes{{background:#fff;border:1px solid #d8e1eb;border-radius:10px;padding:15px 20px;max-width:1200px;line-height:1.7}} .notes li{{margin:5px 0}}
.source-table{{min-width:1700px}} .hidden{{display:none!important}} footer{{margin-top:18px;color:#65798f;font-size:12px}}
</style></head><body>
<h1>DCI-Bench · Domain × Language 数据矩阵</h1>
<p class="sub">合并本轮 pipeline 与冻结旧协议来源。图表标签使用英文；来源、状态和核验说明保留中文。所有计数直接来自来源记录。</p>
<div class="stats"><span><b>{counts['source_views']}</b> source views</span><span><b>{format_int(counts['query_instances'])}</b> query instances</span><span><b>{len(domains)} × {len(languages)}</b> domain × language</span><span><b>{counts['pipeline_source_views']}</b> pipeline / <b>{counts['legacy_source_views']}</b> legacy</span></div>
<div class="controls">
<label>来源<input id="filter-source" type="search" placeholder="source contains…"></label>
<label>origin<select id="filter-origin"><option value="">全部</option>{html_options(origin_values)}</select></label>
<label>domain<select id="filter-domain"><option value="">全部</option>{html_options(domains)}</select></label>
<label>language<select id="filter-language"><option value="">全部</option>{html_options(languages)}</select></label>
<label>state<select id="filter-state"><option value="">全部</option>{html_options(state_values)}</select></label>
<label>release<select id="filter-release"><option value="">全部</option>{html_options(release_values)}</select></label>
<label>translation<select id="filter-translation"><option value="">全部</option>{html_options(translation_values)}</select></label>
</div>
<div class="wrap"><table id="matrix"><thead><tr><th>domain \\ language</th>{''.join(f'<th>{html.escape(language)}</th>' for language in languages)}</tr></thead><tbody>{''.join(matrix_rows)}</tbody></table></div>
<h2>来源明细</h2>
<div class="wrap"><table class="source-table" id="source-table"><thead><tr><th>origin</th><th>source</th><th>domain</th><th>language</th><th>translation</th><th>state</th><th>release</th><th>C documents</th><th>Q</th><th>positive pairs</th><th>evidence</th><th>pending review</th><th>source path</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></div>
<div class="notes"><h2>解释与边界</h2><ul>{notes_html}</ul><p>验证：source views={validation['actual_source_views']}，queries={format_int(validation['actual_query_instances'])}，source IDs unique={str(validation['source_ids_unique']).lower()}，overlap count={len(validation['overlap_sources'])}。</p><p><a href="domain_language_matrix.png">PNG</a> · <a href="domain_language_matrix.svg">SVG</a> · <a href="source_matrix.csv">source CSV</a> · <a href="domain_language_matrix.csv">matrix CSV</a> · <a href="data_matrix.json">JSON</a></p></div>
<footer>pipeline origin=当前 pipeline；legacy origin=legacy_frozen_only。旧协议冻结状态与当前 pipeline 状态均不等于穷尽标注。</footer>
<script>
const fields=['source','origin','domain','language','state','release','translation'];
function applyFilters(){{
  const wanted={{source:document.querySelector('#filter-source').value.toLowerCase(),origin:document.querySelector('#filter-origin').value,domain:document.querySelector('#filter-domain').value,language:document.querySelector('#filter-language').value,state:document.querySelector('#filter-state').value,release:document.querySelector('#filter-release').value,translation:document.querySelector('#filter-translation').value}};
  document.querySelectorAll('.source-card').forEach(card=>{{
    const keep=fields.every(f=>!wanted[f] || (card.dataset[f]||'').toLowerCase().includes(wanted[f].toLowerCase()));
    card.classList.toggle('hidden',!keep);
  }});
  document.querySelectorAll('#source-table tbody tr').forEach(row=>{{
    const values=[...row.children].map(x=>x.textContent.toLowerCase());
    const keep=(!wanted.source||values[1].includes(wanted.source))&&(!wanted.origin||values[0]===wanted.origin)&&(!wanted.domain||values[2]===wanted.domain)&&(!wanted.language||values[3]===wanted.language)&&(!wanted.translation||values[4]===wanted.translation)&&(!wanted.state||values[5]===wanted.state)&&(!wanted.release||values[6]===wanted.release);
    row.classList.toggle('hidden',!keep);
  }});
  document.querySelectorAll('#matrix td[data-cell]').forEach(cell=>{{
    const visible=[...cell.querySelectorAll('.source-card')].some(card=>!card.classList.contains('hidden'));
    cell.classList.toggle('hidden',!visible);
  }});
}}
document.querySelectorAll('.controls input,.controls select').forEach(el=>el.addEventListener('input',applyFilters));
document.querySelectorAll('.controls select').forEach(el=>el.addEventListener('change',applyFilters));
</script></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def cell_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_count": len(items),
        "origins": sorted({item["origin"] for item in items}),
        "source_ids": [item["source_id"] for item in items],
        "sources": [
            {
                "origin": item["origin"],
                "source_id": item["source_id"],
                "source": item["source"],
                "display_name": item.get("display_name", item["source"]),
                "state": item["state"],
                "release_state": item["release_state"],
                "documents": item["documents"],
                "queries": item["queries"],
                "positive_pairs": item["positive_pairs"],
                "evidence": item["evidence"],
                "evidence_queries": item["evidence_queries"],
                "evidence_star": item["evidence_star"],
                "evidence_pending_review": item["evidence_pending_review"],
            }
            for item in items
        ],
    }


def render_csv(output: Path, rows: list[dict[str, Any]], domains: list[str], languages: list[str]) -> None:
    source_fields = [
        "origin", "source_id", "source", "source_path", "dataset", "domain", "language", "domain_basis",
        "language_basis", "language_inferred", "language_note", "corpus_language_unknown", "translation",
        "translation_note", "documents", "queries", "positive_pairs", "evidence", "evidence_queries",
        "evidence_coverage_known", "evidence_star", "evidence_pending_review", "evidence_scope",
        "evidence_note", "document_evidence_qrels", "state", "release_state", "counts_basis",
        "full_counts_from_record", "revision",
    ]
    with (output / "source_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    cells = cell_records(rows)
    cell_fields = [
        "domain", "language", "source_count", "origins", "source_ids", "sources",
    ]
    with (output / "domain_language_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cell_fields)
        writer.writeheader()
        for domain in domains:
            for language in languages:
                summary = cell_summary(cells.get((domain, language), []))
                writer.writerow({"domain": domain, "language": language, **{
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in summary.items()
                }})


def render_plot(output: Path, rows: list[dict[str, Any]], domains: list[str], languages: list[str], payload: dict[str, Any]) -> None:
    # Matplotlib is imported lazily so metadata validation does not require a
    # display backend or a full scientific Python import path.  The figure uses
    # one card per source; cell-level sums and query-volume heatmaps hide shared
    # corpora and individual release states, so neither is drawn here.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    cells = cell_records(rows)
    colors = {
        "C": "#fff0c9",
        "Q+R": "#dff1e6",
        "Q+R+E": "#e2edff",
        "empty": "#edf1f5",
    }
    border = "#cbd7e5"
    x0 = 0.25
    label_w = 2.35
    english_w = 3.75
    other_w = 1.62
    gap = 0.06
    header_h = 1.05
    title_h = 1.05
    footer_h = 1.12
    card_h = 0.62
    row_gap = 0.18
    row_heights = {
        domain: max(1, max((len(cells.get((domain, language), [])) for language in languages), default=1)) * card_h + 0.22
        for domain in domains
    }
    matrix_w = label_w + english_w + max(0, len(languages) - 1) * other_w
    matrix_h = sum(row_heights.values())
    total_h = title_h + header_h + matrix_h + row_gap * max(0, len(domains) - 1) + footer_h
    fig, ax = plt.subplots(figsize=(max(25.0, matrix_w + 1.0), max(18.0, total_h + 0.8)), dpi=150)
    fig.patch.set_facecolor("#f5f7fb")
    ax.set_facecolor("#f5f7fb")
    ax.set_xlim(0, x0 + matrix_w + 0.35)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    ax.text(
        x0,
        total_h - 0.28,
        "DCI-Bench | Domain × Language source matrix",
        fontsize=19,
        fontweight="bold",
        color="#17283d",
        va="top",
    )
    ax.text(
        x0,
        total_h - 0.68,
        f"{payload['counts']['source_views']} source views · {format_int(payload['counts']['query_instances'])} query instances · each card is one source",
        fontsize=11,
        color="#52677f",
        va="top",
    )

    column_x: dict[str, float] = {}
    x = x0 + label_w
    for index, language in enumerate(languages):
        column_x[language] = x
        x += english_w if index == 0 else other_w
    header_y = total_h - title_h - 0.54
    ax.text(x0, header_y, "DOMAIN", fontsize=10, fontweight="bold", color="#52677f", va="center")
    for language in languages:
        width = english_w if language == "en" else other_w
        ax.add_patch(Rectangle((column_x[language], header_y - 0.22), width - gap, 0.44, facecolor="#1a3554", edgecolor="none"))
        ax.text(column_x[language] + (width - gap) / 2, header_y, language.upper(), ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    row_top = header_y - 0.55
    for domain in domains:
        row_height = row_heights[domain]
        row_bottom = row_top - row_height
        ax.text(x0, row_bottom + row_height / 2, domain, fontsize=11, fontweight="bold", color="#203651", va="center")
        for language in languages:
            width = english_w if language == "en" else other_w
            cell_x = column_x[language]
            items = cells.get((domain, language), [])
            ax.add_patch(Rectangle((cell_x, row_bottom), width - gap, row_height, facecolor="#ffffff" if items else colors["empty"], edgecolor=border, linewidth=0.8))
            if not items:
                ax.text(cell_x + (width - gap) / 2, row_bottom + row_height / 2, "—", ha="center", va="center", fontsize=16, color="#9aa8b8")
                continue
            usable_card_h = min(card_h, (row_height - 0.12) / len(items))
            for item_index, item in enumerate(items):
                card_y = row_top - 0.07 - (item_index + 1) * usable_card_h
                state = item["state"].rstrip("*")
                card_color = colors.get(state, colors["empty"])
                card_width = width - gap - 0.12
                ax.add_patch(FancyBboxPatch(
                    (cell_x + 0.06, card_y + 0.04),
                    card_width,
                    max(0.05, usable_card_h - 0.08),
                    boxstyle="round,pad=0.018",
                    facecolor=card_color,
                    edgecolor="#b8c8d9",
                    linewidth=0.55,
                ))
                origin = "P" if item["origin"] == "pipeline" else "L"
                label = f"{origin} {format_plot_source(item)} [{item['state']}]"
                card_text = f"{label}\n{format_int(item['documents'])} D / {format_int(item['queries'])} Q"
                if item["evidence_star"] and item["evidence_pending_review"]:
                    card_text += f"\n{format_int(item['evidence'])} E · review {format_int(item['evidence_pending_review'])}"
                ax.text(
                    cell_x + 0.06 + card_width / 2,
                    card_y + usable_card_h / 2,
                    card_text,
                    ha="center",
                    va="center",
                    fontsize=8.1 if language == "en" else 7.25,
                    linespacing=1.05,
                    color="#17324f",
                )
        row_top = row_bottom - row_gap

    legend_y = 0.60
    legend_x = x0
    ax.text(legend_x, legend_y + 0.27, "Status", fontsize=10, fontweight="bold", color="#203651", va="center")
    for state, label in (("C", "corpus"), ("Q+R", "query + qrels"), ("Q+R+E", "explicit evidence")):
        ax.add_patch(FancyBboxPatch((legend_x, legend_y - 0.03), 0.58, 0.34, boxstyle="round,pad=0.02", facecolor=colors[state], edgecolor="#b8c8d9", linewidth=0.5))
        ax.text(legend_x + 0.08, legend_y + 0.14, state, fontsize=8.5, fontweight="bold", color="#17324f", va="center")
        ax.text(legend_x + 0.70, legend_y + 0.14, label, fontsize=9, color="#52677f", va="center")
        legend_x += 2.25 if state != "Q+R+E" else 2.80
    ax.text(legend_x, legend_y + 0.14, "P = pipeline  ·  L = legacy  ·  T = translated  ·  * = evidence covers only part of queries", fontsize=9, color="#52677f", va="center")
    ax.text(x0, 0.18, "EN is widened for the multi-source enterprise cell. Exact counts and legacy language notes are in the HTML/CSV/JSON.", fontsize=8.7, color="#52677f", va="center")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for name in ("domain_language_matrix.png", "matrix.png"):
        fig.savefig(output / name, dpi=150)
    for name in ("domain_language_matrix.svg", "matrix.svg"):
        fig.savefig(output / name)
    plt.close(fig)


def build_payload(rows: list[dict[str, Any]], pipeline_count: int, legacy_count: int, overlap_sources: list[str]) -> tuple[dict[str, Any], list[str], list[str]]:
    domains = ordered_values(PREFERRED_DOMAINS, (row["domain"] for row in rows))
    languages = ordered_values(PREFERRED_LANGUAGES, (row["language"] for row in rows))
    pipeline_rows = [row for row in rows if row["origin"] == "pipeline"]
    legacy_rows = [row for row in rows if row["origin"] == "legacy_frozen_only"]
    counts = {
        "source_views": len(rows),
        "pipeline_source_views": len(pipeline_rows),
        "legacy_source_views": len(legacy_rows),
        "query_instances": sum(row["queries"] for row in rows),
        "pipeline_query_instances": sum(row["queries"] for row in pipeline_rows),
        "legacy_query_instances": sum(row["queries"] for row in legacy_rows),
        "document_task_entries": sum(row["documents"] for row in rows),
        "positive_pairs": sum(row["positive_pairs"] for row in rows),
        "evidence_items": sum(row["evidence"] for row in rows),
        "evidence_pending_review": sum(row["evidence_pending_review"] for row in rows),
    }
    cells = cell_records(rows)
    matrix = []
    for domain in domains:
        for language in languages:
            matrix.append({"domain": domain, "language": language, **cell_summary(cells.get((domain, language), []))})
    validation = {
        "expected_source_views": EXPECTED_SOURCE_VIEWS,
        "actual_source_views": len(rows),
        "expected_query_instances": EXPECTED_QUERY_INSTANCES,
        "actual_query_instances": counts["query_instances"],
        "pipeline_count_from_input": pipeline_count,
        "legacy_count_selected": legacy_count,
        "source_ids_unique": len({row["source_id"] for row in rows}) == len(rows),
        "overlap_sources": overlap_sources,
        "full_counts_from_source_records": all(row["full_counts_from_record"] for row in rows),
        "candidate_pool_not_unique_document_claim": True,
    }
    payload = {
        "schema_version": "dci-data-overview-v2",
        "domains": domains,
        "languages": languages,
        "counts": counts,
        "validation": validation,
        "sources": rows,
        "matrix": matrix,
        "notes": [
            "The 23 pipeline source records and 8 selected legacy records are source views; they are not a claim of exhaustive annotation.",
            "Legacy BrowseComp+ evidence is document-level qrels and is excluded from span evidence E.",
            "English inferred language fields and BrowseComp corpus_language_unknown are explicit metadata.",
            "Each domain-language cell contains independent source records; shared candidate documents are not presented as a unique-document total.",
        ],
    }
    return payload, domains, languages


def load_legacy_rows(path: Path, pipeline_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    records = load_source_list(path)
    by_task = {str(record.get("task")): record for record in records}
    missing = [task for task in LEGACY_TASKS if task not in by_task]
    if missing:
        raise ValueError(f"legacy index missing required non-overlap tasks: {', '.join(missing)}")

    pipeline_aliases: set[str] = set()
    for row in pipeline_rows:
        pipeline_aliases.update(source_aliases({"source": row["source"], "source_id": row["source_id"], "path": row["source_path"], "dataset": row["dataset"]}))
    legacy_rows: list[dict[str, Any]] = []
    overlaps: list[str] = []
    for task_name, spec in LEGACY_TASKS.items():
        record = by_task[task_name]
        aliases = source_aliases(record)
        aliases.add(slug(task_name))
        if aliases & pipeline_aliases:
            overlaps.append(task_name)
        legacy_rows.append(normalize_legacy_record(record, spec))
    return legacy_rows, overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-json", type=Path, required=True, help="23-source pipeline matrix JSON")
    parser.add_argument("--legacy-index", type=Path, required=True, help="frozen document_tasks.json")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--allow-count-mismatch", action="store_true", help="render while recording expected-count mismatches")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    pipeline_records = load_source_list(args.pipeline_json)
    pipeline_rows = [normalize_pipeline_record(record) for record in pipeline_records]
    pipeline_ids = [row["source_id"] for row in pipeline_rows]
    if len(set(pipeline_ids)) != len(pipeline_ids):
        duplicates = sorted({source_id for source_id in pipeline_ids if pipeline_ids.count(source_id) > 1})
        raise ValueError(f"duplicate pipeline source IDs: {duplicates}")

    legacy_rows, overlaps = load_legacy_rows(args.legacy_index, pipeline_rows)
    if overlaps:
        raise ValueError(f"legacy/pipeline overlap detected; refusing duplicate source views: {overlaps}")
    rows = pipeline_rows + legacy_rows
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate source IDs after pipeline + legacy merge")

    payload, domains, languages = build_payload(rows, len(pipeline_rows), len(legacy_rows), overlaps)
    validation = payload["validation"]
    mismatch = (
        validation["actual_source_views"] != EXPECTED_SOURCE_VIEWS
        or validation["actual_query_instances"] != EXPECTED_QUERY_INSTANCES
        or not validation["full_counts_from_source_records"]
    )
    if mismatch and not args.allow_count_mismatch:
        raise ValueError(
            "matrix acceptance mismatch: "
            f"source_views={validation['actual_source_views']} (expected {EXPECTED_SOURCE_VIEWS}), "
            f"query_instances={validation['actual_query_instances']} (expected {EXPECTED_QUERY_INSTANCES})"
        )

    render_csv(args.output, rows, domains, languages)
    render_html(args.output, rows, domains, languages, payload)
    render_plot(args.output, rows, domains, languages, payload)
    (args.output / "data_matrix.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_hashes = {
        path.name: {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": payload["schema_version"],
        "inputs": {
            "pipeline_json": {"path": str(args.pipeline_json), "sha256": sha256_file(args.pipeline_json)},
            "legacy_index": {"path": str(args.legacy_index), "sha256": sha256_file(args.legacy_index)},
        },
        "outputs": output_hashes,
        "manifest_self_hash_excluded": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["counts"], "domains": domains, "languages": languages}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
