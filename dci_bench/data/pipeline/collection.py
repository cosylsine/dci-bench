"""Leakage-safe split assignment for a collection of DCI bundles.

``quality.assign_splits`` deliberately operates on one bundle and therefore
cannot distinguish two equal source IDs.  This module builds one temporary,
namespaced bundle, delegates the actual component assignment to that helper,
and maps the result back to ``(source_index, original_id)`` records.  The
temporary bundle is never exposed and the caller's bundles are never changed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .quality import (
    _translation_class as _quality_translation_class,
    assign_splits,
    coverage_report as _quality_coverage_report,
)


COLLECTION_SPLIT_VERSION = "dci-data-collection-split-v1"
COLLECTION_COVERAGE_VERSION = "dci-data-collection-coverage-v1"


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _internal_id(kind: str, source_index: int, original_id: str) -> str:
    """Encode a source-qualified ID without assuming anything about ID text."""

    encoded = json.dumps(original_id, ensure_ascii=False, separators=(",", ":"))
    return f"__dci_collection__:{kind}:{source_index}:{encoded}"


class _UnionFind:
    def __init__(self, nodes: list[tuple[str, int, str]]) -> None:
        self.parent: dict[tuple[str, int, str], tuple[str, int, str]] = {node: node for node in nodes}
        self.rank: dict[tuple[str, int, str], int] = {node: 0 for node in nodes}

    def find(self, node: tuple[str, int, str]) -> tuple[str, int, str]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: tuple[str, int, str], right: tuple[str, int, str]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def _public_key(source_index: int, original_id: str) -> str:
    return f"{source_index}:{json.dumps(original_id, ensure_ascii=False, separators=(',', ':'))}"


def _safe_copy(value: Any) -> Any:
    """Copy source metadata while keeping the returned report JSON-friendly."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_copy(item) for item in value]
    try:
        return _safe_copy(copy.deepcopy(value))
    except Exception:  # pragma: no cover - defensive for unusual host objects
        return repr(value)


def _record_ref(node: tuple[str, int, str]) -> dict[str, Any]:
    _, source_index, original_id = node
    return {"source_index": source_index, "original_id": original_id, "id": original_id}


def _validate_and_index(
    bundles: list[dict],
) -> tuple[
    dict[tuple[str, int, str], Mapping[str, Any]],
    dict[tuple[str, int, str], str],
    dict[tuple[int, str], list[tuple[str, int, str]]],
    list[dict[str, Any]],
]:
    records: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    internal_ids: dict[tuple[str, int, str], str] = {}
    by_source_id: dict[tuple[int, str], list[tuple[str, int, str]]] = defaultdict(list)
    source_rows: list[dict[str, Any]] = []

    for source_index, bundle in enumerate(bundles):
        if not isinstance(bundle, dict):
            raise TypeError(f"bundles[{source_index}] must be an ordinary dict")
        source_rows.append(
            {
                "source_index": source_index,
                "source": _safe_copy(bundle.get("source", {})) if isinstance(bundle.get("source", {}), Mapping) else {},
            }
        )
        for kind, field in (("document", "documents"), ("query", "queries")):
            values = bundle.get(field, [])
            if values is None:
                values = []
            if not isinstance(values, list):
                raise TypeError(f"bundles[{source_index}].{field} must be a list")
            seen: set[str] = set()
            for row_index, record in enumerate(values):
                if not isinstance(record, Mapping):
                    raise TypeError(f"bundles[{source_index}].{field}[{row_index}] must be a mapping")
                original_id = record.get("_id")
                if not _valid_id(original_id):
                    raise ValueError(f"bundles[{source_index}].{field}[{row_index}] has an invalid _id")
                if original_id in seen:
                    raise ValueError(f"duplicate {kind} ID {original_id!r} in source {source_index}")
                seen.add(original_id)
                node = (kind, source_index, original_id)
                records[node] = record
                internal_ids[node] = _internal_id(kind, source_index, original_id)
                by_source_id[(source_index, original_id)].append(node)
    return records, internal_ids, by_source_id, source_rows


def _relation_components(
    records: Mapping[tuple[str, int, str], Mapping[str, Any]],
    by_source_id: Mapping[tuple[int, str], list[tuple[str, int, str]]],
) -> dict[tuple[str, int, str], str]:
    """Return a stable synthetic family key for explicit cross-source links."""

    nodes = sorted(records)
    union_find = _UnionFind(nodes)
    relation_groups: defaultdict[tuple[str, str], list[tuple[str, int, str]]] = defaultdict(list)

    for node, record in records.items():
        metadata = _metadata(record)
        for field in ("family_id", "group_id"):
            value = metadata.get(field)
            if _valid_id(value):
                relation_groups[(field, value)].append(node)

        translation_of = metadata.get("translation_of")
        if _valid_id(translation_of):
            relation_groups[("translation_of", translation_of)].append(node)
            # A local target is preferred by the source contract.  If no
            # local target exists, the explicit reference may identify a
            # target in another source; connect all matching source records.
            local_targets = by_source_id.get((node[1], translation_of), [])
            if local_targets:
                targets = local_targets
            else:
                targets = [candidate for (source, original), members in by_source_id.items() if original == translation_of for candidate in members]
            relation_groups[("translation_of", translation_of)].extend(targets)

    for members in relation_groups.values():
        unique_members = sorted(set(members))
        if not unique_members:
            continue
        first = unique_members[0]
        for member in unique_members[1:]:
            union_find.union(first, member)

    relation_nodes = {member for members in relation_groups.values() for member in members}
    grouped: defaultdict[tuple[str, int, str], list[tuple[str, int, str]]] = defaultdict(list)
    for node in relation_nodes:
        grouped[union_find.find(node)].append(node)

    result: dict[tuple[str, int, str], str] = {}
    for root, members in grouped.items():
        stable_members = sorted(set(members))
        digest = hashlib.sha256(
            json.dumps(stable_members, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        key = f"collection-relation:{digest}"
        for member in stable_members:
            result[member] = key
    return result


def _translation_class(record: Mapping[str, Any]) -> str:
    """Use the quality audit's explicit native/translated/unknown semantics."""

    return _quality_translation_class(record)


def _query_translation_counts(queries: Any) -> dict[str, int]:
    counts = {"native": 0, "translated": 0, "unknown": 0}
    if not isinstance(queries, list):
        return counts
    for record in queries:
        category = _translation_class(record) if isinstance(record, Mapping) else "unknown"
        if category not in counts:
            category = "unknown"
        counts[category] += 1
    return counts


def assign_collection_splits(bundles: list[dict], seed: int = 20261022) -> dict:
    """Assign global leakage-safe splits across multiple source bundles.

    IDs are namespaced by source index before calling :func:`assign_splits`.
    Exact qrels/evidence links, duplicate text, and explicit family/group/
    translation metadata therefore connect nodes without treating equal raw
    IDs as equal.  A returned split map always carries ``source_index`` and
    ``original_id``; per-source ``assign_splits`` results must not be used as a
    substitute for this global result.
    """

    if not isinstance(bundles, list):
        raise TypeError("bundles must be a list")
    records, internal_ids, by_source_id, source_rows = _validate_and_index(bundles)
    relation_keys = _relation_components(records, by_source_id)

    aggregate_documents: list[dict[str, Any]] = []
    aggregate_queries: list[dict[str, Any]] = []
    reverse_internal: dict[str, tuple[str, int, str]] = {}
    for node in sorted(records):
        kind, _, _ = node
        clone = dict(records[node])
        clone["_id"] = internal_ids[node]
        metadata = _metadata(records[node])
        if node in relation_keys:
            # All explicit cross-source relations have already been resolved
            # into one component.  Using one synthetic family key lets the
            # existing splitter enforce those links without mutating source
            # metadata or losing any of the original identifiers.
            metadata["family_id"] = relation_keys[node]
        clone["metadata"] = metadata
        reverse_internal[internal_ids[node]] = node
        if kind == "document":
            aggregate_documents.append(clone)
        else:
            aggregate_queries.append(clone)

    aggregate_qrels: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    for source_index, bundle in enumerate(bundles):
        qrels = bundle.get("qrels", {})
        if qrels is None:
            qrels = {}
        if not isinstance(qrels, Mapping):
            warnings.append({"code": "invalid_qrels", "source_index": source_index})
            continue
        for query_id, row in qrels.items():
            query_node = ("query", source_index, query_id) if _valid_id(query_id) else None
            if query_node not in records:
                warnings.append({"code": "unknown_qrel_query", "source_index": source_index, "original_id": query_id})
                continue
            if not isinstance(row, Mapping):
                warnings.append({"code": "invalid_qrel_row", "source_index": source_index, "original_id": query_id})
                continue
            converted: dict[str, Any] = {}
            for document_id, score in row.items():
                document_node = ("document", source_index, document_id) if _valid_id(document_id) else None
                if document_node not in records:
                    warnings.append(
                        {"code": "unknown_qrel_document", "source_index": source_index, "original_id": document_id}
                    )
                    continue
                converted[internal_ids[document_node]] = score
            aggregate_qrels[internal_ids[query_node]] = converted

    aggregate_evidence: list[dict[str, Any]] = []
    for source_index, bundle in enumerate(bundles):
        evidence = bundle.get("evidence", [])
        if evidence is None:
            evidence = []
        if not isinstance(evidence, list):
            warnings.append({"code": "invalid_evidence", "source_index": source_index})
            continue
        for row_index, item in enumerate(evidence):
            if not isinstance(item, Mapping):
                warnings.append({"code": "invalid_evidence_row", "source_index": source_index, "row": row_index})
                continue
            query_id = item.get("query_id")
            document_id = item.get("doc_id")
            query_node = ("query", source_index, query_id) if _valid_id(query_id) else None
            document_node = ("document", source_index, document_id) if _valid_id(document_id) else None
            if query_node not in records or document_node not in records:
                warnings.append(
                    {
                        "code": "unknown_evidence_link",
                        "source_index": source_index,
                        "query_id": query_id,
                        "doc_id": document_id,
                    }
                )
                continue
            converted = dict(item)
            converted["query_id"] = internal_ids[query_node]
            converted["doc_id"] = internal_ids[document_node]
            aggregate_evidence.append(converted)

    aggregate = {
        "documents": aggregate_documents,
        "queries": aggregate_queries,
        "qrels": aggregate_qrels,
        "evidence": aggregate_evidence,
    }
    split = assign_splits(aggregate, seed=seed)

    def converted_nodes(values: list[str], split_name: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for internal in values:
            node = reverse_internal[internal]
            row = _record_ref(node)
            if split_name is not None:
                row["split"] = split_name
            rows.append(row)
        return rows

    query_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    query_splits: dict[str, dict[str, str]] = defaultdict(dict)
    document_splits: dict[str, dict[str, str]] = defaultdict(dict)
    query_split_map: dict[str, str] = {}
    document_split_map: dict[str, str] = {}
    for internal, split_name in sorted(split["query_splits"].items()):
        node = reverse_internal[internal]
        row = _record_ref(node)
        row["split"] = split_name
        query_rows.append(row)
        query_splits[str(node[1])][node[2]] = split_name
        query_split_map[_public_key(node[1], node[2])] = split_name
    for internal, split_name in sorted(split["document_splits"].items()):
        node = reverse_internal[internal]
        row = _record_ref(node)
        row["split"] = split_name
        document_rows.append(row)
        document_splits[str(node[1])][node[2]] = split_name
        document_split_map[_public_key(node[1], node[2])] = split_name

    components: list[dict[str, Any]] = []
    for component in split["components"]:
        row = {key: value for key, value in component.items() if key not in {"queries", "documents"}}
        query_refs = converted_nodes(component.get("queries", []))
        document_refs = converted_nodes(component.get("documents", []))
        row["queries"] = query_refs
        row["documents"] = document_refs
        row["source_indices"] = sorted(
            {ref["source_index"] for ref in query_refs + document_refs}
        )
        row["cross_source"] = len(row["source_indices"]) > 1
        components.append(row)
    component_by_id = {row["component_id"]: row for row in components}
    large_components = [component_by_id[row["component_id"]] for row in split.get("large_components", [])]
    for row in large_components:
        row["large"] = True

    cross_source_components = [row for row in components if row["cross_source"]]
    sizes = [int(row.get("size", 0)) for row in components]
    component_stats = {
        "count": len(components),
        "total_nodes": sum(sizes),
        "largest_size": max(sizes, default=0),
        "cross_source_count": len(cross_source_components),
        "single_source_count": len(components) - len(cross_source_components),
        "large_count": len(large_components),
    }

    per_source: list[dict[str, Any]] = []
    for source_index, bundle in enumerate(bundles):
        documents = bundle.get("documents", []) if isinstance(bundle.get("documents", []), list) else []
        queries = bundle.get("queries", []) if isinstance(bundle.get("queries", []), list) else []
        translation_counts = _query_translation_counts(queries)
        per_source.append(
            {
                "source_index": source_index,
                "documents": len(documents),
                "queries": len(queries),
                "native_queries": translation_counts["native"],
                "translated_queries": translation_counts["translated"],
                "unknown_queries": translation_counts["unknown"],
            }
        )

    return _safe_copy({
        "split_version": COLLECTION_SPLIT_VERSION,
        "seed": seed,
        "source_count": len(bundles),
        "sources": source_rows,
        "query_splits": {key: dict(value) for key, value in sorted(query_splits.items())},
        "document_splits": {key: dict(value) for key, value in sorted(document_splits.items())},
        "query_split_rows": query_rows,
        "document_split_rows": document_rows,
        "query_split_map": query_split_map,
        "document_split_map": document_split_map,
        "components": components,
        "component_count": len(components),
        "component_stats": component_stats,
        "cross_source_components": cross_source_components,
        "large_components": large_components,
        "counts": split["counts"],
        "per_source": per_source,
        "warnings": warnings
        + [
            {
                "code": "large_shared_component",
                "message": "a global component is large; requested ratios were not enforced by splitting it",
                "component_id": row["component_id"],
                "size": row["size"],
            }
            for row in large_components
        ],
    })


def collection_coverage(bundles: list[dict]) -> dict:
    """Extend the ordinary quality coverage report with query translation counts."""

    if not isinstance(bundles, list):
        raise TypeError("bundles must be a list")
    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, dict):
            raise TypeError(f"bundles[{index}] must be an ordinary dict")
    report = _quality_coverage_report(bundles)
    translation_totals = {"native": 0, "translated": 0, "unknown": 0}
    source_rows: list[dict[str, Any]] = []
    for source_index, bundle in enumerate(bundles):
        queries = bundle.get("queries", []) if isinstance(bundle.get("queries", []), list) else []
        counts = _query_translation_counts(queries)
        for category, count in counts.items():
            translation_totals[category] += count
        source_rows.append(
            {
                "source_index": source_index,
                "queries": len(queries),
                "native_queries": counts["native"],
                "translated_queries": counts["translated"],
                "unknown_queries": counts["unknown"],
            }
        )
        if source_index < len(report.get("bundles", [])) and isinstance(report["bundles"][source_index], dict):
            report["bundles"][source_index].update(
                {
                    "native_queries": counts["native"],
                    "translated_queries": counts["translated"],
                    "unknown_queries": counts["unknown"],
                }
            )
    report["collection_coverage_version"] = COLLECTION_COVERAGE_VERSION
    report["query_counts"] = {
        "total": sum(translation_totals.values()),
        **translation_totals,
    }
    report["native_query_count"] = translation_totals["native"]
    report["translated_query_count"] = translation_totals["translated"]
    report["unknown_query_count"] = translation_totals["unknown"]
    report["source_query_counts"] = source_rows
    return report


# A short alias makes the collection-specific coverage function discoverable
# alongside quality.coverage_report without changing the existing API.
coverage_report = collection_coverage


__all__ = [
    "COLLECTION_COVERAGE_VERSION",
    "COLLECTION_SPLIT_VERSION",
    "assign_collection_splits",
    "collection_coverage",
    "coverage_report",
]
