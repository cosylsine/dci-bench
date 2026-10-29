"""Deterministic task-local aliases for document identifiers.

Document identifiers in source datasets are often long, contain URL-unfriendly
characters, or expose information that should stay in the host-side metadata.
New document workspaces use the aliases defined here.  The source-to-alias
mapping is deterministic for the complete source corpus and is independent of
qrels, query selection, and model rankings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any


# This is intentionally a stable, JSON-friendly identifier rather than a
# Python implementation detail.  It is recorded in generated manifests so a
# verifier can distinguish new short-ID artifacts from frozen legacy artifacts.
DOCUMENT_ID_POLICY = "short-document-id-v1"
DOCUMENT_ID_PREFIX = "d"
DOCUMENT_ID_MIN_DIGITS = 6

_ALIAS_RE = re.compile(r"^d[0-9]{6,}$")


def _validate_original_ids(original_ids: Iterable[str]) -> list[str]:
    """Return unique source IDs while rejecting malformed or duplicate input."""

    values: list[str] = []
    seen: set[str] = set()
    for value in original_ids:
        if not isinstance(value, str) or not value:
            raise ValueError(f"document IDs must be non-empty strings: {value!r}")
        if value in seen:
            raise ValueError(f"duplicate document ID: {value!r}")
        seen.add(value)
        values.append(value)
    return values


def build_document_id_map(original_ids: Iterable[str]) -> dict[str, str]:
    """Build an original-ID -> short-alias mapping.

    Source IDs are sorted by their UTF-8 byte representation before aliases are
    assigned.  The numeric portion is one-based and padded to at least six
    digits (``d000001``, ``d000002``, ...).  The complete input inventory must
    be supplied before any query/pool filtering so aliases never depend on
    qrels or retrieval order.
    """

    values = _validate_original_ids(original_ids)
    ordered = sorted(values, key=lambda value: value.encode("utf-8"))
    width = max(DOCUMENT_ID_MIN_DIGITS, len(str(len(ordered))))
    return {
        original: f"{DOCUMENT_ID_PREFIX}{ordinal:0{width}d}"
        for ordinal, original in enumerate(ordered, start=1)
    }


def validate_document_id_map(
    original_to_alias: Mapping[str, str],
    *,
    require_canonical: bool = True,
) -> dict[str, str]:
    """Validate and return a copy of an original-ID -> alias mapping.

    A canonical map must use the aliases produced by
    :func:`build_document_id_map`; this catches reordered or partially rebuilt
    inventories.  Set ``require_canonical=False`` only when validating a
    legacy/externally defined mapping whose aliases are otherwise known to be
    safe.
    """

    if not isinstance(original_to_alias, Mapping):
        raise ValueError("document ID mapping must be an object")
    copied: dict[str, str] = {}
    aliases: set[str] = set()
    for original, alias in original_to_alias.items():
        if not isinstance(original, str) or not original:
            raise ValueError(f"mapping source IDs must be non-empty strings: {original!r}")
        if not isinstance(alias, str) or not alias:
            raise ValueError(f"mapping aliases must be non-empty strings: {alias!r}")
        if alias in aliases:
            raise ValueError(f"mapping aliases are not unique: {alias!r}")
        if not _ALIAS_RE.fullmatch(alias):
            raise ValueError(f"invalid short document alias: {alias!r}")
        aliases.add(alias)
        copied[original] = alias
    if require_canonical and copied != build_document_id_map(copied):
        raise ValueError("document ID mapping is not canonical for its source IDs")
    return copied


def reverse_document_id_map(
    original_to_alias: Mapping[str, str],
    *,
    require_canonical: bool = True,
) -> dict[str, str]:
    """Return the host-side alias -> original mapping after validation."""

    validated = validate_document_id_map(
        original_to_alias,
        require_canonical=require_canonical,
    )
    return {alias: original for original, alias in sorted(validated.items())}


def validate_source_document_ids(
    alias_to_original: Mapping[str, str],
    *,
    require_canonical: bool = True,
) -> dict[str, str]:
    """Validate an alias -> original mapping stored in ``source_doc_ids.json``."""

    if not isinstance(alias_to_original, Mapping):
        raise ValueError("source document IDs must be an object")
    reverse = {}
    originals: list[str] = []
    aliases: set[str] = set()
    for alias, original in alias_to_original.items():
        if not isinstance(alias, str) or not alias or not _ALIAS_RE.fullmatch(alias):
            raise ValueError(f"invalid short document alias: {alias!r}")
        if not isinstance(original, str) or not original:
            raise ValueError(f"source document IDs must be non-empty strings: {original!r}")
        if alias in aliases:
            raise ValueError(f"duplicate short document alias: {alias!r}")
        aliases.add(alias)
        originals.append(original)
        reverse[alias] = original
    if len(set(originals)) != len(originals):
        raise ValueError("source document mapping is not one-to-one")
    if require_canonical:
        expected = reverse_document_id_map(build_document_id_map(originals))
        if reverse != expected:
            raise ValueError("source document mapping is not canonical for its source IDs")
    return reverse


def remap_qrels(
    qrels: Mapping[str, Mapping[str, Any]],
    original_to_alias: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Replace source document IDs in qrels with their short aliases.

    Query IDs, scores, and qrel insertion semantics are preserved; only the
    document key is changed.  Every referenced source ID must be present in the
    complete mapping, which prevents silently dropping a positive judgment.
    """

    mapping = validate_document_id_map(original_to_alias)
    if not isinstance(qrels, Mapping):
        raise ValueError("qrels must be an object")
    result: dict[str, dict[str, Any]] = {}
    for query_id, judgments in qrels.items():
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"qrel query IDs must be non-empty strings: {query_id!r}")
        if not isinstance(judgments, Mapping):
            raise ValueError(f"qrels for {query_id!r} must be an object")
        remapped: dict[str, Any] = {}
        for original, score in judgments.items():
            if original not in mapping:
                raise ValueError(f"qrel references an unmapped source document: {original!r}")
            alias = mapping[original]
            if alias in remapped:
                raise ValueError(f"duplicate remapped qrel document: {alias!r}")
            remapped[alias] = score
        result[query_id] = remapped
    return result


def restore_ranking(
    ranked_aliases: Iterable[str],
    alias_to_original: Mapping[str, str],
) -> list[str]:
    """Restore a model ranking from aliases to source document IDs."""

    mapping = validate_source_document_ids(alias_to_original)
    ranking: list[str] = []
    seen: set[str] = set()
    for alias in ranked_aliases:
        if not isinstance(alias, str) or alias not in mapping:
            raise ValueError(f"ranking references an unknown document alias: {alias!r}")
        if alias in seen:
            raise ValueError(f"ranking contains duplicate document alias: {alias!r}")
        seen.add(alias)
        ranking.append(mapping[alias])
    return ranking


__all__ = [
    "DOCUMENT_ID_POLICY",
    "DOCUMENT_ID_PREFIX",
    "DOCUMENT_ID_MIN_DIGITS",
    "build_document_id_map",
    "validate_document_id_map",
    "reverse_document_id_map",
    "validate_source_document_ids",
    "remap_qrels",
    "restore_ranking",
]
