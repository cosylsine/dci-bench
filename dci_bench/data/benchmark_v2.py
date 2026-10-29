"""Data contracts and deterministic preparation for the benchmark v2 suite.

The v2 data contract is intentionally host side.  It samples query IDs from
already prepared source metadata, derives a 200-document pool from the saved
1000-document pool, and writes references to the source shared corpus.  It
never downloads data, starts a retriever, or copies document bodies.

The functions in this module accept small in-memory records as well as the
metadata shape emitted by the existing retrieval-pool builder.  Keeping the
sampling and pool selection functions independent from the filesystem makes
the rules testable without touching the existing user-owned pools.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Protocol
import unicodedata
import uuid


SCHEMA_VERSION = "dci-benchmark-v2-20261029"
SAMPLING_ALGORITHM_VERSION = "random-without-replacement-v1"
POOL_DERIVATION_VERSION = "nested-ranked-pool-v1"
DEFAULT_SEED = 42
DEV_SIZE = 10
TEST_SIZE = 50
SMOKE_SIZE = 1
LITE_CAPACITY = 200
FULL_CAPACITY = 1000
DOMAINS = (
    "encyclopedia",
    "general_qa",
    "scientific",
    "medical",
    "finance",
    "legal",
    "enterprise",
    "web",
    "software",
    "argumentation",
)
LANGUAGES = ("en", "zh")
TASK_IDS = tuple(f"{domain}_{language}" for domain in DOMAINS for language in LANGUAGES)


class SamplingStrategy(Protocol):
    """Protocol implemented by query sampling strategies."""

    name: str

    def sample(
        self,
        query_ids: Sequence[str],
        count: int,
        *,
        seed: int,
        task_id: str,
    ) -> list[str]: ...


class SamplingShortageError(ValueError):
    """Raised when the requested split cannot be formed without replacement."""


class SourceBlockedError(ValueError):
    """Raised when a source lacks the evidence required by the v2 contract."""


def stable_json_sha256(value: Any) -> str:
    """Hash a JSON value independently of mapping order and whitespace."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest independently of its self-reported digest."""

    without_hash = dict(manifest)
    without_hash.pop("overall_sha256", None)
    return stable_json_sha256(without_hash)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_id(record: Mapping[str, Any] | str) -> str:
    if isinstance(record, str):
        query_id = record
    elif isinstance(record, Mapping):
        query_id = record.get("query_id", record.get("_id", record.get("id")))
    else:
        query_id = None
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"query record has no non-empty query ID: {record!r}")
    return query_id


def _records(records: Iterable[Mapping[str, Any] | str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        query_id = _query_id(record)
        if query_id in seen:
            raise ValueError(f"duplicate query ID: {query_id!r}")
        seen.add(query_id)
        rows.append(dict(record) if isinstance(record, Mapping) else {"query_id": record})
    rows.sort(key=lambda row: _query_id(row))
    return rows


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _validate_count(count: int, name: str) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _rng(seed: int, scope: str) -> random.Random:
    _validate_seed(seed)
    digest = hashlib.sha256(f"{SAMPLING_ALGORITHM_VERSION}:{seed}:{scope}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _random_without_replacement(
    query_ids: Sequence[str], count: int, *, seed: int, task_id: str
) -> list[str]:
    _validate_count(count, "count")
    if count > len(query_ids):
        raise SamplingShortageError(
            f"requested {count} query IDs for {task_id!r}, only {len(query_ids)} available"
        )
    ordered = sorted(query_ids)
    return _rng(seed, task_id).sample(ordered, count)


class RandomWithoutReplacement:
    """Default deterministic sampler used by benchmark v2."""

    name = "random_without_replacement"
    algorithm_version = SAMPLING_ALGORITHM_VERSION

    def sample(
        self,
        query_ids: Sequence[str],
        count: int,
        *,
        seed: int,
        task_id: str,
    ) -> list[str]:
        return _random_without_replacement(query_ids, count, seed=seed, task_id=task_id)


SamplingFunction = Callable[..., Sequence[str]]
_SAMPLING_STRATEGIES: dict[str, Any] = {
    "random_without_replacement": RandomWithoutReplacement(),
}


def register_sampling_strategy(name: str, strategy: Any) -> None:
    """Register a named strategy for callers that need another sampler."""

    if not isinstance(name, str) or not name:
        raise ValueError("sampling strategy name must be a non-empty string")
    if not callable(strategy) and not hasattr(strategy, "sample"):
        raise TypeError("sampling strategy must be callable or implement sample()")
    _SAMPLING_STRATEGIES[name] = strategy


def _resolve_strategy(strategy: str | Any) -> tuple[str, Any]:
    if isinstance(strategy, str):
        try:
            implementation = _SAMPLING_STRATEGIES[strategy]
        except KeyError as exc:
            raise ValueError(f"unknown sampling strategy: {strategy!r}") from exc
        return strategy, implementation
    name = getattr(strategy, "name", getattr(strategy, "__name__", "custom"))
    if not callable(strategy) and not hasattr(strategy, "sample"):
        raise TypeError("sampling strategy must be a name, callable, or sample() object")
    return str(name), strategy


def _strategy_sample(
    strategy: Any,
    query_ids: Sequence[str],
    count: int,
    *,
    seed: int,
    task_id: str,
) -> list[str]:
    if hasattr(strategy, "sample"):
        selected = strategy.sample(query_ids, count, seed=seed, task_id=task_id)
    else:
        selected = strategy(query_ids, count, seed=seed, task_id=task_id)
    result = list(selected)
    if len(result) != count or len(set(result)) != len(result):
        raise ValueError("sampling strategy returned the wrong number of unique query IDs")
    if not set(result).issubset(set(query_ids)):
        raise ValueError("sampling strategy returned an unknown query ID")
    return result


def _group_value(
    row: Mapping[str, Any],
    *,
    group_key: str | Callable[[Mapping[str, Any]], str] | None,
) -> str:
    if callable(group_key):
        value = group_key(row)
    elif isinstance(group_key, str):
        value = row.get(group_key)
    elif group_key is None:
        value = row.get("group_id", row.get("leakage_group", _query_id(row)))
    else:
        raise TypeError("group_key must be a field name, callable, or None")
    if not isinstance(value, str) or not value:
        raise ValueError(f"query group must be a non-empty string: {value!r}")
    return value


def _choose_group_subset(
    groups: Sequence[tuple[str, list[str]]], target: int, *, seed: int, scope: str
) -> list[str]:
    """Choose whole groups whose query count is exactly ``target``.

    A bounded dynamic program keeps group boundaries intact while randomizing
    the order in which equally valid solutions are considered.
    """

    _validate_count(target, "target")
    if target == 0:
        return []
    if target > sum(len(ids) for _, ids in groups):
        raise SamplingShortageError(f"cannot fill {target} IDs for {scope!r}")
    order = list(groups)
    _rng(seed, scope).shuffle(order)
    choices: dict[int, list[int]] = {0: []}
    for index, (_, ids) in enumerate(order):
        size = len(ids)
        for total in sorted(list(choices), reverse=True):
            new_total = total + size
            if new_total > target or new_total in choices:
                continue
            choices[new_total] = choices[total] + [index]
        if target in choices:
            break
    if target not in choices:
        raise SamplingShortageError(
            f"group boundaries cannot form exactly {target} IDs for {scope!r}"
        )
    selected: list[str] = []
    for index in choices[target]:
        selected.extend(order[index][1])
    return sorted(selected)


def sample_query_splits(
    records: Iterable[Mapping[str, Any] | str],
    *,
    dev_size: int = DEV_SIZE,
    test_size: int = TEST_SIZE,
    smoke_size: int = SMOKE_SIZE,
    seed: int = DEFAULT_SEED,
    strategy: str | Any = "random_without_replacement",
    task_id: str = "task",
    group_key: str | Callable[[Mapping[str, Any]], str] | None = None,
) -> dict[str, list[str]]:
    """Sample disjoint dev/test IDs and a smoke subset of dev.

    Records are canonically sorted before sampling, so reordering input rows
    cannot change the result.  When ``group_key`` is supplied, whole groups
    are assigned to one split and no group can leak between dev and test.
    """

    _validate_seed(seed)
    for value, name in (
        (dev_size, "dev_size"),
        (test_size, "test_size"),
        (smoke_size, "smoke_size"),
    ):
        _validate_count(value, name)
    if smoke_size > dev_size:
        raise ValueError("smoke_size cannot exceed dev_size")
    rows = _records(records)
    if dev_size + test_size > len(rows):
        raise SamplingShortageError(
            f"{task_id!r} requires {dev_size + test_size} unique queries, only {len(rows)} available"
        )
    strategy_name, implementation = _resolve_strategy(strategy)
    if group_key is None:
        candidate_ids = [_query_id(row) for row in rows]
        selected = _strategy_sample(
            implementation,
            candidate_ids,
            dev_size + test_size,
            seed=seed,
            task_id=task_id,
        )
        dev = sorted(selected[:dev_size])
        test = sorted(selected[dev_size:])
    else:
        by_group: dict[str, list[str]] = {}
        for row in rows:
            by_group.setdefault(_group_value(row, group_key=group_key), []).append(_query_id(row))
        groups = [(name, sorted(ids)) for name, ids in sorted(by_group.items())]
        selected = set(
            _choose_group_subset(
                groups, dev_size + test_size, seed=seed, scope=f"{task_id}:combined"
            )
        )
        selected_groups = [
            (name, ids) for name, ids in groups if set(ids).issubset(selected)
        ]
        dev = _choose_group_subset(
            selected_groups, dev_size, seed=seed, scope=f"{task_id}:dev"
        )
        dev_set = set(dev)
        test = sorted(selected - dev_set)
        if len(test) != test_size:
            raise SamplingShortageError(
                f"group boundaries cannot form dev={dev_size}, test={test_size} for {task_id!r}"
            )
    if smoke_size:
        smoke = _strategy_sample(
            implementation,
            dev,
            smoke_size,
            seed=seed,
            task_id=f"{task_id}:smoke",
        )
        smoke = sorted(smoke)
    else:
        smoke = []
    result = {"dev": dev, "test": test, "smoke": smoke}
    assert_no_group_leakage(result, rows, group_key=group_key)
    return result


def sample_query_ids(
    records: Iterable[Mapping[str, Any] | str],
    *,
    count: int,
    seed: int = DEFAULT_SEED,
    strategy: str | Any = "random_without_replacement",
    task_id: str = "task",
) -> list[str]:
    """Convenience wrapper for one deterministic sample."""

    rows = _records(records)
    _, implementation = _resolve_strategy(strategy)
    return sorted(
        _strategy_sample(
            implementation,
            [_query_id(row) for row in rows],
            count,
            seed=seed,
            task_id=task_id,
        )
    )


def assert_no_group_leakage(
    splits: Mapping[str, Sequence[str]],
    records: Iterable[Mapping[str, Any] | str],
    *,
    group_key: str | Callable[[Mapping[str, Any]], str] | None = None,
) -> dict[str, list[str]]:
    """Return and reject groups present in both dev and test."""

    rows = {_query_id(row): row for row in _records(records)}
    memberships: dict[str, set[str]] = {}
    for split in ("dev", "test"):
        for query_id in splits.get(split, []):
            if query_id not in rows:
                raise ValueError(f"split references unknown query ID: {query_id!r}")
            group = _group_value(rows[query_id], group_key=group_key)
            memberships.setdefault(group, set()).add(split)
    leakage = {
        group: sorted(parts)
        for group, parts in memberships.items()
        if len(parts) > 1
    }
    if leakage:
        raise ValueError(f"query group leakage between dev/test: {sorted(leakage)}")
    return leakage


def assert_global_group_leakage(
    task_splits: Mapping[str, Mapping[str, Sequence[str]]],
    records_by_task: Mapping[str, Iterable[Mapping[str, Any] | str]],
    *,
    group_key: str | Callable[[Mapping[str, Any]], str] | None = None,
) -> dict[str, list[str]]:
    """Check dev/test leakage across all configured tasks.

    Source-family names are deliberately not used as the default group key:
    one source may legitimately supply all 60 query samples.  Callers that
    have documented equivalent or parallel query groups should pass that
    field (for example ``parallel_group``) explicitly.
    """

    memberships: dict[str, set[str]] = {}
    for task_id, splits in task_splits.items():
        rows = {_query_id(row): row for row in _records(records_by_task.get(task_id, []))}
        for split in ("dev", "test"):
            for query_id in splits.get(split, []):
                if query_id not in rows:
                    raise ValueError(f"{task_id} split references unknown query ID: {query_id!r}")
                group = _group_value(rows[query_id], group_key=group_key)
                memberships.setdefault(group, set()).add(f"{split}:{task_id}")
    leakage = {
        group: sorted(parts)
        for group, parts in memberships.items()
        if any(part.startswith("dev:") for part in parts)
        and any(part.startswith("test:") for part in parts)
    }
    if leakage:
        raise ValueError(f"global query group leakage between dev/test: {sorted(leakage)}")
    return leakage


def _normalise_query_text(value: Any) -> str | None:
    """Return a stable, deliberately conservative query-text normalisation."""

    if not isinstance(value, str):
        return None
    normalised = unicodedata.normalize("NFKC", value).casefold()
    normalised = re.sub(r"\s+", " ", normalised).strip()
    return normalised or None


def _query_text(row: Mapping[str, Any]) -> str | None:
    for field in ("text", "query", "question", "query_text", "prompt"):
        value = _normalise_query_text(row.get(field))
        if value is not None:
            return value
    return None


def _query_label_signature(labels: Mapping[str, Any], *, query_id: str) -> tuple[tuple[str, float], ...]:
    """Canonicalise qrels for exact duplicate-query comparison."""

    if not isinstance(labels, Mapping):
        raise SourceBlockedError(f"query {query_id!r} qrels are not an object")
    signature: list[tuple[str, float]] = []
    for document_id, value in labels.items():
        if not isinstance(document_id, str) or not document_id:
            raise SourceBlockedError(f"query {query_id!r} qrels contain an invalid document ID")
        if isinstance(value, bool):
            raise SourceBlockedError(f"query {query_id!r} qrels contain a boolean score")
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise SourceBlockedError(
                f"query {query_id!r} qrels contain a non-numeric score"
            ) from exc
        if not math.isfinite(score):
            raise SourceBlockedError(f"query {query_id!r} qrels contain a non-finite score")
        signature.append((document_id, score))
    return tuple(sorted(signature))


def _prepare_query_eligibility(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter and deduplicate source queries before global sampling.

    Query text is the source-local deduplication key.  Exact duplicate labels
    keep the lexicographically smallest query ID; conflicting labels exclude
    every member of the duplicate group so no label set is silently chosen.
    Overflow is counted per source row and is applied before a sampler sees
    any query ID.
    """

    canonical_rows = _records(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for row in canonical_rows:
        query_id = _query_id(row)
        if query_id not in qrels:
            raise SourceBlockedError(f"query {query_id!r} has no qrels")
        labels = qrels[query_id]
        signature = _query_label_signature(labels, query_id=query_id)
        positive_count = sum(score > 0 for _, score in signature)
        if positive_count > LITE_CAPACITY:
            overflow.append(
                {
                    "query_id": query_id,
                    "tier": "both" if positive_count > FULL_CAPACITY else "lite",
                    "reason": "positive_overflow",
                    "positive_count": positive_count,
                    "overflow": positive_count - LITE_CAPACITY,
                    "excluded_before_sampling": True,
                }
            )
        text = _query_text(row)
        if text is None:
            exclusions.append(
                {
                    "query_id": query_id,
                    "reason": "missing_query_text",
                    "excluded_before_sampling": True,
                }
            )
            continue
        grouped.setdefault(text, []).append(
            {
                "row": dict(row),
                "signature": signature,
                "positive_count": positive_count,
            }
        )

    eligible: list[dict[str, Any]] = []
    for text, members in sorted(grouped.items()):
        members.sort(key=lambda item: _query_id(item["row"]))
        signatures = {member["signature"] for member in members}
        if len(signatures) > 1:
            group_id = f"query-text-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"
            for member in members:
                exclusions.append(
                    {
                        "query_id": _query_id(member["row"]),
                        "reason": "duplicate_query_conflicting_qrels",
                        "dedup_group": group_id,
                        "excluded_before_sampling": True,
                    }
                )
            continue
        representative = members[0]
        if representative["positive_count"] <= 0:
            for member in members:
                exclusions.append(
                    {
                        "query_id": _query_id(member["row"]),
                        "reason": "no_positive_labels",
                        "excluded_before_sampling": True,
                    }
                )
            continue
        if representative["positive_count"] > LITE_CAPACITY:
            # The overflow records above retain the per-row count and are the
            # canonical exclusion audit; no member enters the sampler.
            continue
        eligible.append(representative["row"])
        group_id = (
            f"query-text-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"
        )
        for member in members[1:]:
            exclusions.append(
                {
                    "query_id": _query_id(member["row"]),
                    "reason": "duplicate_query_text",
                    "canonical_query_id": _query_id(representative["row"]),
                    "dedup_group": group_id,
                    "excluded_before_sampling": True,
                }
            )
    return sorted(eligible, key=_query_id), exclusions + overflow, overflow


def _equivalence_tokens(
    task_id: str,
    row: Mapping[str, Any],
    source_context: Mapping[str, Any] | None,
) -> list[str]:
    """Build query-level equivalence tokens without grouping whole sources."""

    tokens: list[str] = []
    text = _query_text(row)
    if text is not None:
        tokens.append(f"text:{text}")
    for field in (
        "group_id",
        "leakage_group",
        "parallel_group",
        "translation_group",
        "equivalent_group",
        "query_group",
        "source_query_id",
        "original_query_id",
        "equivalent_query_id",
        "parallel_query_id",
    ):
        value = _normalise_query_text(row.get(field))
        if value is not None:
            tokens.append(f"{field}:{value}")
    context = source_context or {}
    source_family = _normalise_query_text(context.get("source_family"))
    source_id = _normalise_query_text(context.get("source_id"))
    query_id = _query_id(row)
    # Source identity is scoped to the query ID.  It therefore catches the
    # same source query reused by two task views without making every query in
    # a source one giant group.
    if source_family is not None:
        tokens.append(f"source-family:{source_family}:query:{query_id}")
    if source_id is not None:
        tokens.append(f"source-id:{source_id}:query:{query_id}")
    if not tokens:
        tokens.append(f"row:{task_id}:{query_id}")
    return tokens


class _DisjointSet:
    def __init__(self, keys: Iterable[tuple[str, str]]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: tuple[str, str]) -> tuple[str, str]:
        parent = self.parent[key]
        if parent != key:
            parent = self.find(parent)
            self.parent[key] = parent
        return parent

    def union(self, left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _global_query_groups(
    records_by_task: Mapping[str, Iterable[Mapping[str, Any] | str]],
    source_context_by_task: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """Return deterministic query-level components and their public IDs."""

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    tokens: dict[str, tuple[str, str]] = {}
    for task_id in sorted(records_by_task):
        for row in _records(records_by_task[task_id]):
            query_id = _query_id(row)
            key = (task_id, query_id)
            rows_by_key[key] = row
    disjoint = _DisjointSet(rows_by_key)
    for key, row in rows_by_key.items():
        task_id, _ = key
        context = (source_context_by_task or {}).get(task_id)
        for token in _equivalence_tokens(task_id, row, context):
            previous = tokens.get(token)
            if previous is not None:
                disjoint.union(previous, key)
            else:
                tokens[token] = key

    components: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in sorted(rows_by_key):
        components.setdefault(disjoint.find(key), []).append(key)
    ordered_components = sorted(components.values(), key=lambda members: tuple(members))
    groups: dict[str, list[dict[str, Any]]] = {}
    group_map: dict[str, dict[str, str]] = {task_id: {} for task_id in records_by_task}
    for members in ordered_components:
        digest = hashlib.sha256(
            json.dumps(members, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        group_id = f"query-group-{digest}"
        group_rows = []
        for task_id, query_id in members:
            row = dict(rows_by_key[(task_id, query_id)])
            row["__task_id"] = task_id
            group_rows.append(row)
            group_map.setdefault(task_id, {})[query_id] = group_id
        groups[group_id] = group_rows
    return groups, group_map


def _assign_global_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    task_ids: Sequence[str],
    *,
    target: int,
    seed: int,
    split: str,
    assigned: dict[str, str],
    counts: dict[str, int],
    attempt: int,
    strategy_ranks: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    """Greedily assign whole equivalence groups to one split.

    The task-local quotas are small and source query equivalence components
    are normally singletons.  The bounded retry gives deterministic freedom
    to avoid a poor early choice when a component spans task views.
    """

    order = list(groups)
    _rng(seed, f"global:{split}:{attempt}").shuffle(order)
    priority = {group_id: index for index, group_id in enumerate(order)}
    group_counts: dict[str, dict[str, int]] = {}
    groups_by_task: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
    for group_id, rows in groups.items():
        per_task: dict[str, int] = {}
        for row in rows:
            task_id = row["__task_id"]
            per_task[task_id] = per_task.get(task_id, 0) + 1
        group_counts[group_id] = per_task
        for task_id in per_task:
            groups_by_task.setdefault(task_id, []).append(group_id)

    while True:
        remaining = {task_id: target - counts[task_id] for task_id in task_ids}
        if not any(remaining.values()):
            return
        task_id = min(
            (name for name in task_ids if remaining[name]),
            key=lambda name: (
                sum(
                    1
                    for group_id in groups_by_task[name]
                    if group_id not in assigned
                    and all(
                        counts[other] + amount <= target
                        for other, amount in group_counts[group_id].items()
                    )
                ),
                -remaining[name],
                name,
            ),
        )
        candidates = []
        for group_id in groups_by_task[task_id]:
            per_task = group_counts[group_id]
            if group_id in assigned:
                continue
            if any(counts[other] + amount > target for other, amount in per_task.items()):
                continue
            candidates.append(group_id)
        if not candidates:
            raise SamplingShortageError(
                f"global {split} group boundaries cannot fill {target} IDs for {task_id!r}"
            )
        task_ranks = (strategy_ranks or {}).get(task_id, {})
        candidates.sort(
            key=lambda group_id: (
                group_counts[group_id][task_id],
                min(
                    task_ranks.get(_query_id(row), len(task_ranks))
                    for row in groups[group_id]
                    if row["__task_id"] == task_id
                ),
                sum(group_counts[group_id].values()),
                priority[group_id],
            )
        )
        selected = candidates[0]
        assigned[selected] = split
        for name, amount in group_counts[selected].items():
            counts[name] += amount


def sample_global_query_splits(
    records_by_task: Mapping[str, Iterable[Mapping[str, Any] | str]],
    *,
    source_context_by_task: Mapping[str, Mapping[str, Any]] | None = None,
    dev_size: int = DEV_SIZE,
    test_size: int = TEST_SIZE,
    smoke_size: int = SMOKE_SIZE,
    seed: int = DEFAULT_SEED,
    strategy: str | Any = "random_without_replacement",
) -> dict[str, dict[str, list[str]]]:
    """Sample all task splits while keeping equivalent queries together."""

    _validate_seed(seed)
    for value, name in (
        (dev_size, "dev_size"),
        (test_size, "test_size"),
        (smoke_size, "smoke_size"),
    ):
        _validate_count(value, name)
    if smoke_size > dev_size:
        raise ValueError("smoke_size cannot exceed dev_size")
    canonical_records = {
        task_id: _records(rows) for task_id, rows in records_by_task.items()
    }
    task_ids = sorted(canonical_records)
    if not task_ids:
        return {}
    _, implementation = _resolve_strategy(strategy)
    groups, group_map = _global_query_groups(canonical_records, source_context_by_task)
    eligible_ids_by_task = {
        task_id: [_query_id(row) for row in canonical_records[task_id]]
        for task_id in task_ids
    }
    for task_id in task_ids:
        available = len(eligible_ids_by_task[task_id])
        if available < dev_size + test_size:
            raise SamplingShortageError(
                f"{task_id!r} requires {dev_size + test_size} unique eligible queries, "
                f"only {available} available"
            )
    last_error: Exception | None = None
    strategy_ranks = {
        split: {
            task_id: {
                query_id: index
                for index, query_id in enumerate(
                    _strategy_sample(
                        implementation,
                        eligible_ids_by_task[task_id],
                        len(eligible_ids_by_task[task_id]),
                        seed=seed,
                        task_id=f"{task_id}:{split}",
                    )
                )
            }
            for task_id in task_ids
        }
        for split in ("dev", "test")
    }
    for attempt in range(64):
        # Let the configured sampler determine the per-task order used by the
        # group-constrained assignment.  Calling it with the complete eligible
        # set makes a sampler replaceable without changing this executor's
        # quota or leakage checks; the assignment still keeps whole global
        # equivalence groups together.
        assigned: dict[str, str] = {}
        dev_counts = {task_id: 0 for task_id in task_ids}
        test_counts = {task_id: 0 for task_id in task_ids}
        try:
            _assign_global_groups(
                groups,
                task_ids,
                target=dev_size,
                seed=seed,
                split="dev",
                assigned=assigned,
                counts=dev_counts,
                attempt=attempt,
                strategy_ranks=strategy_ranks["dev"],
            )
            _assign_global_groups(
                groups,
                task_ids,
                target=test_size,
                seed=seed,
                split="test",
                assigned=assigned,
                counts=test_counts,
                attempt=attempt,
                strategy_ranks=strategy_ranks["test"],
            )
        except SamplingShortageError as exc:
            last_error = exc
            continue
        result: dict[str, dict[str, list[str]]] = {}
        for task_id in task_ids:
            dev = sorted(
                _query_id(row)
                for group_id, rows in groups.items()
                if assigned.get(group_id) == "dev"
                for row in rows
                if row["__task_id"] == task_id
            )
            test = sorted(
                _query_id(row)
                for group_id, rows in groups.items()
                if assigned.get(group_id) == "test"
                for row in rows
                if row["__task_id"] == task_id
            )
            smoke = sorted(
                _strategy_sample(
                    implementation,
                    dev,
                    smoke_size,
                    seed=seed,
                    task_id=f"{task_id}:smoke",
                )
            ) if smoke_size else []
            result[task_id] = {"dev": dev, "test": test, "smoke": smoke}
        # The assignment itself is component based; this assertion catches
        # accidental future changes to the extraction above.
        for task_id, splits in result.items():
            assert_no_group_leakage(
                splits,
                canonical_records[task_id],
                group_key=lambda row, task_id=task_id: group_map[task_id][_query_id(row)],
            )
        memberships: dict[str, set[str]] = {}
        for task_id, splits in result.items():
            for split in ("dev", "test"):
                for query_id in splits[split]:
                    memberships.setdefault(group_map[task_id][query_id], set()).add(split)
        if any(len(parts) > 1 for parts in memberships.values()):
            raise AssertionError("global query group assignment leaked across splits")
        return result
    raise SamplingShortageError(str(last_error or "global group boundaries cannot satisfy quotas"))


def _normalise_ranking(
    ranked_ids: Sequence[Any] | Mapping[str, Any],
    scores: Mapping[str, float] | Sequence[float] | None,
) -> tuple[list[str], dict[str, float] | None]:
    ranking: list[str] = []
    score_map: dict[str, float] | None = None
    if isinstance(ranked_ids, Mapping):
        score_map = {}
        for doc_id, score in ranked_ids.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("ranking IDs must be non-empty strings")
            try:
                numeric_score = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ranking score is not numeric for {doc_id!r}") from exc
            if not math.isfinite(numeric_score):
                raise ValueError(f"ranking score is not finite for {doc_id!r}")
            score_map[doc_id] = numeric_score
        ranking = sorted(score_map, key=lambda doc_id: (-score_map[doc_id], doc_id))
    else:
        for item in ranked_ids:
            if isinstance(item, Mapping):
                doc_id = item.get("doc_id", item.get("id", item.get("_id")))
                item_score = item.get("score")
                if item_score is not None:
                    if score_map is None:
                        score_map = {}
                    numeric_score = float(item_score)
                    if not math.isfinite(numeric_score):
                        raise ValueError(f"ranking score is not finite for {doc_id!r}")
                    score_map[doc_id] = numeric_score
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                doc_id, item_score = item
                if score_map is None:
                    score_map = {}
                numeric_score = float(item_score)
                if not math.isfinite(numeric_score):
                    raise ValueError(f"ranking score is not finite for {doc_id!r}")
                score_map[doc_id] = numeric_score
            else:
                doc_id = item
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("ranking IDs must be non-empty strings")
            ranking.append(doc_id)
        if score_map is None and scores is not None:
            score_map = {}
            if isinstance(scores, Mapping):
                for doc_id in ranking:
                    if doc_id not in scores:
                        raise ValueError(f"missing ranking score for {doc_id!r}")
                    numeric_score = float(scores[doc_id])
                    if not math.isfinite(numeric_score):
                        raise ValueError(f"ranking score is not finite for {doc_id!r}")
                    score_map[doc_id] = numeric_score
            else:
                if len(scores) != len(ranking):
                    raise ValueError("ranking scores must align with ranked IDs")
                score_map = {}
                for doc_id, value in zip(ranking, scores):
                    numeric_score = float(value)
                    if not math.isfinite(numeric_score):
                        raise ValueError(f"ranking score is not finite for {doc_id!r}")
                    score_map[doc_id] = numeric_score
    if len(ranking) != len(set(ranking)):
        raise ValueError("ranking contains duplicate document IDs")
    if score_map is not None:
        if set(score_map) != set(ranking):
            raise ValueError("ranking score IDs differ from ranked IDs")
        if any(not isinstance(value, (int, float)) for value in score_map.values()):
            raise ValueError("ranking scores must be numeric")
    return ranking, score_map


def _normalise_qrels(qrels: Mapping[str, Any] | Sequence[str]) -> dict[str, float]:
    if isinstance(qrels, Mapping):
        result: dict[str, float] = {}
        for doc_id, value in qrels.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("qrels IDs must be non-empty strings")
            if isinstance(value, bool):
                raise ValueError("qrels values must be numeric")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"qrels score is not numeric for {doc_id!r}") from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"qrels score is not finite for {doc_id!r}")
            result[doc_id] = numeric_value
        return result
    result = {}
    for doc_id in qrels:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("qrels IDs must be non-empty strings")
        result[doc_id] = 1.0
    return result


def select_nested_pools(
    ranked_ids: Sequence[Any] | Mapping[str, Any],
    qrels: Mapping[str, Any] | Sequence[str],
    *,
    scores: Mapping[str, float] | Sequence[float] | None = None,
    lite_capacity: int = LITE_CAPACITY,
    full_capacity: int = FULL_CAPACITY,
) -> dict[str, Any]:
    """Derive nested lite/full candidate IDs while retaining positives.

    ``ranked_ids`` is the ordered compact ranking from the existing full pool.
    The full set is selected first; the lite set is then selected from that
    set, guaranteeing ``lite ⊂ full`` whenever lite is eligible.  A query with
    more positives than the lite capacity is explicitly excluded from lite and
    counted in the audit instead of silently dropping labels.
    """

    _validate_count(lite_capacity, "lite_capacity")
    _validate_count(full_capacity, "full_capacity")
    if lite_capacity <= 0 or full_capacity <= 0 or lite_capacity > full_capacity:
        raise ValueError("capacities must be positive and lite_capacity <= full_capacity")
    ranking, score_map = _normalise_ranking(ranked_ids, scores)
    labels = _normalise_qrels(qrels)
    positives = {doc_id for doc_id, value in labels.items() if value > 0}
    unknown = sorted(positives - set(ranking))
    if unknown:
        raise ValueError(f"positive qrels are absent from the saved ranking: {unknown[:5]}")
    rank_index = {doc_id: index for index, doc_id in enumerate(ranking)}

    def choose(capacity: int) -> list[str] | None:
        if len(positives) > capacity:
            return None
        selected = [doc_id for doc_id in ranking if doc_id in positives]
        selected.extend(
            doc_id
            for doc_id in ranking
            if doc_id not in positives and len(selected) < capacity
        )
        return sorted(selected, key=rank_index.__getitem__)

    full = choose(full_capacity)
    lite = None if full is None else choose_from_full(full, positives, lite_capacity, rank_index)
    lite_overflow = max(0, len(positives) - lite_capacity)
    full_overflow = max(0, len(positives) - full_capacity)
    audit = {
        "schema_version": POOL_DERIVATION_VERSION,
        "ranking_size": len(ranking),
        "positive_count": len(positives),
        "positive_overflow": lite_overflow,
        "positive_overflow_lite": lite_overflow,
        "positive_overflow_full": full_overflow,
        "lite_excluded": lite is None,
        "full_excluded": full is None,
        "lite_exclusion_reason": (
            "positive_overflow" if lite is None and full is not None else
            "full_positive_overflow" if lite is None else None
        ),
        "full_exclusion_reason": "positive_overflow" if full is None else None,
        "selected_counts": {
            "lite": len(lite) if lite is not None else 0,
            "full": len(full) if full is not None else 0,
        },
        "nested": bool(lite is not None and full is not None and set(lite).issubset(full)),
        "capacities": {"lite": lite_capacity, "full": full_capacity},
    }
    return {
        "lite": lite,
        "full": full,
        "lite_scores": [score_map[doc_id] for doc_id in lite] if lite is not None and score_map else None,
        "full_scores": [score_map[doc_id] for doc_id in full] if full is not None and score_map else None,
        "audit": audit,
    }


def choose_from_full(
    full: Sequence[str],
    positives: set[str],
    capacity: int,
    rank_index: Mapping[str, int],
) -> list[str] | None:
    if len(positives) > capacity:
        return None
    selected = [doc_id for doc_id in full if doc_id in positives]
    selected.extend(
        doc_id for doc_id in full if doc_id not in positives and len(selected) < capacity
    )
    return sorted(selected, key=rank_index.__getitem__)


def build_sampling_manifest(
    tasks: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    strategy: str = "random_without_replacement",
    config_identity: Mapping[str, Any] | None = None,
    input_identity: Any | None = None,
) -> dict[str, Any]:
    """Build the immutable suite-level sampling manifest."""

    _validate_seed(seed)
    canonical_tasks = {name: copy.deepcopy(dict(value)) for name, value in sorted(tasks.items())}
    for task_id, task in canonical_tasks.items():
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("sampling manifest task IDs must be non-empty strings")
        splits = task.get("splits")
        if splits is not None:
            if not isinstance(splits, Mapping):
                raise ValueError(f"task {task_id!r} splits must be an object")
            dev = list(splits.get("dev", []))
            test = list(splits.get("test", []))
            smoke = list(splits.get("smoke", []))
            if len(dev) != len(set(dev)) or len(test) != len(set(test)):
                raise ValueError(f"task {task_id!r} split IDs overlap")
            if set(dev) & set(test):
                raise ValueError(f"task {task_id!r} dev/test IDs overlap")
            if len(smoke) != len(set(smoke)) or not set(smoke) <= set(dev):
                raise ValueError(f"task {task_id!r} smoke IDs are not a dev subset")
    canonical_input = input_identity if input_identity is not None else canonical_tasks
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {"domains": list(DOMAINS), "languages": list(LANGUAGES), "task_count": len(TASK_IDS)},
        "sampling": {
            "algorithm_version": SAMPLING_ALGORITHM_VERSION,
            "strategy": strategy,
            "seed": seed,
            "sizes": {"dev": DEV_SIZE, "test": TEST_SIZE, "smoke": SMOKE_SIZE},
            "input_digest": stable_json_sha256(canonical_input),
            "selection_policy": "global_equivalence_group_constrained",
        },
        "config_identity": dict(config_identity or {}),
        "tasks": canonical_tasks,
    }


def _json_load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc


def load_benchmark_config(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the v2 scope/roster configuration."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "benchmark_v2.json"
    config = _json_load(Path(path))
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark_v2 schema_version")
    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("benchmark_v2 scope must be an object")
    if tuple(scope.get("domains", ())) != DOMAINS or tuple(scope.get("languages", ())) != LANGUAGES:
        raise ValueError("benchmark_v2 scope domains/languages are invalid")
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("seed") != DEFAULT_SEED:
        raise ValueError("benchmark_v2 sampling must declare seed 42")
    if sampling.get("algorithm_version") != SAMPLING_ALGORITHM_VERSION:
        raise ValueError("benchmark_v2 sampling algorithm version is invalid")
    sizes = sampling.get("sizes")
    if sizes != {"dev": DEV_SIZE, "test": TEST_SIZE, "smoke": SMOKE_SIZE}:
        raise ValueError("benchmark_v2 sampling sizes are invalid")
    sources = config.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(TASK_IDS):
        raise ValueError("benchmark_v2 source roster must contain exactly 20 tasks")
    for task_id, entry in sources.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"source roster entry {task_id!r} must be an object")
        status = entry.get("status")
        if status not in {"eligible", "blocked"}:
            raise ValueError(f"source roster entry {task_id!r} has invalid status")
        candidates = entry.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"source roster entry {task_id!r} has no candidates")
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("source_id"), str):
                raise ValueError(f"source roster candidate for {task_id!r} is invalid")
            provenance = candidate.get("provenance")
            if provenance not in {"native", "translated", "unknown", "missing"}:
                raise ValueError(f"source roster provenance for {task_id!r} is invalid")
            if provenance == "unknown" and candidate.get("provenance_override"):
                override = candidate["provenance_override"]
                if not isinstance(override, Mapping) or not override.get("evidence"):
                    raise ValueError(
                        f"source roster provenance override for {task_id!r} needs evidence"
                    )
                if override.get("value") not in {"native", "translated"}:
                    raise ValueError(
                        f"source roster provenance override for {task_id!r} must be native or translated"
                    )
    return config


def _catalog_by_source(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = catalog.get("sources")
    if not isinstance(rows, list):
        raise ValueError("catalog sources must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("source"), str):
            raise ValueError("catalog contains an invalid source row")
        if row["source"] in result:
            raise ValueError(f"catalog contains duplicate source: {row['source']!r}")
        result[row["source"]] = row
    return result


def _pool_path(source: Mapping[str, Any], capacity: int) -> Path | None:
    pools = source.get("pools")
    if not isinstance(pools, list):
        return None
    for pool in pools:
        if isinstance(pool, Mapping) and pool.get("capacity") == capacity and isinstance(pool.get("path"), str):
            return Path(pool["path"])
    return None


def assess_source_eligibility(
    config: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Assess all 20 configured tasks without changing the catalog."""

    sources = _catalog_by_source(catalog)
    roster = config["sources"]
    assessment: dict[str, dict[str, Any]] = {}
    for task_id in TASK_IDS:
        entry = roster[task_id]
        selected: Mapping[str, Any] | None = None
        reasons: list[str] = []
        for candidate in entry["candidates"]:
            source_id = candidate["source_id"]
            source = sources.get(source_id)
            if source is None:
                continue
            if candidate.get("provenance") in {"missing", "unknown"} and not candidate.get(
                "provenance_override"
            ):
                continue
            effective_candidate = dict(candidate)
            override = candidate.get("provenance_override")
            if isinstance(override, Mapping):
                effective_candidate["provenance"] = override.get("value")
            selected = {"config": effective_candidate, "catalog": source}
            break
        if selected is None:
            configured_reason = entry.get("blocked_reason", "no source with evidenced provenance")
            assessment[task_id] = {
                "task": task_id,
                "status": "blocked",
                "reason": configured_reason,
                "candidates": [candidate.get("source_id") for candidate in entry["candidates"]],
            }
            continue
        candidate = selected["config"]
        source = selected["catalog"]
        documents = source.get("documents")
        queries = source.get("queries")
        if not isinstance(documents, int) or documents < FULL_CAPACITY:
            reasons.append("insufficient_documents_for_full_1000")
        if not isinstance(queries, int) or queries < DEV_SIZE + TEST_SIZE:
            reasons.append("insufficient_queries_for_dev10_test50")
        if _pool_path(source, FULL_CAPACITY) is None:
            reasons.append("missing_full_1000_pool")
        if _pool_path(source, LITE_CAPACITY) is None:
            reasons.append("missing_lite_200_pool")
        status = "eligible" if not reasons and entry.get("status") == "eligible" else "blocked"
        if entry.get("status") == "blocked":
            reasons.insert(0, entry.get("blocked_reason", "blocked by configured data gap"))
            status = "blocked"
        assessment[task_id] = {
            "task": task_id,
            "status": status,
            "reason": "; ".join(reasons) if reasons else None,
            "source_id": candidate["source_id"],
            "source": dict(source),
            "candidate": dict(candidate),
        }
    return assessment


def _metadata_dir(pool_root: Path, task: str) -> Path:
    direct = pool_root / "metadata" / task
    if direct.is_dir():
        return direct
    candidates = [path for path in (pool_root / "metadata").glob("*/") if path.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"cannot locate pool metadata for {task!r} under {pool_root}")


def _decode_candidate_payload(
    payload: Mapping[str, Any], source_doc_ids: Mapping[str, str]
) -> dict[str, Any]:
    """Expand the compact source-pool audit fields without touching bodies."""

    result = copy.deepcopy(dict(payload))
    encoding = result.pop("storage_encoding", None)
    if encoding is None:
        return result
    if encoding != "short-id-derived-audit-v1":
        raise ValueError(f"unsupported candidate pool storage encoding: {encoding!r}")
    pools = result.get("pools")
    if not isinstance(pools, Mapping):
        raise ValueError("compact candidate pool has no pools object")
    for query_id, raw_record in pools.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"candidate pool record is invalid: {query_id!r}")
        record = dict(raw_record)
        aliases = record.get("candidate_doc_ids")
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or alias not in source_doc_ids for alias in aliases
        ):
            raise ValueError(f"compact candidate pool IDs are invalid: {query_id!r}")
        originals = [source_doc_ids[alias] for alias in aliases]
        record["candidate_original_doc_ids"] = originals
        audit = dict(record.get("audit", {}))
        if "selected_ids" in audit or "selected_scores" in audit:
            raise ValueError(f"compact candidate pool has conflicting audit fields: {query_id!r}")
        audit["selected_ids"] = list(originals)
        audit["selected_scores"] = record.get("candidate_scores")
        for key in ("initial_top_ids", "injected_positive_ids", "removed_nonpositive_ids"):
            aliases_in_audit = audit.get(key, [])
            if not isinstance(aliases_in_audit, list) or any(
                alias not in source_doc_ids for alias in aliases_in_audit
            ):
                raise ValueError(f"compact candidate pool audit IDs are invalid: {query_id!r}")
            audit[key] = [source_doc_ids[alias] for alias in aliases_in_audit]
        record["audit"] = audit
        pools[query_id] = record
    return result


class _JsonStream:
    """Small forward-only JSON reader used to avoid materialising rankings."""

    def __init__(self, path: Path, *, chunk_size: int = 1024 * 1024) -> None:
        self.handle = Path(path).open("r", encoding="utf-8")
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def close(self) -> None:
        self.handle.close()

    def _fill(self) -> None:
        if self.eof:
            return
        chunk = self.handle.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def _compact(self) -> None:
        if self.position > self.chunk_size:
            self.buffer = self.buffer[self.position :]
            self.position = 0

    def skip_whitespace(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self._fill()

    def character(self) -> str:
        self.skip_whitespace()
        while self.position >= len(self.buffer) and not self.eof:
            self._fill()
            self.skip_whitespace()
        if self.position >= len(self.buffer):
            raise ValueError("unexpected end of JSON stream")
        return self.buffer[self.position]

    def consume(self, expected: str) -> None:
        self.skip_whitespace()
        if self.character() != expected:
            raise ValueError(f"expected {expected!r} in JSON stream")
        self.position += 1

    def value(self) -> Any:
        while True:
            self.skip_whitespace()
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError("invalid or truncated JSON stream")
                self._fill()
                continue
            self.position = end
            self._compact()
            return value


def _load_selected_candidate_payload(
    pool_root: Path,
    task: str,
    query_ids: Sequence[str],
    source_doc_ids: Mapping[str, str],
) -> dict[str, Any]:
    """Read only selected records from a compact candidate-pool JSON object."""

    metadata = _metadata_dir(pool_root, task)
    wanted = set(query_ids)
    payload: dict[str, Any] = {}
    stream = _JsonStream(metadata / "candidate_pools.json")
    try:
        stream.consume("{")
        if stream.character() == "}":
            stream.position += 1
        else:
            while True:
                key = stream.value()
                if not isinstance(key, str):
                    raise ValueError("candidate pool top-level key is not a string")
                stream.consume(":")
                if key != "pools":
                    payload[key] = stream.value()
                else:
                    stream.consume("{")
                    selected: dict[str, Any] = {}
                    if stream.character() == "}":
                        stream.position += 1
                    else:
                        while True:
                            query_id = stream.value()
                            if not isinstance(query_id, str):
                                raise ValueError("candidate pool query ID is not a string")
                            stream.consume(":")
                            record = stream.value()
                            if query_id in wanted:
                                selected[query_id] = record
                            delimiter = stream.character()
                            if delimiter == ",":
                                stream.position += 1
                                continue
                            if delimiter == "}":
                                stream.position += 1
                                break
                            raise ValueError("invalid candidate pool object delimiter")
                    payload["pools"] = selected
                delimiter = stream.character()
                if delimiter == ",":
                    stream.position += 1
                    continue
                if delimiter == "}":
                    stream.position += 1
                    break
                raise ValueError("invalid candidate pool top-level delimiter")
    finally:
        stream.close()
    if set(payload.get("pools", {})) != wanted:
        missing = sorted(wanted - set(payload.get("pools", {})))
        raise ValueError(f"saved pool has no selected query records: {missing[:5]}")
    return _decode_candidate_payload(payload, source_doc_ids)


def _load_pool_metadata(
    pool_root: Path,
    task: str,
    *,
    include_candidate_pools: bool = True,
) -> dict[str, Any]:
    metadata = _metadata_dir(pool_root, task)
    source_doc_ids = _json_load(metadata / "source_doc_ids.json")
    if not isinstance(source_doc_ids, Mapping):
        raise ValueError("source pool source_doc_ids must be an object")
    result: dict[str, Any] = {
        "root": pool_root,
        "metadata": metadata,
        "manifest": _json_load(metadata / "manifest.json"),
        "candidate_pools": None,
        "queries": [
            dict(row)
            for line in (metadata / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        ],
        "qrels": _json_load(metadata / "qrels.json"),
        "original_qrels": _json_load(metadata / "original_qrels.json"),
        "source_doc_ids": dict(source_doc_ids),
        "doc_id_files": _json_load(metadata / "doc_id_files.json"),
        "file_inventory": _json_load(metadata / "file_inventory.json"),
        "source_identity": _json_load(metadata / "source_identity.json"),
    }
    if include_candidate_pools:
        candidate = _json_load(metadata / "candidate_pools.json")
        if not isinstance(candidate, Mapping):
            raise ValueError("source candidate pools must be an object")
        result["candidate_pools"] = _decode_candidate_payload(candidate, source_doc_ids)
    return result


def _load_pool_identity(pool_root: Path, task: str) -> dict[str, Any]:
    """Load only the small release identity needed for a tier cross-check."""

    metadata = _metadata_dir(pool_root, task)
    return {
        "manifest": _json_load(metadata / "manifest.json"),
        "source_identity": _json_load(metadata / "source_identity.json"),
    }


def _query_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        query_id = _query_id(row)
        if query_id in result:
            raise ValueError(f"duplicate query ID: {query_id!r}")
        result[query_id] = row
    return result


def _source_corpus_reference(pool_root: Path, task: str) -> Path:
    marker = pool_root / "shared_corpus.json"
    if not marker.is_file():
        raise ValueError(f"pool has no shared_corpus.json reference: {pool_root}")
    value = _json_load(marker)
    if not isinstance(value, Mapping) or value.get("schema_version") != "dci-shared-pool-corpus-v1":
        raise ValueError(f"unsupported source shared corpus marker: {marker}")
    relative = value.get("corpus")
    if relative != f"source/workspaces/{task}/corpus":
        raise ValueError(f"source shared corpus marker task mismatch: {marker}")
    corpus = pool_root.parent.parent / relative
    if corpus.is_symlink() or not corpus.is_dir():
        raise ValueError(f"source shared corpus directory is missing: {corpus}")
    return corpus.resolve()


def _selected_pool_record(
    candidate_payload: Mapping[str, Any], query_id: str
) -> Mapping[str, Any]:
    pools = candidate_payload.get("pools")
    if not isinstance(pools, Mapping) or query_id not in pools or not isinstance(pools[query_id], Mapping):
        raise ValueError(f"saved pool has no record for query {query_id!r}")
    return pools[query_id]


def _tier_suite(
    stage: Path,
    *,
    task: str,
    tier: str,
    capacity: int,
    query_ids: Sequence[str],
    query_map: Mapping[str, Mapping[str, Any]],
    qrels: Mapping[str, Any],
    original_qrels: Mapping[str, Any],
    source_doc_ids: Mapping[str, str],
    full_records: Mapping[str, Mapping[str, Any]],
    selected_records: Mapping[str, Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    source_corpus: Path,
    source_pool_root: Path,
    task_selection: Mapping[str, Any],
) -> Path:
    """Write one metadata-only tier suite in the legacy loader layout."""

    from dci_bench.data.document_pools import sha256_file as legacy_sha256_file
    from dci_bench.data.document_pools import stable_json_sha256 as legacy_stable_json_sha256

    tier_root = stage / "tasks" / task / tier
    metadata = tier_root / "metadata" / task
    workspace = tier_root / "workspaces" / task
    metadata.mkdir(parents=True)
    workspace.mkdir(parents=True)

    selected_aliases: set[str] = set()
    selected_originals: dict[str, str] = {}
    records: dict[str, Any] = {}
    canonical_qrels: dict[str, Any] = {}
    raw_original: dict[str, Any] = {}
    query_rows: list[dict[str, Any]] = []
    for query_id in sorted(query_ids):
        record = selected_records[query_id]
        aliases = list(record["candidate_doc_ids"])
        originals = list(record.get("candidate_original_doc_ids", aliases))
        if len(aliases) != len(originals):
            raise ValueError(f"candidate IDs differ for {task}/{query_id}")
        selected_aliases.update(aliases)
        selected_originals.update(dict(zip(aliases, originals)))
        selected_scores = record.get("candidate_scores")
        ordering = record.get("ordering", "scored" if selected_scores is not None else "unscored")
        record_audit = dict(record.get("audit", {}))
        nested_audits = task_selection.get("nested_audits", {})
        nested_audit = nested_audits.get(query_id) if isinstance(nested_audits, Mapping) else None
        record_audit.update(
            {
                "query_id": query_id,
                "selected_ids": originals,
                "selected_scores": selected_scores,
                "ordering": ordering,
                "capacity": capacity,
                "source_size": len(source_doc_ids),
                "benchmark_v2_derivation": POOL_DERIVATION_VERSION,
            }
        )
        if isinstance(nested_audit, Mapping):
            record_audit["nested_pool_audit"] = copy.deepcopy(dict(nested_audit))
        records[query_id] = {
            "query_id": query_id,
            "candidate_doc_ids": aliases,
            "candidate_original_doc_ids": originals,
            "candidate_scores": selected_scores,
            "ordering": ordering,
            "num_docs": len(aliases),
            "audit": record_audit,
        }
        # A selected-query suite carries only labels whose documents are in
        # that tier's selected union.  Existing source qrels may contain
        # labels for documents outside the saved candidate pool; retaining
        # those rows would make the compact mapping incomplete.
        canonical_qrels[query_id] = {
            alias: score for alias, score in qrels[query_id].items() if alias in selected_originals
        }
        raw_original[query_id] = {
            original: score
            for original, score in original_qrels[query_id].items()
            if original in selected_originals.values()
        }
        query_rows.append(dict(query_map[query_id]))

    # Keep the complete canonical alias map from the source release.  It is
    # compact metadata (no document bodies) and is required by the legacy
    # runtime to reconstruct exact source IDs; queries, qrels, and inventories
    # below remain selected-query/tier scoped.
    mapping = {alias: source_doc_ids[alias] for alias in sorted(source_doc_ids)}
    doc_id_files = {alias: f"{alias}.md" for alias in mapping}
    for record in records.values():
        record["audit"]["source_size"] = len(mapping)
    source_files = _file_inventory_by_alias(source_pool_root, task)
    inventory: dict[str, Any] = {}
    workspace_hashes: dict[str, str] = {}
    for alias in sorted(selected_aliases):
        row = source_files.get(alias)
        if row is None:
            raise ValueError(f"source pool inventory has no selected alias: {alias}")
        relative = f"workspaces/{task}/corpus/{alias}.md"
        digest = row.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"source pool inventory has no digest for alias: {alias}")
        inventory[relative] = {
            "alias": alias,
            "original_id": mapping[alias],
            "bytes": row["bytes"],
            "sha256": digest,
        }
        workspace_hashes[relative] = digest

    query_rows.sort(key=lambda row: _query_id(row))
    qrels_hash = legacy_stable_json_sha256(canonical_qrels)
    queries_hash = legacy_stable_json_sha256(query_rows)
    mapping_hash = legacy_stable_json_sha256(mapping)
    original_qrels_hash = legacy_stable_json_sha256(raw_original)
    corpus_hash = legacy_stable_json_sha256(
        {row["alias"]: row["sha256"] for _, row in sorted(inventory.items())}
    )
    identity = {
        "queries_hash": queries_hash,
        "corpus_files_sha256": corpus_hash,
        "source_doc_ids_hash": mapping_hash,
    }
    manifest = {
        "schema_version": "dci-document-pools-20261029-v2",
        "benchmark_v2": True,
        "task": task,
        "dataset": str(source_manifest.get("dataset", task)),
        "revision": str(source_manifest.get("revision", "catalog")),
        "resolved_revision": str(source_manifest.get("resolved_revision", source_manifest.get("revision", "catalog"))),
        "task_instruction": str(source_manifest.get("task_instruction", "")),
        "document_id_policy": source_manifest.get("document_id_policy", "short-document-id-v1"),
        "pool_size": tier,
        "capacity": capacity,
        "seed": DEFAULT_SEED,
        "num_queries": len(query_rows),
        "corpus_size": len(mapping),
        "qrels_hash": qrels_hash,
        "queries_hash": queries_hash,
        "source_doc_ids_hash": mapping_hash,
        "original_qrels_hash": original_qrels_hash,
        "source_identity": identity,
        "source_manifest_sha256": source_manifest.get("source_manifest_sha256"),
        "source_inventory_sha256": source_manifest.get("source_inventory_sha256"),
        "metadata_dir": f"metadata/{task}",
        "workspace_dir": f"workspaces/{task}",
        "shared_corpus_dir": f"workspaces/{task}/corpus",
        "rankings_schema_version": None,
        "rankings_path": None,
        "verification_policy": "declared-source-digests",
        "content_hashes_verified": False,
        "sampling": dict(task_selection),
    }
    candidate_payload = {
        "schema_version": "dci-document-pools-20261029-v2",
        "benchmark_v2": True,
        "task": task,
        "pool_size": tier,
        "capacity": capacity,
        "seed": DEFAULT_SEED,
        "source_identity": identity,
        "source_release": {
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "source_inventory_sha256": manifest.get("source_inventory_sha256"),
        },
        "shared_corpus": "corpus",
        "num_queries": len(query_rows),
        "corpus_size": len(mapping),
        "pools": records,
    }

    def write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    write_json(metadata / "candidate_pools.json", candidate_payload)
    write_json(metadata / "doc_id_files.json", doc_id_files)
    write_json(metadata / "file_inventory.json", inventory)
    write_json(metadata / "manifest.json", manifest)
    write_json(metadata / "original_qrels.json", raw_original)
    write_json(metadata / "qrels.json", canonical_qrels)
    (metadata / "queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in query_rows),
        encoding="utf-8",
    )
    write_json(metadata / "source_doc_ids.json", mapping)
    write_json(metadata / "source_identity.json", identity)

    marker = {
        "schema_version": "dci-shared-pool-corpus-v2",
        "task": task,
        "corpus": str(source_corpus),
        "source_pool": str(source_pool_root),
    }
    write_json(tier_root / "shared_corpus.json", marker)
    metadata_hashes = {
        name: legacy_sha256_file(metadata / name)
        for name in (
            "candidate_pools.json",
            "doc_id_files.json",
            "file_inventory.json",
            "manifest.json",
            "original_qrels.json",
            "qrels.json",
            "queries.jsonl",
            "source_doc_ids.json",
            "source_identity.json",
        )
    }
    completion = {
        "schema_version": "dci-document-pools-20261029-v2",
        "benchmark_v2": True,
        "task": task,
        "metadata_hashes": metadata_hashes,
        "workspace_hashes": workspace_hashes,
        "file_inventory_sha256": legacy_sha256_file(metadata / "file_inventory.json"),
        "shared_corpus_sha256": legacy_sha256_file(tier_root / "shared_corpus.json"),
        "verification_policy": "declared-source-digests",
        "content_hashes_verified": False,
    }
    write_json(metadata / "_COMPLETE.json", completion)
    return tier_root


def _file_inventory_by_alias(pool_root: Path, task: str) -> dict[str, Mapping[str, Any]]:
    metadata = _metadata_dir(pool_root, task)
    inventory = _json_load(metadata / "file_inventory.json")
    if not isinstance(inventory, Mapping):
        raise ValueError("source pool file inventory must be an object")
    result: dict[str, Mapping[str, Any]] = {}
    for row in inventory.values():
        if isinstance(row, Mapping) and isinstance(row.get("alias"), str):
            result[row["alias"]] = row
    return result


def _write_suite_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_benchmark_v2(
    *,
    output_dir: Path,
    config_path: Path | None = None,
    catalog_path: Path,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare a metadata-only v2 suite from the existing pool catalog.

    ``output_dir`` is immutable by default.  ``force`` is accepted only for
    API compatibility and deliberately rejected so an old artifact cannot be
    replaced accidentally.
    """

    _validate_seed(seed)
    if seed != DEFAULT_SEED:
        raise ValueError("benchmark_v2 config freezes seed 42")
    if force:
        raise ValueError("benchmark_v2 refuses overwrite; choose a new output directory")
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite benchmark output: {output_dir}")
    config = load_benchmark_config(config_path)
    catalog = _json_load(Path(catalog_path))
    if not isinstance(catalog, Mapping):
        raise ValueError("catalog must be a JSON object")
    assessment = assess_source_eligibility(config, catalog)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)

    def empty_task(task_id: str, *, reason: str | None, source_id: str | None = None) -> dict[str, Any]:
        domain, language = task_id.rsplit("_", 1)
        return {
            "task": task_id,
            "domain": domain,
            "language": language,
            "source_task": None,
            "status": "blocked",
            "reason": reason,
            "source_id": source_id,
            "query_ids": {"dev": [], "test": [], "smoke": []},
            "query_groups": {},
            "pools": {"lite": None, "full": None},
            "exclusions": [],
            "overflow_excluded_count": 0,
            "excluded_query_count": 0,
            "dedup_excluded_count": 0,
        }

    task_manifests = {
        task_id: empty_task(
            task_id,
            reason=assessment[task_id].get("reason"),
            source_id=assessment[task_id].get("source_id"),
        )
        for task_id in TASK_IDS
    }
    source_data: dict[str, dict[str, Any]] = {}
    records_for_sampling: dict[str, list[dict[str, Any]]] = {}
    contexts_for_sampling: dict[str, dict[str, Any]] = {}
    for task_id in TASK_IDS:
        item = assessment[task_id]
        if item["status"] != "eligible":
            continue
        source = item["source"]
        source_id = item["source_id"]
        source_task = source.get("task", source_id)
        try:
            full_root = _pool_path(source, FULL_CAPACITY)
            lite_root = _pool_path(source, LITE_CAPACITY)
            if full_root is None or lite_root is None:
                raise SourceBlockedError(f"source {source_id!r} has no full/lite pools")
            # Query/qrel metadata is small enough to retain for the global
            # sampler.  Candidate pools are intentionally loaded only after
            # the 60 IDs for a task have been selected.
            full_meta = _load_pool_metadata(
                full_root, source_task, include_candidate_pools=False
            )
            lite_meta = _load_pool_identity(lite_root, source_task)
            if (
                full_meta["manifest"].get("capacity") != FULL_CAPACITY
                or lite_meta["manifest"].get("capacity") != LITE_CAPACITY
            ):
                raise SourceBlockedError(f"source {source_id!r} pool capacities are inconsistent")
            for identity_key in ("queries_hash", "source_doc_ids_hash"):
                if full_meta["source_identity"].get(identity_key) != lite_meta["source_identity"].get(identity_key):
                    raise SourceBlockedError(f"source {source_id!r} full/lite identities differ")
            rows = _records(full_meta["queries"])
            qrels = full_meta["qrels"]
            if not isinstance(qrels, Mapping):
                raise SourceBlockedError(f"source {source_id!r} qrels are not an object")
            eligible_rows, exclusions, overflow = _prepare_query_eligibility(rows, qrels)
            if len(eligible_rows) < DEV_SIZE + TEST_SIZE:
                raise SourceBlockedError(
                    f"source {source_id!r} has only {len(eligible_rows)} eligible queries after "
                    f"{len(exclusions)} pre-sampling exclusions"
                )
            source_data[task_id] = {
                "item": item,
                "source": source,
                "source_id": source_id,
                "source_task": source_task,
                "full_root": full_root,
                "lite_root": lite_root,
                "full_meta": full_meta,
                "rows": rows,
                "eligible_rows": eligible_rows,
                "overflow": overflow,
                "exclusions": exclusions,
            }
            records_for_sampling[task_id] = eligible_rows
            contexts_for_sampling[task_id] = {
                "source_id": source_id,
                "source_family": item["candidate"].get("source_family"),
            }
        except (OSError, TypeError, ValueError, KeyError) as exc:
            task_manifests[task_id]["reason"] = f"source_validation_failed: {type(exc).__name__}: {exc}"

    strategy_name = config["sampling"].get("strategy", "random_without_replacement")
    global_splits: dict[str, dict[str, list[str]]] = {}
    group_maps: dict[str, dict[str, str]] = {}
    if records_for_sampling:
        try:
            global_splits = sample_global_query_splits(
                records_for_sampling,
                source_context_by_task=contexts_for_sampling,
                seed=seed,
                strategy=strategy_name,
            )
            _, group_maps = _global_query_groups(records_for_sampling, contexts_for_sampling)
        except SamplingShortageError as exc:
            for task_id in list(records_for_sampling):
                task_manifests[task_id]["reason"] = f"sampling_blocked: {exc}"
                source_data.pop(task_id, None)
                records_for_sampling.pop(task_id, None)
        except (TypeError, ValueError) as exc:
            for task_id in list(records_for_sampling):
                task_manifests[task_id]["reason"] = f"sampling_failed: {type(exc).__name__}: {exc}"
                source_data.pop(task_id, None)
                records_for_sampling.pop(task_id, None)

    for task_id in TASK_IDS:
        data = source_data.get(task_id)
        if data is None or task_id not in global_splits:
            continue
        item = data["item"]
        source = data["source"]
        source_id = data["source_id"]
        source_task = data["source_task"]
        full_meta = data["full_meta"]
        rows = data["rows"]
        splits = global_splits[task_id]
        qrels = full_meta["qrels"]
        original_qrels = full_meta["original_qrels"]
        source_ids = full_meta["source_doc_ids"]
        query_map = _query_map(rows)
        full_records: dict[str, Mapping[str, Any]] = {}
        full_selected: dict[str, Mapping[str, Any]] = {}
        lite_records: dict[str, Mapping[str, Any]] = {}
        nested_audits: dict[str, Any] = {}
        alias_to_original = dict(source_ids)
        selected_ids = sorted(set(splits["dev"]) | set(splits["test"]))
        try:
            full_meta = {
                **full_meta,
                "candidate_pools": _load_selected_candidate_payload(
                    data["full_root"],
                    source_task,
                    selected_ids,
                    source_ids,
                ),
            }
            for query_id in selected_ids:
                base = _selected_pool_record(full_meta["candidate_pools"], query_id)
                full_records[query_id] = base
                aliases = base["candidate_doc_ids"]
                originals = base.get("candidate_original_doc_ids")
                if originals is None:
                    originals = [alias_to_original[alias] for alias in aliases]
                alias_to_original.update(dict(zip(aliases, originals)))
                selection = select_nested_pools(
                    aliases,
                    qrels[query_id],
                    scores=base.get("candidate_scores"),
                    lite_capacity=LITE_CAPACITY,
                    full_capacity=FULL_CAPACITY,
                )
                nested_audits[query_id] = selection["audit"]
                if selection["full"] is None or selection["lite"] is None:
                    raise SourceBlockedError(
                        f"selected query {query_id!r} cannot satisfy both v2 pool capacities"
                    )
                full_selected[query_id] = {
                    **dict(base),
                    "candidate_doc_ids": selection["full"],
                    "candidate_original_doc_ids": [alias_to_original[doc_id] for doc_id in selection["full"]],
                    "candidate_scores": selection["full_scores"],
                    "ordering": "scored" if selection["full_scores"] is not None else "unscored",
                    "num_docs": len(selection["full"]),
                }
                lite_records[query_id] = {
                    **dict(base),
                    "candidate_doc_ids": selection["lite"],
                    "candidate_original_doc_ids": [alias_to_original[doc_id] for doc_id in selection["lite"]],
                    "candidate_scores": selection["lite_scores"],
                    "ordering": "scored" if selection["lite_scores"] is not None else "unscored",
                    "num_docs": len(selection["lite"]),
                }
            source_corpus = _source_corpus_reference(data["full_root"], source_task)
            task_selection = {
                "source_id": source_id,
                "source_family": item["candidate"].get("source_family"),
                "provenance": item["candidate"].get("provenance"),
                "splits": splits,
                "nested_audits": nested_audits,
                "exclusions": data["exclusions"],
                "full_source_manifest_sha256": full_meta["manifest"].get("source_manifest_sha256"),
                "full_pool_manifest_sha256": sha256_file(
                    data["full_root"] / "metadata" / source_task / "manifest.json"
                ),
            }
            full_root = _tier_suite(
                stage,
                task=task_id,
                tier="full",
                capacity=FULL_CAPACITY,
                query_ids=selected_ids,
                query_map=query_map,
                qrels=qrels,
                original_qrels=original_qrels,
                source_doc_ids=source_ids,
                full_records=full_records,
                selected_records=full_selected,
                source_manifest=full_meta["manifest"],
                source_identity=full_meta["source_identity"],
                source_corpus=source_corpus,
                source_pool_root=data["full_root"],
                task_selection=task_selection,
            )
            lite_root = _tier_suite(
                stage,
                task=task_id,
                tier="lite",
                capacity=LITE_CAPACITY,
                query_ids=selected_ids,
                query_map=query_map,
                qrels=qrels,
                original_qrels=original_qrels,
                source_doc_ids=source_ids,
                full_records=full_records,
                selected_records=lite_records,
                source_manifest=full_meta["manifest"],
                source_identity=full_meta["source_identity"],
                source_corpus=source_corpus,
                source_pool_root=data["full_root"],
                task_selection=task_selection,
            )
            domain, language = task_id.rsplit("_", 1)
            query_groups = {
                query_id: group_maps[task_id][query_id] for query_id in selected_ids
            }
            pool_entries = {
                "full": {
                    "suite": f"tasks/{task_id}/full",
                    "profile": "document-pool-v2-full",
                    "capacity": FULL_CAPACITY,
                    "query_ids": selected_ids,
                    "pool_sha256": sha256_file(full_root / "metadata" / task_id / "_COMPLETE.json"),
                },
                "lite": {
                    "suite": f"tasks/{task_id}/lite",
                    "profile": "document-pool-v2-lite",
                    "capacity": LITE_CAPACITY,
                    "query_ids": selected_ids,
                    "pool_sha256": sha256_file(lite_root / "metadata" / task_id / "_COMPLETE.json"),
                },
            }
            task_manifests[task_id] = {
                "task": task_id,
                "domain": domain,
                "language": language,
                "source_task": source_task,
                "status": "ready",
                "reason": None,
                "source_id": source_id,
                "source_identity": full_meta["source_identity"],
                "source_manifest_sha256": full_meta["manifest"].get("source_manifest_sha256"),
                "splits": splits,
                "query_ids": splits,
                "query_groups": query_groups,
                "group_policy": (
                    "normalized query text plus explicit query-level group/parallel/source-query IDs; "
                    "source-family names never group an entire dataset"
                ),
                "tier_query_ids": {"full": selected_ids, "lite": selected_ids},
                "pools": pool_entries,
                "exclusions": data["exclusions"],
                "overflow_excluded_count": len(
                    {exclusion["query_id"] for exclusion in data["overflow"]}
                ),
                "excluded_query_count": len(
                    {exclusion["query_id"] for exclusion in data["exclusions"]}
                ),
                "dedup_excluded_count": len(
                    {
                        exclusion["query_id"]
                        for exclusion in data["exclusions"]
                        if exclusion.get("reason") in {
                            "duplicate_query_text",
                            "duplicate_query_conflicting_qrels",
                        }
                    }
                ),
                "eligible_query_count": len(data["eligible_rows"]),
                "sampling_input_digest": stable_json_sha256(data["eligible_rows"]),
            }
        except (OSError, TypeError, ValueError, KeyError) as exc:
            task_manifests[task_id]["reason"] = f"pool_preparation_failed: {type(exc).__name__}: {exc}"

    manifest = build_sampling_manifest(
        task_manifests,
        seed=seed,
        strategy=strategy_name,
        config_identity={
            "config_sha256": stable_json_sha256(config),
            "catalog_sha256": stable_json_sha256(catalog),
        },
        input_identity={
            task: task_manifests[task].get("sampling_input_digest") for task in TASK_IDS
        },
    )
    manifest["tasks"] = {task_id: task_manifests[task_id] for task_id in TASK_IDS}
    all_ready = all(
        task_manifests[task_id].get("status") == "ready"
        and len(task_manifests[task_id]["query_ids"]["dev"]) == DEV_SIZE
        and len(task_manifests[task_id]["query_ids"]["test"]) == TEST_SIZE
        and task_manifests[task_id]["pools"]["lite"] is not None
        and task_manifests[task_id]["pools"]["full"] is not None
        for task_id in TASK_IDS
    )
    manifest["status"] = "ready" if all_ready else "partial_blocked"
    manifest["ready_for_formal"] = all_ready
    manifest["group_policy"] = (
        "normalized query text plus explicit query-level group/parallel/source-query IDs; "
        "source-family names never group an entire dataset"
    )
    manifest["overall_sha256"] = _manifest_sha256(manifest)
    _write_suite_json(stage / "manifest.json", manifest)
    _write_suite_json(stage / "benchmark-manifest.json", manifest)
    _write_suite_json(stage / "config.json", config)
    _write_suite_json(stage / "assessment.json", assessment)
    os.replace(stage, output_dir)
    return manifest


def audit_benchmark_v2(output_dir: Path) -> dict[str, Any]:
    """Audit a prepared metadata-only suite without touching source bodies."""

    output_dir = Path(output_dir)
    manifest = _json_load(output_dir / "manifest.json")
    benchmark_manifest = _json_load(output_dir / "benchmark-manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid benchmark v2 manifest")
    if benchmark_manifest != manifest:
        raise ValueError("benchmark-manifest.json differs from manifest.json")
    declared_hash = manifest.get("overall_sha256")
    if declared_hash != _manifest_sha256(manifest):
        raise ValueError("benchmark v2 overall_sha256 does not match manifest contents")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASK_IDS):
        raise ValueError("benchmark v2 manifest must contain all 20 tasks")

    memberships: dict[str, set[str]] = {}
    for task_id, task in tasks.items():
        if not isinstance(task, Mapping):
            raise ValueError(f"task manifest is invalid: {task_id}")
        splits = task.get("query_ids", {})
        if not isinstance(splits, Mapping):
            raise ValueError(f"task {task_id} query_ids are invalid")
        if task.get("status") == "blocked":
            if any(splits.get(split) for split in ("dev", "test", "smoke")):
                raise ValueError(f"blocked task {task_id} has sampled queries")
            if task.get("pools") != {"lite": None, "full": None}:
                raise ValueError(f"blocked task {task_id} has pools")
            continue
        if task.get("status") != "ready":
            raise ValueError(f"task {task_id} has invalid status")
        if len(splits.get("dev", [])) != DEV_SIZE or len(splits.get("test", [])) != TEST_SIZE:
            raise ValueError(f"task {task_id} has invalid dev/test counts")
        for split in ("dev", "test", "smoke"):
            values = splits.get(split, [])
            if not isinstance(values, list) or len(values) != len(set(values)):
                raise ValueError(f"task {task_id} has duplicate or invalid {split} IDs")
        if len(splits.get("smoke", [])) != SMOKE_SIZE or not set(splits.get("smoke", [])).issubset(set(splits.get("dev", []))):
            raise ValueError(f"task {task_id} smoke IDs are not a dev subset")
        if set(splits.get("dev", [])) & set(splits.get("test", [])):
            raise ValueError(f"task {task_id} dev/test IDs overlap")
        if task.get("splits") != splits:
            raise ValueError(f"task {task_id} splits and query_ids differ")
        groups = task.get("query_groups", {})
        if not isinstance(groups, Mapping):
            raise ValueError(f"task {task_id} query_groups are invalid")
        for split in ("dev", "test"):
            for query_id in splits[split]:
                group = groups.get(query_id)
                if not isinstance(group, str) or not group:
                    raise ValueError(f"task {task_id} has no query group for {query_id}")
                memberships.setdefault(group, set()).add(split)
        selected_ids = sorted(set(splits["dev"]) | set(splits["test"]))
        tier_ids = task.get("tier_query_ids")
        if tier_ids != {"full": selected_ids, "lite": selected_ids}:
            raise ValueError(f"task {task_id} tier query IDs differ from sampled IDs")
        pools = task.get("pools")
        if not isinstance(pools, Mapping):
            raise ValueError(f"task {task_id} pools are invalid")
        if set(pools) != {"lite", "full"}:
            raise ValueError(f"task {task_id} pools must contain exactly lite and full")
        payloads: dict[str, Mapping[str, Any]] = {}
        from dci_bench.data.document_pools import load_document_pool_manifest

        for tier, capacity in (("lite", LITE_CAPACITY), ("full", FULL_CAPACITY)):
            entry = pools.get(tier)
            if not isinstance(entry, Mapping):
                raise ValueError(f"task {task_id} {tier} pool entry is missing")
            suite = entry.get("suite")
            if not isinstance(suite, str) or not suite or os.path.isabs(suite) or "\\" in suite:
                raise ValueError(f"task {task_id} {tier} suite path is invalid")
            root = output_dir / Path(suite)
            if root.resolve().relative_to(output_dir.resolve()) is None:
                raise ValueError(f"task {task_id} {tier} suite escapes output")
            completion = root / "metadata" / task_id / "_COMPLETE.json"
            if entry.get("pool_sha256") != sha256_file(completion):
                raise ValueError(f"task {task_id} {tier} pool_sha256 mismatch")
            if (
                entry.get("capacity") != capacity
                or entry.get("query_ids") != selected_ids
                or entry.get("profile") != f"document-pool-v2-{tier}"
            ):
                raise ValueError(f"task {task_id} {tier} manifest pool entry is inconsistent")
            payload = load_document_pool_manifest(root, task=task_id, verify_content_hashes=False)
            if payload.get("capacity") != capacity or payload.get("pool_size") != tier:
                raise ValueError(f"task {task_id} {tier} tier identity mismatch")
            if set(payload.get("pools", {})) != set(selected_ids):
                raise ValueError(f"task {task_id} {tier} query IDs differ from manifest")
            payloads[tier] = payload
        lite_payload = payloads["lite"]
        full_payload = payloads["full"]
        lite_qrels = _json_load(output_dir / Path(pools["lite"]["suite"]) / "metadata" / task_id / "qrels.json")
        full_qrels = _json_load(output_dir / Path(pools["full"]["suite"]) / "metadata" / task_id / "qrels.json")
        if lite_qrels != full_qrels:
            raise ValueError(f"task {task_id} lite/full qrels differ")
        for query_id in selected_ids:
            lite_record = lite_payload["pools"][query_id]
            full_record = full_payload["pools"][query_id]
            lite_ids = set(lite_record["candidate_doc_ids"])
            full_ids = set(full_record["candidate_doc_ids"])
            if len(lite_ids) != LITE_CAPACITY or len(full_ids) != FULL_CAPACITY:
                raise ValueError(f"task {task_id} {query_id} actual pool count is invalid")
            if not lite_ids < full_ids:
                raise ValueError(f"task {task_id} {query_id} lite pool is not a strict full subset")
            positives = {
                doc_id for doc_id, score in lite_qrels[query_id].items() if float(score) > 0
            }
            if not positives.issubset(lite_ids) or not positives.issubset(full_ids):
                raise ValueError(f"task {task_id} {query_id} dropped a positive label")
            for tier, record, ids in (
                ("lite", lite_record, lite_ids),
                ("full", full_record, full_ids),
            ):
                originals = record.get("candidate_original_doc_ids")
                audit = record.get("audit")
                if (
                    not isinstance(originals, list)
                    or len(originals) != len(ids)
                    or not isinstance(audit, Mapping)
                    or audit.get("selected_ids") != originals
                ):
                    raise ValueError(
                        f"task {task_id} {query_id} {tier} audit IDs are not actual source IDs"
                    )
                nested_audit = audit.get("nested_pool_audit")
                if nested_audit is not None and (
                    not isinstance(nested_audit, Mapping)
                    or nested_audit.get("nested") is not True
                    or nested_audit.get("selected_counts") != {
                        "lite": LITE_CAPACITY,
                        "full": FULL_CAPACITY,
                    }
                ):
                    raise ValueError(f"task {task_id} {query_id} nested pool audit is invalid")
    leakage = {group: sorted(parts) for group, parts in memberships.items() if len(parts) > 1}
    if leakage:
        raise ValueError(f"global query group leakage between dev/test: {sorted(leakage)}")
    ready_for_formal = all(task.get("status") == "ready" for task in tasks.values())
    if bool(manifest.get("ready_for_formal")) != ready_for_formal:
        raise ValueError("ready_for_formal does not reflect all twenty complete tasks")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": manifest.get("status"),
        "ready_for_formal": ready_for_formal,
        "overall_sha256": declared_hash,
        "task_count": len(tasks),
        "ready_tasks": sorted(task_id for task_id, task in tasks.items() if task.get("status") == "ready"),
        "blocked_tasks": sorted(task_id for task_id, task in tasks.items() if task.get("status") == "blocked"),
    }


__all__ = [
    "DEFAULT_SEED",
    "DEV_SIZE",
    "DOMAINS",
    "FULL_CAPACITY",
    "LANGUAGES",
    "LITE_CAPACITY",
    "POOL_DERIVATION_VERSION",
    "RandomWithoutReplacement",
    "SAMPLING_ALGORITHM_VERSION",
    "SCHEMA_VERSION",
    "SMOKE_SIZE",
    "SourceBlockedError",
    "SamplingShortageError",
    "TASK_IDS",
    "TEST_SIZE",
    "assess_source_eligibility",
    "assert_no_group_leakage",
    "assert_global_group_leakage",
    "audit_benchmark_v2",
    "build_sampling_manifest",
    "load_benchmark_config",
    "prepare_benchmark_v2",
    "register_sampling_strategy",
    "sample_query_ids",
    "sample_global_query_splits",
    "sample_query_splits",
    "select_nested_pools",
    "sha256_file",
    "stable_json_sha256",
]
