"""Query-level publication hooks for the v2 Phase 4 runner.

Inspect owns one eager hook instance for the lifetime of a process.  The
active publication callback therefore lives in a :class:`ContextVar`, so a
hook event can only publish into the evaluation invocation that installed the
context.  The runner checks recorded errors after Inspect returns because
Inspect deliberately swallows ordinary hook exceptions.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

from inspect_ai.hooks import Hooks, SampleEnd, hooks


class IncrementalPublicationError(RuntimeError):
    """A query result could not be published by the v2 sample-end hook."""

    def __init__(self, errors: Sequence[tuple[str, Exception]]) -> None:
        self.errors = tuple(errors)
        details = "; ".join(f"{query_id}: {error}" for query_id, error in self.errors)
        super().__init__(f"incremental query publication failed: {details}")


@dataclass
class IncrementalPublication:
    """State scoped to one ``inspect_eval_fn`` invocation."""

    expected_query_ids: frozenset[str]
    publish: Callable[[str, Any], dict[str, Any]]
    published: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[tuple[str, Exception]] = field(default_factory=list)

    def publish_sample(self, query_id: str, sample: Any) -> None:
        if query_id not in self.expected_query_ids:
            raise ValueError(f"Inspect sample id is not pending: {query_id}")
        if query_id in self.published:
            raise ValueError(f"Inspect sample was published more than once: {query_id}")
        self.published[query_id] = self.publish(query_id, sample)

    def record_error(self, query_id: str, error: Exception) -> None:
        self.errors.append((query_id, error))

    def raise_if_errors(self) -> None:
        if self.errors:
            raise IncrementalPublicationError(self.errors)


_ACTIVE_PUBLICATION: ContextVar[IncrementalPublication | None] = ContextVar(
    "dci_bench_active_incremental_publication",
    default=None,
)


@hooks(
    name="dci-bench incremental query publication",
    description="Publishes v2 Phase 4 query results as Inspect samples finish.",
)
class IncrementalPublicationHook(Hooks):
    """Bridge Inspect's sample-end event to the active runner context."""

    async def on_sample_end(self, data: SampleEnd) -> None:
        publication = _ACTIVE_PUBLICATION.get()
        if publication is None:
            return
        # SampleEnd.sample is the finalized EvalSample.  Its id is the query
        # id; SampleEnd.sample_id is Inspect's per-execution UUID.
        try:
            query_id = str(data.sample.id)
            publication.publish_sample(query_id, data.sample)
        except Exception as error:
            # Inspect catches hook exceptions itself.  Retain the failure so
            # the parent can fail before task aggregation/finalization.
            query_id = str(getattr(data.sample, "id", "<unknown>"))
            publication.record_error(query_id, error)


@contextmanager
def incremental_publication(
    *,
    expected_query_ids: Sequence[str],
    publish: Callable[[str, Any], dict[str, Any]],
) -> Iterator[IncrementalPublication]:
    """Install one publication callback for the dynamic scope of an eval."""

    publication = IncrementalPublication(frozenset(expected_query_ids), publish)
    token = _ACTIVE_PUBLICATION.set(publication)
    try:
        yield publication
    finally:
        _ACTIVE_PUBLICATION.reset(token)


__all__ = [
    "IncrementalPublication",
    "IncrementalPublicationError",
    "IncrementalPublicationHook",
    "incremental_publication",
]
