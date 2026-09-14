"""Offline retrieval scoring for DCI-Bench MVP."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from dci_bench.protocol.contracts import TOP_K

DEFAULT_METRIC_KS = (1, 3, 5, 10, 20)


class InvalidRankingError(ValueError):
    """Raised when an agent final answer violates the Phase 0 JSON contract."""


def validate_ranked_doc_ids(payload: str | Mapping[str, Any], top_k: int = TOP_K) -> list[str]:
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise InvalidRankingError(f"Final answer is not valid JSON: {exc}") from exc
    else:
        parsed = dict(payload)

    if set(parsed) != {"ranked_doc_ids"}:
        raise InvalidRankingError("Final answer must contain only the ranked_doc_ids key")
    ranked = parsed["ranked_doc_ids"]
    if not isinstance(ranked, list) or not ranked:
        raise InvalidRankingError("ranked_doc_ids must be a non-empty list")
    if len(ranked) > top_k:
        raise InvalidRankingError(f"ranked_doc_ids must contain at most {top_k} ids")
    if not all(isinstance(doc_id, str) and doc_id for doc_id in ranked):
        raise InvalidRankingError("Every ranked_doc_ids item must be a non-empty string")
    if len(set(ranked)) != len(ranked):
        raise InvalidRankingError("ranked_doc_ids must not contain duplicate ids")
    return ranked


def dcg_at_k(relevances: Sequence[float], k: int = TOP_K) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(ranked_doc_ids: Sequence[str], qrels: Mapping[str, float], k: int = TOP_K) -> float:
    if not qrels:
        return 0.0
    gains = [float(qrels.get(doc_id, 0.0)) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted((float(score) for score in qrels.values()), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg


def recall_at_k(ranked_doc_ids: Sequence[str], qrels: Mapping[str, float], k: int = TOP_K) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if float(score) > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant.intersection(retrieved)) / len(relevant)


def f1_at_k(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, float],
    k: int = TOP_K,
) -> float:
    """Return the harmonic mean of precision@k and recall@k."""
    relevant = {doc_id for doc_id, score in qrels.items() if float(score) > 0}
    if not relevant:
        return 0.0
    true_positives = len(relevant.intersection(ranked_doc_ids[:k]))
    precision = true_positives / k
    recall = true_positives / len(relevant)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_metric_ks(metric_ks: Sequence[int] | None = None) -> tuple[int, ...]:
    ks = tuple(DEFAULT_METRIC_KS if metric_ks is None else metric_ks)
    if not ks:
        raise ValueError("metric_ks must contain at least one cutoff")
    if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("metric_ks must contain only positive integers")
    if len(set(ks)) != len(ks):
        raise ValueError("metric_ks must not contain duplicate cutoffs")
    return ks


def zero_metrics(metric_ks: Sequence[int] | None = None) -> dict[str, float]:
    ks = normalize_metric_ks(metric_ks)
    return {
        **{f"recall_at_{k}": 0.0 for k in ks},
        **{f"f1_at_{k}": 0.0 for k in ks},
        **{f"ndcg_at_{k}": 0.0 for k in ks},
    }


def score_query(
    payload: str | Mapping[str, Any] | Sequence[str],
    qrels: Mapping[str, float],
    k: int | None = None,
    *,
    metric_ks: Sequence[int] | None = None,
) -> dict[str, Any]:
    if k is not None and metric_ks is not None:
        raise ValueError("Pass either k or metric_ks, not both")
    ks = normalize_metric_ks((k,) if k is not None else metric_ks)
    try:
        if isinstance(payload, Sequence) and not isinstance(
            payload, (str, bytes, bytearray)
        ):
            ranked_doc_ids = list(payload)
            validate_ranked_doc_ids({"ranked_doc_ids": ranked_doc_ids})
        else:
            ranked_doc_ids = validate_ranked_doc_ids(payload)  # type: ignore[arg-type]
    except InvalidRankingError as exc:
        return {
            "valid_output": False,
            "failure_reason": str(exc),
            **zero_metrics(ks),
        }

    return {
        "valid_output": True,
        "failure_reason": None,
        **{f"recall_at_{k}": recall_at_k(ranked_doc_ids, qrels, k) for k in ks},
        **{f"f1_at_{k}": f1_at_k(ranked_doc_ids, qrels, k) for k in ks},
        **{f"ndcg_at_{k}": ndcg_at_k(ranked_doc_ids, qrels, k) for k in ks},
    }


def score_run(
    predictions: Mapping[str, str | Mapping[str, Any] | Sequence[str]],
    qrels_by_query: Mapping[str, Mapping[str, float]],
    k: int | None = None,
    *,
    metric_ks: Sequence[int] | None = None,
) -> dict[str, Any]:
    if k is not None and metric_ks is not None:
        raise ValueError("Pass either k or metric_ks, not both")
    ks = normalize_metric_ks((k,) if k is not None else metric_ks)
    per_query = {
        query_id: score_query(
            predictions.get(query_id, {"ranked_doc_ids": []}),
            qrels,
            metric_ks=ks,
        )
        for query_id, qrels in qrels_by_query.items()
    }
    total = len(per_query)
    valid = sum(1 for result in per_query.values() if result["valid_output"])
    return {
        "num_queries": total,
        "valid_outputs": valid,
        "invalid_outputs": total - valid,
        "failure_rate": 0.0 if total == 0 else (total - valid) / total,
        **{
            f"mean_{metric}_at_{k}": (
                0.0
                if total == 0
                else sum(result[f"{metric}_at_{k}"] for result in per_query.values()) / total
            )
            for metric in ("recall", "f1", "ndcg")
            for k in ks
        },
        "per_query": per_query,
    }
