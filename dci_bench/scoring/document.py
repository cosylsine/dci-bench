"""Explicit document-track API; legacy retrieval scoring remains unchanged.

Both import paths expose the very same functions, preserving the document
output schema, actual-window F1, metric defaults, and failure handling.
"""

from dci_bench.scoring.retrieval import (
    DEFAULT_METRIC_KS,
    InvalidRankingError,
    dcg_at_k,
    f1_at_k,
    ndcg_at_k,
    normalize_metric_ks,
    precision_at_k,
    recall_at_k,
    score_query,
    score_run,
    validate_ranked_doc_ids,
    zero_metrics,
)

SCORING_VERSION = "dci-retrieval-scoring-v2"

__all__ = [
    "DEFAULT_METRIC_KS", "InvalidRankingError", "SCORING_VERSION", "dcg_at_k",
    "f1_at_k", "ndcg_at_k", "normalize_metric_ks", "precision_at_k",
    "recall_at_k", "score_query", "score_run", "validate_ranked_doc_ids",
    "zero_metrics",
]
