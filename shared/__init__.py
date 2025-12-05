"""Shared utilities for Mixpeek benchmarks."""

from .metrics import (
    ndcg_at_k,
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    average_precision,
    mean_average_precision,
    compute_all_metrics,
    print_metrics
)

from .evaluator import (
    Query,
    RelevanceJudgment,
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkEvaluator
)

__all__ = [
    # Metrics
    'ndcg_at_k',
    'recall_at_k',
    'precision_at_k',
    'mean_reciprocal_rank',
    'average_precision',
    'mean_average_precision',
    'compute_all_metrics',
    'print_metrics',
    # Evaluator
    'Query',
    'RelevanceJudgment',
    'BenchmarkResult',
    'BenchmarkReport',
    'BenchmarkEvaluator',
]
