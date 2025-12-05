"""
Shared metrics for benchmark evaluation.

Standard retrieval metrics used across all benchmarks:
- NDCG@k (Normalized Discounted Cumulative Gain)
- Recall@k
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- Precision@k
"""

from typing import List, Dict, Set
import numpy as np


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain at k).

    Args:
        relevance_scores: List of relevance scores for retrieved documents (in rank order)
        k: Cutoff position

    Returns:
        NDCG@k score between 0 and 1
    """
    if not relevance_scores or k <= 0:
        return 0.0

    relevance_scores = relevance_scores[:k]

    # DCG (Discounted Cumulative Gain)
    dcg = sum(
        (2 ** rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(relevance_scores)
    )

    # IDCG (Ideal DCG) - best possible ranking
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(
        (2 ** rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(ideal_scores)
    )

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
    """
    Calculate Recall@k - what fraction of relevant documents are in top k results.

    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (in rank order)
        k: Cutoff position

    Returns:
        Recall@k score between 0 and 1
    """
    if not relevant_docs or k <= 0:
        return 0.0

    retrieved_at_k = set(retrieved_docs[:k])
    found = relevant_docs.intersection(retrieved_at_k)

    return len(found) / len(relevant_docs)


def precision_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
    """
    Calculate Precision@k - what fraction of top k results are relevant.

    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (in rank order)
        k: Cutoff position

    Returns:
        Precision@k score between 0 and 1
    """
    if not retrieved_docs or k <= 0:
        return 0.0

    retrieved_at_k = retrieved_docs[:k]
    found = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_docs)

    return found / min(k, len(retrieved_at_k))


def mean_reciprocal_rank(relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
    """
    Calculate MRR (Mean Reciprocal Rank) - reciprocal of rank of first relevant document.

    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (in rank order)

    Returns:
        MRR score (1/rank of first relevant doc, or 0 if none found)
    """
    for idx, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs:
            return 1.0 / idx
    return 0.0


def average_precision(relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
    """
    Calculate AP (Average Precision) for a single query.

    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (in rank order)

    Returns:
        AP score between 0 and 1
    """
    if not relevant_docs:
        return 0.0

    precisions_at_relevant = []
    num_relevant_found = 0

    for idx, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs:
            num_relevant_found += 1
            precision = num_relevant_found / idx
            precisions_at_relevant.append(precision)

    if not precisions_at_relevant:
        return 0.0

    return sum(precisions_at_relevant) / len(relevant_docs)


def mean_average_precision(all_results: List[Dict]) -> float:
    """
    Calculate MAP (Mean Average Precision) across all queries.

    Args:
        all_results: List of dicts with 'relevant_docs' and 'retrieved_docs'

    Returns:
        MAP score between 0 and 1
    """
    if not all_results:
        return 0.0

    aps = [
        average_precision(result['relevant_docs'], result['retrieved_docs'])
        for result in all_results
    ]

    return np.mean(aps)


def compute_all_metrics(
    relevant_docs: Set[str],
    retrieved_docs: List[str],
    relevance_scores: List[float] = None,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute all standard retrieval metrics.

    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (in rank order)
        relevance_scores: Optional list of graded relevance scores (0-3)
        k_values: List of k values to compute metrics for

    Returns:
        Dictionary of metric names to scores
    """
    metrics = {}

    # If no relevance scores provided, use binary (1 if relevant, 0 otherwise)
    if relevance_scores is None:
        relevance_scores = [
            1.0 if doc_id in relevant_docs else 0.0
            for doc_id in retrieved_docs
        ]

    # NDCG@k
    for k in k_values:
        metrics[f'ndcg@{k}'] = ndcg_at_k(relevance_scores, k)

    # Recall@k
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(relevant_docs, retrieved_docs, k)

    # Precision@k
    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(relevant_docs, retrieved_docs, k)

    # MRR (no k needed)
    metrics['mrr'] = mean_reciprocal_rank(relevant_docs, retrieved_docs)

    # AP (no k needed)
    metrics['ap'] = average_precision(relevant_docs, retrieved_docs)

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metric names to scores
        title: Title to display
    """
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

    # Group metrics by type
    ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg')}
    recall_metrics = {k: v for k, v in metrics.items() if k.startswith('recall')}
    precision_metrics = {k: v for k, v in metrics.items() if k.startswith('precision')}
    other_metrics = {k: v for k, v in metrics.items() if not any(k.startswith(x) for x in ['ndcg', 'recall', 'precision'])}

    if ndcg_metrics:
        print("\nRanking Quality (NDCG):")
        for metric, score in sorted(ndcg_metrics.items()):
            print(f"  {metric:20s} {score:.4f}")

    if recall_metrics:
        print("\nCoverage (Recall):")
        for metric, score in sorted(recall_metrics.items()):
            print(f"  {metric:20s} {score:.4f}")

    if precision_metrics:
        print("\nAccuracy (Precision):")
        for metric, score in sorted(precision_metrics.items()):
            print(f"  {metric:20s} {score:.4f}")

    if other_metrics:
        print("\nOther Metrics:")
        for metric, score in sorted(other_metrics.items()):
            print(f"  {metric:20s} {score:.4f}")

    print("=" * 80 + "\n")
