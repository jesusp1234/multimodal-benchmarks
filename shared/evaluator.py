"""
Shared benchmark evaluator.

Provides a standard interface for running benchmarks across all three domains.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Callable, Any, Set
from dataclasses import dataclass, asdict
import numpy as np

from .metrics import compute_all_metrics, print_metrics, mean_average_precision


@dataclass
class Query:
    """Standard query format."""
    id: str
    text: str
    intent: str = "general"
    domain: str = ""


@dataclass
class RelevanceJudgment:
    """Standard relevance judgment format."""
    query_id: str
    doc_id: str
    relevance: int  # 0 = not relevant, 1 = relevant, 2 = highly relevant, 3 = perfect match


@dataclass
class BenchmarkResult:
    """Results from a single query."""
    query_id: str
    query_text: str
    retrieved_docs: List[str]
    relevance_scores: List[float]
    latency_ms: float
    metrics: Dict[str, float]


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    benchmark_name: str
    num_queries: int
    results: List[BenchmarkResult]
    aggregate_metrics: Dict[str, float]
    latency_stats: Dict[str, float]
    timestamp: str


class BenchmarkEvaluator:
    """
    Standard evaluator for running retrieval benchmarks.

    Usage:
        evaluator = BenchmarkEvaluator(
            name="finance-benchmark",
            retriever_fn=my_retrieval_function
        )

        report = evaluator.run(queries, judgments)
        evaluator.save_report(report, "results.json")
        evaluator.print_summary(report)
    """

    def __init__(
        self,
        name: str,
        retriever_fn: Callable[[str], List[str]],
        k_values: List[int] = [5, 10, 20]
    ):
        """
        Initialize evaluator.

        Args:
            name: Benchmark name
            retriever_fn: Function that takes query text and returns list of doc IDs
            k_values: List of k values for metrics
        """
        self.name = name
        self.retriever_fn = retriever_fn
        self.k_values = k_values

    def run(
        self,
        queries: List[Query],
        judgments: List[RelevanceJudgment]
    ) -> BenchmarkReport:
        """
        Run benchmark evaluation.

        Args:
            queries: List of queries to evaluate
            judgments: List of relevance judgments

        Returns:
            Complete benchmark report
        """
        # Build judgment map: query_id -> {doc_id: relevance}
        judgment_map = self._build_judgment_map(judgments)

        # Run retrieval for each query
        results = []
        latencies = []

        print(f"\nRunning {self.name} benchmark...")
        print(f"Evaluating {len(queries)} queries\n")

        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Processing query: {query.text[:60]}...")

            # Retrieve documents
            start_time = time.time()
            retrieved_docs = self.retriever_fn(query.text)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

            # Get relevance judgments for this query
            query_judgments = judgment_map.get(query.id, {})
            relevant_docs = {doc_id for doc_id, rel in query_judgments.items() if rel > 0}

            # Convert judgments to relevance scores for retrieved docs
            relevance_scores = [
                query_judgments.get(doc_id, 0)
                for doc_id in retrieved_docs
            ]

            # Compute metrics
            metrics = compute_all_metrics(
                relevant_docs=relevant_docs,
                retrieved_docs=retrieved_docs,
                relevance_scores=relevance_scores,
                k_values=self.k_values
            )

            # Store result
            result = BenchmarkResult(
                query_id=query.id,
                query_text=query.text,
                retrieved_docs=retrieved_docs[:max(self.k_values)],  # Only keep top-k
                relevance_scores=relevance_scores[:max(self.k_values)],
                latency_ms=latency_ms,
                metrics=metrics
            )
            results.append(result)

        # Aggregate metrics across all queries
        aggregate_metrics = self._aggregate_metrics(results)

        # Compute latency stats
        latency_stats = {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies)
        }

        return BenchmarkReport(
            benchmark_name=self.name,
            num_queries=len(queries),
            results=results,
            aggregate_metrics=aggregate_metrics,
            latency_stats=latency_stats,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

    def _build_judgment_map(
        self,
        judgments: List[RelevanceJudgment]
    ) -> Dict[str, Dict[str, int]]:
        """Build map from query_id to {doc_id: relevance}."""
        judgment_map = {}
        for judgment in judgments:
            if judgment.query_id not in judgment_map:
                judgment_map[judgment.query_id] = {}
            judgment_map[judgment.query_id][judgment.doc_id] = judgment.relevance
        return judgment_map

    def _aggregate_metrics(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """Aggregate metrics across all queries."""
        if not results:
            return {}

        # Get all metric names from first result
        metric_names = list(results[0].metrics.keys())

        # Average each metric across all queries
        aggregate = {}
        for metric_name in metric_names:
            values = [result.metrics[metric_name] for result in results]
            aggregate[metric_name] = np.mean(values)

        return aggregate

    def print_summary(self, report: BenchmarkReport):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print(f"{report.benchmark_name.upper()} - BENCHMARK RESULTS")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp}")
        print(f"Queries Evaluated: {report.num_queries}")
        print("=" * 80)

        # Print aggregate metrics
        print_metrics(report.aggregate_metrics, "Aggregate Retrieval Metrics")

        # Print latency stats
        print("Latency Statistics:")
        print("-" * 80)
        for stat, value in report.latency_stats.items():
            print(f"  {stat:20s} {value:8.2f} ms")
        print("=" * 80 + "\n")

    def save_report(self, report: BenchmarkReport, output_path: str):
        """Save benchmark report to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        report_dict = asdict(report)

        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)

        print(f"\nBenchmark report saved to: {output_path}")

    @staticmethod
    def load_queries(file_path: str) -> List[Query]:
        """Load queries from JSON or JSONL file."""
        queries = []
        with open(file_path, 'r') as f:
            # Try JSONL first
            content = f.read()
            f.seek(0)

            if content.strip().startswith('['):
                # JSON array
                data = json.load(f)
                queries = [Query(**item) for item in data]
            else:
                # JSONL
                for line in f:
                    if line.strip():
                        queries.append(Query(**json.loads(line)))

        return queries

    @staticmethod
    def load_judgments(file_path: str) -> List[RelevanceJudgment]:
        """Load relevance judgments from JSON or JSONL file."""
        judgments = []
        with open(file_path, 'r') as f:
            # Try JSONL first
            content = f.read()
            f.seek(0)

            if content.strip().startswith('['):
                # JSON array
                data = json.load(f)
                judgments = [RelevanceJudgment(**item) for item in data]
            else:
                # JSONL
                for line in f:
                    if line.strip():
                        judgments.append(RelevanceJudgment(**json.loads(line)))

        return judgments
