#!/usr/bin/env python3
"""
Financial Document Retrieval Benchmark
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Benchmark for evaluating retrieval performance on financial documents
including SEC filings, earnings reports, and investor presentations.

Learn more: https://mxp.co/finance

Features:
- Multi-vector embeddings (7 named vectors)
- Hybrid search (vector + keyword)
- XBRL fact extraction
- Table-aware retrieval
- Financial reasoning

Usage:
    # Run full benchmark
    python run.py

    # Run quick test
    python run.py --quick

    # Custom data
    python run.py --data-dir /path/to/data
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add financial-document to path
FINANCIAL_DOC_PATH = Path(__file__).parent.parent.parent / "customers" / "financial-document"
sys.path.insert(0, str(FINANCIAL_DOC_PATH))

from shared import BenchmarkEvaluator, Query, RelevanceJudgment

# Try to import financial document components
try:
    from services.search_service import SearchService
    from services.qdrant_service import QdrantService
    from services.embedding_service import EmbeddingService
    FINANCIAL_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import financial document services: {e}")
    print("Running in demo mode with mock retrieval.")
    FINANCIAL_SERVICES_AVAILABLE = False


class FinancialBenchmark:
    """
    Financial document retrieval benchmark.

    Tests retrieval performance on financial documents with queries like:
    - "What was the revenue growth in Q4 2023?"
    - "Show me EBITDA margins from the latest 10-K"
    - "Find risk factors related to supply chain"
    """

    def __init__(self, data_dir: str = None, quick: bool = False):
        """
        Initialize benchmark.

        Args:
            data_dir: Directory containing financial documents
            quick: Run quick test with fewer queries
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.quick = quick

        # Initialize services if available
        if FINANCIAL_SERVICES_AVAILABLE and self.data_dir:
            self._init_services()
        else:
            print("⚠️  Running in DEMO mode - using mock retrieval")
            print("   To run full benchmark, provide --data-dir with documents\n")

    def _init_services(self):
        """Initialize financial document services."""
        print("Initializing financial document services...")
        try:
            self.qdrant_service = QdrantService()
            self.embedding_service = EmbeddingService()
            self.search_service = SearchService(
                self.qdrant_service,
                self.embedding_service
            )
            print("✓ Services initialized\n")
        except Exception as e:
            print(f"Error initializing services: {e}")
            print("Falling back to demo mode\n")
            self.search_service = None

    def retriever(self, query: str) -> list[str]:
        """
        Retrieve documents for a query.

        Args:
            query: Search query

        Returns:
            List of document IDs ranked by relevance
        """
        if hasattr(self, 'search_service') and self.search_service:
            # Real retrieval using financial document services
            results = self.search_service.search(
                query=query,
                top_k=20,
                search_type="hybrid"
            )
            return [result['chunk_id'] for result in results]
        else:
            # Mock retrieval for demo
            return self._mock_retrieval(query)

    def _mock_retrieval(self, query: str) -> list[str]:
        """Mock retrieval for demo purposes."""
        # Simple mock - in reality this would use the actual search service
        return [
            f"doc_{i}_{query[:10].replace(' ', '_')}"
            for i in range(1, 21)
        ]

    def get_sample_queries(self) -> list[Query]:
        """Get sample queries for benchmark."""
        queries = [
            Query(
                id="fin_001",
                text="What was the total revenue for fiscal year 2023?",
                intent="fact_extraction",
                domain="financial_metrics"
            ),
            Query(
                id="fin_002",
                text="Show me the year-over-year revenue growth rate",
                intent="calculation",
                domain="financial_metrics"
            ),
            Query(
                id="fin_003",
                text="What are the main risk factors disclosed in the latest 10-K?",
                intent="summarization",
                domain="risk_disclosure"
            ),
            Query(
                id="fin_004",
                text="Find information about EBITDA margins and operating expenses",
                intent="multi_fact",
                domain="financial_metrics"
            ),
            Query(
                id="fin_005",
                text="What did the company say about supply chain challenges?",
                intent="topic_search",
                domain="operations"
            ),
            Query(
                id="fin_006",
                text="Compare cash flow from operations vs investing activities",
                intent="comparison",
                domain="cash_flow"
            ),
            Query(
                id="fin_007",
                text="What acquisitions were made in the last fiscal year?",
                intent="fact_extraction",
                domain="corporate_actions"
            ),
            Query(
                id="fin_008",
                text="Show me the breakdown of revenue by geographic segment",
                intent="table_lookup",
                domain="financial_metrics"
            ),
            Query(
                id="fin_009",
                text="What were the key highlights from the earnings call?",
                intent="summarization",
                domain="earnings"
            ),
            Query(
                id="fin_010",
                text="Find details on stock-based compensation expense",
                intent="fact_extraction",
                domain="financial_metrics"
            ),
        ]

        if self.quick:
            return queries[:3]

        return queries

    def get_sample_judgments(self) -> list[RelevanceJudgment]:
        """Get sample relevance judgments."""
        # In a real benchmark, these would be human-annotated
        # For demo, we'll create placeholder judgments
        judgments = []

        for query in self.get_sample_queries():
            # Create 3 relevant and 2 highly relevant docs per query
            for i in range(1, 4):
                judgments.append(RelevanceJudgment(
                    query_id=query.id,
                    doc_id=f"doc_{i}_{query.text[:10].replace(' ', '_')}",
                    relevance=1 if i <= 3 else 2
                ))

        return judgments

    def run(self) -> dict:
        """Run the benchmark."""
        print("=" * 80)
        print("FINANCIAL DOCUMENT RETRIEVAL BENCHMARK")
        print("https://mxp.co/finance")
        print("=" * 80)
        print()

        # Get queries and judgments
        queries = self.get_sample_queries()
        judgments = self.get_sample_judgments()

        # Create evaluator
        evaluator = BenchmarkEvaluator(
            name="finance-benchmark",
            retriever_fn=self.retriever,
            k_values=[5, 10, 20]
        )

        # Run evaluation
        report = evaluator.run(queries, judgments)

        # Print summary
        evaluator.print_summary(report)

        # Save report
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "benchmark_results.json"
        evaluator.save_report(report, str(output_file))

        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Financial Document Retrieval Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python run.py

  # Quick test
  python run.py --quick

  # With custom data directory
  python run.py --data-dir /path/to/financial/documents

Learn more: https://mxp.co/finance
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing financial documents to index'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with fewer queries'
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = FinancialBenchmark(
        data_dir=args.data_dir,
        quick=args.quick
    )

    benchmark.run()


if __name__ == "__main__":
    main()
