#!/usr/bin/env python3
"""
Curriculum Search Benchmark
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Benchmark for evaluating retrieval performance on educational video content
including lecture transcripts, slides, code examples, and visual demonstrations.

Learn more: https://mxp.co/learning

Features:
- Multi-modal content (video, slides, code)
- Whisper ASR with word-level timestamps
- PySceneDetect scene segmentation
- Code extraction and analysis
- HyDE (Hypothetical Document Embeddings)
- Multi-vector fusion (RRF)

Usage:
    # Run full benchmark
    python run.py

    # Run quick test
    python run.py --quick

    # Custom data
    python run.py --data-dir /path/to/course/content
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add curriculum to path
CURRICULUM_PATH = Path(__file__).parent.parent.parent / "extractors" / "curriculum"
sys.path.insert(0, str(CURRICULUM_PATH))

from shared import BenchmarkEvaluator, Query, RelevanceJudgment

# Try to import curriculum components
try:
    from main import CurriculumPipeline
    from models import Query as CurriculumQuery
    CURRICULUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import curriculum pipeline: {e}")
    print("Running in demo mode with mock retrieval.")
    CURRICULUM_AVAILABLE = False


class CurriculumBenchmark:
    """
    Curriculum search benchmark.

    Tests retrieval performance on educational content with queries like:
    - "How do pointers work in C?"
    - "Show me examples of memory allocation"
    - "What is the difference between stack and heap?"
    - "Explain recursion with code examples"
    """

    def __init__(self, data_dir: str = None, quick: bool = False):
        """
        Initialize benchmark.

        Args:
            data_dir: Directory containing course content (video, slides, code)
            quick: Run quick test with fewer queries
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.quick = quick
        self.pipeline = None

        # Initialize pipeline if available
        if CURRICULUM_AVAILABLE:
            self._init_pipeline()
        else:
            print("⚠️  Running in DEMO mode - using mock retrieval")
            print("   To run full benchmark, ensure curriculum extractor is available\n")

    def _init_pipeline(self):
        """Initialize curriculum pipeline."""
        print("Initializing curriculum pipeline...")
        try:
            self.pipeline = CurriculumPipeline()

            # If data directory provided, process content
            if self.data_dir and self.data_dir.exists():
                print(f"Processing content from: {self.data_dir}")

                # Look for video, slides, and code
                video_path = None
                slides_path = None
                code_path = None

                for file in self.data_dir.glob("*"):
                    if file.suffix in ['.mp4', '.avi', '.mov']:
                        video_path = file
                    elif file.suffix == '.pdf':
                        slides_path = file
                    elif file.suffix == '.zip' or file.name == 'code':
                        code_path = file

                if video_path or slides_path or code_path:
                    print(f"  Video: {video_path.name if video_path else 'None'}")
                    print(f"  Slides: {slides_path.name if slides_path else 'None'}")
                    print(f"  Code: {code_path.name if code_path else 'None'}")

                    # Process content
                    self.segments = self.pipeline.process_course_content(
                        course_id="benchmark-course",
                        course_title="Benchmark Course",
                        video_path=str(video_path) if video_path else None,
                        slides_path=str(slides_path) if slides_path else None,
                        code_path=str(code_path) if code_path else None
                    )
                    print(f"✓ Processed {len(self.segments)} content segments\n")
                else:
                    print("⚠️  No valid content found in data directory\n")
                    self.segments = []
            else:
                self.segments = []
                print("⚠️  No data directory provided\n")

        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            print("Falling back to demo mode\n")
            self.pipeline = None
            self.segments = []

    def retriever(self, query: str) -> list[str]:
        """
        Retrieve content segments for a query.

        Args:
            query: Search query

        Returns:
            List of segment IDs ranked by relevance
        """
        if self.pipeline and hasattr(self, 'segments') and self.segments:
            # Real retrieval using curriculum pipeline
            try:
                curriculum_query = CurriculumQuery(
                    query_text=query,
                    intent="general",
                    domain="programming"
                )

                results = self.pipeline.retrieval_engine.retrieve(
                    curriculum_query,
                    k=20
                )

                return [result['segment_id'] for result in results]
            except Exception as e:
                print(f"Warning: Retrieval error: {e}")
                return self._mock_retrieval(query)
        else:
            # Mock retrieval for demo
            return self._mock_retrieval(query)

    def _mock_retrieval(self, query: str) -> list[str]:
        """Mock retrieval for demo purposes."""
        return [
            f"segment_{i}_{query[:10].replace(' ', '_')}"
            for i in range(1, 21)
        ]

    def get_sample_queries(self) -> list[Query]:
        """Get sample queries for benchmark."""
        queries = [
            Query(
                id="learn_001",
                text="How do pointers work in C?",
                intent="concept_explanation",
                domain="systems_programming"
            ),
            Query(
                id="learn_002",
                text="Show me examples of memory allocation with malloc",
                intent="code_example",
                domain="systems_programming"
            ),
            Query(
                id="learn_003",
                text="What is the difference between stack and heap memory?",
                intent="comparison",
                domain="memory_management"
            ),
            Query(
                id="learn_004",
                text="Explain how recursion works with code examples",
                intent="concept_with_code",
                domain="algorithms"
            ),
            Query(
                id="learn_005",
                text="How do I prevent memory leaks?",
                intent="troubleshooting",
                domain="debugging"
            ),
            Query(
                id="learn_006",
                text="What is pointer arithmetic?",
                intent="concept_explanation",
                domain="systems_programming"
            ),
            Query(
                id="learn_007",
                text="Show me how to use structs in C",
                intent="code_example",
                domain="data_structures"
            ),
            Query(
                id="learn_008",
                text="What are common segmentation fault causes?",
                intent="troubleshooting",
                domain="debugging"
            ),
            Query(
                id="learn_009",
                text="Explain the difference between malloc and calloc",
                intent="comparison",
                domain="memory_management"
            ),
            Query(
                id="learn_010",
                text="How do I debug memory issues with valgrind?",
                intent="tool_usage",
                domain="debugging"
            ),
        ]

        if self.quick:
            return queries[:3]

        return queries

    def get_sample_judgments(self) -> list[RelevanceJudgment]:
        """Get sample relevance judgments."""
        # In a real benchmark, these would be human-annotated
        judgments = []

        for query in self.get_sample_queries():
            # Create 3 relevant segments per query
            for i in range(1, 4):
                judgments.append(RelevanceJudgment(
                    query_id=query.id,
                    doc_id=f"segment_{i}_{query.text[:10].replace(' ', '_')}",
                    relevance=2 if i == 1 else 1
                ))

        return judgments

    def run(self) -> dict:
        """Run the benchmark."""
        print("=" * 80)
        print("CURRICULUM SEARCH BENCHMARK")
        print("https://mxp.co/learning")
        print("=" * 80)
        print()

        # Get queries and judgments
        queries = self.get_sample_queries()
        judgments = self.get_sample_judgments()

        # Create evaluator
        evaluator = BenchmarkEvaluator(
            name="learning-benchmark",
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
        description="Curriculum Search Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python run.py

  # Quick test
  python run.py --quick

  # With custom course content
  python run.py --data-dir /path/to/course/content

Learn more: https://mxp.co/learning
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing course content (video, slides, code)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with fewer queries'
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = CurriculumBenchmark(
        data_dir=args.data_dir,
        quick=args.quick
    )

    benchmark.run()


if __name__ == "__main__":
    main()
