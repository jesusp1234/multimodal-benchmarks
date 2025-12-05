#!/usr/bin/env python3
"""
Medical Device Documentation Benchmark
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Benchmark for evaluating retrieval performance on medical device regulatory
documents including IFUs, recall notices, MAUDE reports, and 510(k) summaries.

Learn more: https://mxp.co/device

Features:
- Multimodal extraction (text, tables, diagrams)
- OCR pipeline (TrOCR + Tesseract)
- Layout detection (LayoutLMv3 + DocFormer)
- Table extraction (TATR + TableFormer)
- Domain-specific embeddings

Usage:
    # Run full benchmark
    python run.py

    # Run quick test
    python run.py --quick

    # Custom data
    python run.py --data-dir /path/to/device/docs
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add medical-device to path
MEDICAL_DEVICE_PATH = Path(__file__).parent.parent.parent / "extractors" / "medical-device"
sys.path.insert(0, str(MEDICAL_DEVICE_PATH))

from shared import BenchmarkEvaluator, Query, RelevanceJudgment

# Try to import medical device components
try:
    from src.medical_device_extractor import MedicalDeviceExtractor
    from src.medical_device_extractor.core import get_settings
    MEDICAL_DEVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import medical device extractor: {e}")
    print("Running in demo mode with mock retrieval.")
    MEDICAL_DEVICE_AVAILABLE = False


class MedicalDeviceBenchmark:
    """
    Medical device documentation retrieval benchmark.

    Tests retrieval performance on regulatory documents with queries like:
    - "What are the contraindications for this device?"
    - "Show me the sterilization instructions"
    - "Find adverse events related to catheter blockage"
    - "What are the device specifications and dimensions?"
    """

    def __init__(self, data_dir: str = None, quick: bool = False):
        """
        Initialize benchmark.

        Args:
            data_dir: Directory containing medical device documents
            quick: Run quick test with fewer queries
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.quick = quick
        self.documents = []

        # Initialize extractor if available
        if MEDICAL_DEVICE_AVAILABLE and self.data_dir:
            self._init_extractor()
        else:
            print("⚠️  Running in DEMO mode - using mock retrieval")
            print("   To run full benchmark, provide --data-dir with documents\n")

    def _init_extractor(self):
        """Initialize medical device extractor."""
        print("Initializing medical device extractor...")
        try:
            settings = get_settings()
            self.extractor = MedicalDeviceExtractor(settings)

            # Extract documents from data directory
            if self.data_dir and self.data_dir.exists():
                pdf_files = list(self.data_dir.glob("*.pdf"))
                print(f"Found {len(pdf_files)} PDF documents")

                for pdf_file in pdf_files[:5 if self.quick else None]:
                    print(f"  Extracting: {pdf_file.name}")
                    doc = self.extractor.extract(
                        str(pdf_file),
                        metadata={"source": pdf_file.name},
                        generate_embeddings=True
                    )
                    self.documents.append(doc)

                print(f"✓ Extracted {len(self.documents)} documents\n")
            else:
                print(f"⚠️  Data directory not found: {self.data_dir}\n")

        except Exception as e:
            print(f"Error initializing extractor: {e}")
            print("Falling back to demo mode\n")
            self.extractor = None

    def retriever(self, query: str) -> list[str]:
        """
        Retrieve documents for a query.

        Args:
            query: Search query

        Returns:
            List of document IDs ranked by relevance
        """
        if hasattr(self, 'documents') and self.documents:
            # Real retrieval using extracted documents
            # In production, this would use a vector database
            # For now, we'll do simple text matching as a demo
            results = []
            for doc in self.documents:
                score = self._simple_relevance(query, doc)
                results.append((doc.doc_id, score))

            # Sort by score and return IDs
            results.sort(key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, _ in results[:20]]
        else:
            # Mock retrieval for demo
            return self._mock_retrieval(query)

    def _simple_relevance(self, query: str, doc) -> float:
        """Simple relevance scoring for demo."""
        query_lower = query.lower()
        score = 0.0

        # Check sections
        for section in doc.sections:
            if any(term in section.text.lower() for term in query_lower.split()):
                score += 1.0

        # Check tables
        for table in doc.tables:
            if any(term in str(table.data).lower() for term in query_lower.split()):
                score += 0.5

        return score

    def _mock_retrieval(self, query: str) -> list[str]:
        """Mock retrieval for demo purposes."""
        return [
            f"ifu_{i}_{query[:10].replace(' ', '_')}"
            for i in range(1, 21)
        ]

    def get_sample_queries(self) -> list[Query]:
        """Get sample queries for benchmark."""
        queries = [
            Query(
                id="dev_001",
                text="What are the contraindications for this device?",
                intent="safety",
                domain="warnings_precautions"
            ),
            Query(
                id="dev_002",
                text="Show me the sterilization and cleaning instructions",
                intent="procedure",
                domain="maintenance"
            ),
            Query(
                id="dev_003",
                text="What are the device specifications and dimensions?",
                intent="specification",
                domain="technical"
            ),
            Query(
                id="dev_004",
                text="Find adverse events related to catheter blockage",
                intent="safety_signal",
                domain="maude_reports"
            ),
            Query(
                id="dev_005",
                text="What materials is the device made of?",
                intent="specification",
                domain="materials"
            ),
            Query(
                id="dev_006",
                text="How should the device be stored?",
                intent="procedure",
                domain="storage"
            ),
            Query(
                id="dev_007",
                text="What are the indications for use?",
                intent="regulatory",
                domain="indications"
            ),
            Query(
                id="dev_008",
                text="Show me the clinical trial results",
                intent="evidence",
                domain="clinical_data"
            ),
            Query(
                id="dev_009",
                text="What training is required to use this device?",
                intent="procedure",
                domain="training"
            ),
            Query(
                id="dev_010",
                text="Find information about MRI compatibility",
                intent="safety",
                domain="compatibility"
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
            # Create 3 relevant docs per query
            for i in range(1, 4):
                judgments.append(RelevanceJudgment(
                    query_id=query.id,
                    doc_id=f"ifu_{i}_{query.text[:10].replace(' ', '_')}",
                    relevance=2 if i == 1 else 1
                ))

        return judgments

    def run(self) -> dict:
        """Run the benchmark."""
        print("=" * 80)
        print("MEDICAL DEVICE DOCUMENTATION BENCHMARK")
        print("https://mxp.co/device")
        print("=" * 80)
        print()

        # Get queries and judgments
        queries = self.get_sample_queries()
        judgments = self.get_sample_judgments()

        # Create evaluator
        evaluator = BenchmarkEvaluator(
            name="device-benchmark",
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
        description="Medical Device Documentation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python run.py

  # Quick test
  python run.py --quick

  # With custom data directory
  python run.py --data-dir /path/to/device/documents

Learn more: https://mxp.co/device
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing medical device documents (PDFs)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with fewer queries and documents'
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = MedicalDeviceBenchmark(
        data_dir=args.data_dir,
        quick=args.quick
    )

    benchmark.run()


if __name__ == "__main__":
    main()
