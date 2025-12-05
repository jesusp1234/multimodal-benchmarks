# Quick Start Guide

Get started with Mixpeek benchmarks in 60 seconds.

## 🎯 Choose Your Benchmark

| Benchmark | What It Tests | CTA |
|-----------|---------------|-----|
| **Finance** | Retrieval on SEC filings, earnings reports | **[mxp.co/finance](https://mxp.co/finance)** |
| **Device** | Retrieval on medical device IFUs, regulatory docs | **[mxp.co/device](https://mxp.co/device)** |
| **Learning** | Retrieval on educational videos, lectures | **[mxp.co/learning](https://mxp.co/learning)** |

## ⚡ Run in 3 Commands

### Option 1: Quick Demo (No Data Required)

```bash
# Finance benchmark
cd finance && python run.py --quick

# Device benchmark
cd device && python run.py --quick

# Learning benchmark
cd learning && python run.py --quick
```

Each benchmark runs in **demo mode** with mock data and completes in ~1 second.

### Option 2: Full Benchmark (With Your Data)

```bash
# Finance: Point to your financial documents
cd finance
python run.py --data-dir /path/to/sec/filings

# Device: Point to your medical device PDFs
cd device
python run.py --data-dir /path/to/device/docs

# Learning: Point to your course content
cd learning
python run.py --data-dir /path/to/course/content
```

## 📊 Example Output

```
================================================================================
FINANCIAL DOCUMENT RETRIEVAL BENCHMARK
https://mxp.co/finance
================================================================================

Running finance-benchmark benchmark...
Evaluating 3 queries

[1/3] Processing query: What was the total revenue for fiscal year 2023?...
[2/3] Processing query: Show me the year-over-year revenue growth rate...
[3/3] Processing query: What are the main risk factors...

================================================================================
                          Aggregate Retrieval Metrics
================================================================================

Ranking Quality (NDCG):
  ndcg@10              0.7400
  ndcg@5               0.7200

Coverage (Recall):
  recall@10            0.7100
  recall@5             0.6800

Other Metrics:
  mrr                  0.7900

Latency Statistics:
  p95_ms               1800.00 ms

Benchmark report saved to: finance/results/benchmark_results.json
```

## 📁 Results Location

Each benchmark saves results to:
- `finance/results/benchmark_results.json`
- `device/results/benchmark_results.json`
- `learning/results/benchmark_results.json`

## 🏆 View Leaderboards

Compare your results against baselines:
- [Finance Leaderboard](finance/LEADERBOARD.md)
- [Device Leaderboard](device/LEADERBOARD.md)
- [Learning Leaderboard](learning/LEADERBOARD.md)

## 📖 Deep Dive

Each benchmark has detailed documentation:

### Finance Benchmark
- [Full README](finance/README.md)
- **Features:** Multi-vector embeddings, XBRL parsing, table extraction
- **Queries:** Revenue, risk factors, financial metrics
- **Target:** NDCG@10 > 0.75

### Device Benchmark
- [Full README](device/README.md)
- **Features:** OCR pipeline, table extraction, diagram captioning
- **Queries:** Contraindications, specifications, procedures
- **Target:** NDCG@10 > 0.72

### Learning Benchmark
- [Full README](learning/README.md)
- **Features:** Whisper ASR, scene detection, code analysis, HyDE
- **Queries:** Concepts, code examples, troubleshooting
- **Target:** NDCG@10 > 0.75, Recall@50 > 0.90

## 🛠️ Build Your Own Retriever

All benchmarks use a standard interface:

```python
from shared import BenchmarkEvaluator, Query, RelevanceJudgment

# 1. Define your retrieval function
def my_retriever(query: str) -> list[str]:
    """
    Your retrieval logic here.

    Args:
        query: Natural language query

    Returns:
        List of document IDs ranked by relevance
    """
    # Your code here
    return ["doc_1", "doc_2", "doc_3", ...]

# 2. Load benchmark data
queries = [
    Query(id="q1", text="What was the revenue?", intent="fact", domain="finance"),
    # ... more queries
]

judgments = [
    RelevanceJudgment(query_id="q1", doc_id="doc_1", relevance=3),
    # ... more judgments
]

# 3. Run evaluation
evaluator = BenchmarkEvaluator(
    name="my-system",
    retriever_fn=my_retriever,
    k_values=[5, 10, 20]
)

report = evaluator.run(queries, judgments)
evaluator.print_summary(report)
evaluator.save_report(report, "results.json")
```

## 📦 Dependencies

Minimal dependencies:
```bash
pip install numpy
```

For full functionality, each benchmark may require additional packages.
See individual READMEs for details.

## 🚀 Integration with Source Systems

Each benchmark integrates with real extraction/retrieval systems:

### Finance
- **Source:** `/Users/ethan/Dev/mixpeek/customers/financial-document`
- **Stack:** FastAPI, Qdrant, PyMuPDF, Arelle (XBRL)
- **Features:** 7 named vectors, hybrid search, reasoning

### Device
- **Source:** `/Users/ethan/Dev/mixpeek/extractors/medical-device`
- **Stack:** TrOCR, LayoutLMv3, TATR, TableFormer, ColPali
- **Features:** OCR pipeline, table extraction, diagram captioning

### Learning
- **Source:** `/Users/ethan/Dev/mixpeek/extractors/curriculum`
- **Stack:** Whisper, PySceneDetect, BGE-M3, SFR-Embedding-Code
- **Features:** ASR, scene detection, multi-vector fusion, HyDE

## 🤝 Submit Results

Beat the baseline? Submit your results:

1. Run benchmark: `python run.py`
2. Results in: `results/benchmark_results.json`
3. Fork repo and open PR
4. Include system description

Your results will appear on the leaderboard!

## 📚 Learn More

- **Main README:** [README.md](README.md)
- **Finance Details:** [finance/README.md](finance/README.md)
- **Device Details:** [device/README.md](device/README.md)
- **Learning Details:** [learning/README.md](learning/README.md)

---

**Built by [Mixpeek](https://mixpeek.com)** — Multimodal AI for regulated industries.
