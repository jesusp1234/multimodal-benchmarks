# Financial Document Retrieval Benchmark

**Learn more: https://mxp.co/finance**

Benchmark for evaluating retrieval performance on financial documents including SEC filings (10-K, 10-Q, 8-K), earnings reports, and investor presentations.

## Features

### Advanced Retrieval
- **Multi-vector embeddings** (7 named vectors per chunk)
  - Title, summary, full_text, propositions, contextual, visual, financial
- **Hybrid search** combining vector and keyword (BM25)
- **XBRL fact extraction** for structured financial data
- **Table-aware retrieval** with TableFormer
- **Reciprocal Rank Fusion** for result combination

### Query Types
- **Fact extraction**: "What was the revenue in Q4 2023?"
- **Calculations**: "What is the YoY revenue growth rate?"
- **Comparisons**: "Compare gross margins between quarters"
- **Table lookup**: "Show revenue breakdown by segment"
- **Risk analysis**: "What are the main risk factors?"
- **Topic search**: "Find supply chain discussions"

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/ethan/Dev/mixpeek/benchmarks/finance
pip install -r ../shared/requirements.txt
```

### 2. Run Benchmark

```bash
# Run with demo data
python run.py

# Run quick test (3 queries)
python run.py --quick

# Run with your own financial documents
python run.py --data-dir /path/to/financial/documents
```

### 3. View Results

Results are saved to `results/benchmark_results.json`

```bash
cat results/benchmark_results.json | jq '.aggregate_metrics'
```

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `ndcg@10` | Ranking quality (primary metric) | >0.75 |
| `recall@5` | Coverage in top 5 results | >0.70 |
| `mrr` | Mean reciprocal rank | >0.65 |
| `precision@10` | Accuracy in top 10 | >0.60 |
| `latency_p95` | 95th percentile latency | <2s |

## Architecture

```
Financial Document → PDF Extraction → Chunking → Multi-Vector Embedding → Qdrant
                     XBRL Parsing     (Semantic)  (7 vectors)
                     Table Extraction

Query → Query Embedding → Hybrid Search → Reranking → Results
        (7 vectors)      (Vector + BM25)  (RRF)
```

## Sample Queries

The benchmark includes queries across different intents:

1. **Fact Extraction**
   - "What was the total revenue for fiscal year 2023?"
   - "What acquisitions were made in the last fiscal year?"

2. **Calculations**
   - "Show me the year-over-year revenue growth rate"
   - "Compare cash flow from operations vs investing activities"

3. **Table Lookup**
   - "Show me the breakdown of revenue by geographic segment"
   - "Find details on stock-based compensation expense"

4. **Summarization**
   - "What are the main risk factors disclosed in the latest 10-K?"
   - "What were the key highlights from the earnings call?"

5. **Topic Search**
   - "What did the company say about supply chain challenges?"
   - "Find information about EBITDA margins and operating expenses"

## Data Format

### Documents
Place financial documents in your data directory:
```
data/
  ├── company_10k_2023.pdf
  ├── company_10q_q3_2024.pdf
  ├── earnings_call_q4.pdf
  └── investor_presentation.pdf
```

### Queries
Queries are defined in code but can be loaded from JSON:
```json
{
  "id": "fin_001",
  "text": "What was the total revenue for fiscal year 2023?",
  "intent": "fact_extraction",
  "domain": "financial_metrics"
}
```

### Relevance Judgments
Ground truth judgments (0-3 scale):
```json
{
  "query_id": "fin_001",
  "doc_id": "doc_revenue_table_fy2023",
  "relevance": 3
}
```

Relevance scale:
- **0**: Not relevant
- **1**: Somewhat relevant
- **2**: Highly relevant
- **3**: Perfect match

## Integration with Financial Document System

This benchmark integrates with the financial document system at:
`/Users/ethan/Dev/mixpeek/customers/financial-document`

The system provides:
- PDF extraction with PyMuPDF
- XBRL parsing with Arelle
- Table extraction
- Multi-vector embeddings
- Qdrant vector database
- Hybrid search with RRF

## Extending the Benchmark

### Add Custom Queries

Edit `run.py` and add to `get_sample_queries()`:

```python
Query(
    id="fin_custom_001",
    text="Your custom query here",
    intent="fact_extraction",  # or calculation, comparison, etc.
    domain="financial_metrics"
)
```

### Add Ground Truth Judgments

Add to `get_sample_judgments()`:

```python
RelevanceJudgment(
    query_id="fin_custom_001",
    doc_id="your_doc_id",
    relevance=3  # 0-3 scale
)
```

### Use Real Documents

```bash
# Index your documents first using the financial document system
cd /Users/ethan/Dev/mixpeek/customers/financial-document

# Ingest documents
curl -X POST "http://localhost:8000/v1/documents/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "company_10k.pdf",
    "doc_type": "10-K",
    "metadata": {"company_ticker": "AAPL"}
  }'

# Then run benchmark
cd /Users/ethan/Dev/mixpeek/benchmarks/finance
python run.py --data-dir /path/to/documents
```

## Performance Targets

Based on our analysis of SOTA systems:

| System | NDCG@10 | Recall@5 | Notes |
|--------|---------|----------|-------|
| BM25 Baseline | 0.38 | 0.32 | Keyword only |
| Dense Retrieval (BGE) | 0.61 | 0.58 | Single vector |
| **Mixpeek Multi-Vector** | **0.74** | **0.71** | 7 named vectors + hybrid |
| + Cross-encoder | 0.78 | 0.75 | With reranking |

## Citation

```bibtex
@misc{mixpeek-financial-benchmark,
  title={Financial Document Retrieval Benchmark},
  author={Mixpeek},
  year={2025},
  url={https://mxp.co/finance}
}
```

## Learn More

- **Full Documentation**: https://mxp.co/finance
- **Source Code**: `/Users/ethan/Dev/mixpeek/customers/financial-document`
- **Benchmark Suite**: https://github.com/mixpeek/benchmarks

---

Built by [Mixpeek](https://mixpeek.com) — Multimodal AI for regulated industries.
