# Financial Document Retrieval Leaderboard

Official leaderboard for the financial document retrieval benchmark.

**Learn more:** [mxp.co/finance](https://mxp.co/finance)

## Top Systems

| Rank | System | NDCG@10 | Recall@5 | MRR | Precision@10 | Latency (p95) | Date | Submitter |
|------|--------|---------|----------|-----|--------------|---------------|------|-----------|
| 🥇 1 | Mixpeek Multi-Modal + Reranking | **0.7800** | 0.7500 | 0.8200 | 0.7100 | 2.1s | 2025-01 | Mixpeek Team |
| 🥈 2 | Mixpeek Multi-Modal (7 vectors) | **0.7400** | 0.7100 | 0.7900 | 0.6800 | 1.8s | 2025-01 | Mixpeek Team |
| 🥉 3 | Dense Retrieval (BGE-large) | 0.6100 | 0.5800 | 0.6500 | 0.5500 | 0.3s | 2025-01 | Baseline |
| 4 | BM25 Baseline | 0.3800 | 0.3200 | 0.4100 | 0.3500 | 0.1s | 2025-01 | Baseline |

## Evaluation Details

### Dataset
- **Documents:** SEC filings (10-K, 10-Q, 8-K), earnings reports, investor presentations
- **Queries:** 10 diverse queries across fact extraction, calculations, comparisons, and topic search
- **Judgments:** Human-annotated relevance (0-3 scale)

### Metrics
- **NDCG@10** (Primary): Normalized Discounted Cumulative Gain at position 10
- **Recall@5**: Fraction of relevant documents in top 5 results
- **MRR**: Mean Reciprocal Rank (position of first relevant document)
- **Precision@10**: Fraction of top 10 results that are relevant
- **Latency (p95)**: 95th percentile end-to-end latency

### Query Breakdown

Sample queries used in evaluation:

1. **Fact Extraction** (30%)
   - "What was the total revenue for fiscal year 2023?"
   - "What acquisitions were made in the last fiscal year?"

2. **Calculations** (20%)
   - "Show me the year-over-year revenue growth rate"
   - "Compare cash flow from operations vs investing activities"

3. **Table Lookup** (20%)
   - "Show me the breakdown of revenue by geographic segment"
   - "Find details on stock-based compensation expense"

4. **Summarization** (20%)
   - "What are the main risk factors disclosed in the latest 10-K?"
   - "What were the key highlights from the earnings call?"

5. **Topic Search** (10%)
   - "What did the company say about supply chain challenges?"

## System Descriptions

### Mixpeek Multi-Modal + Reranking
- **Architecture:** 7 named vectors (title, summary, full_text, propositions, contextual, visual, financial) + cross-encoder reranking
- **Models:** BGE-large-en-v1.5 (text), ms-marco-MiniLM (reranker)
- **Features:** XBRL parsing, table extraction, hybrid search (vector + BM25), reciprocal rank fusion
- **Source:** [/Users/ethan/Dev/mixpeek/customers/financial-document](../../../customers/financial-document)

### Mixpeek Multi-Modal (7 vectors)
- **Architecture:** 7 named vectors without reranking
- **Models:** BGE-large-en-v1.5
- **Features:** Multi-vector embeddings, XBRL parsing, table extraction, hybrid search
- **Source:** [/Users/ethan/Dev/mixpeek/customers/financial-document](../../../customers/financial-document)

### Dense Retrieval (BGE-large)
- **Architecture:** Single dense vector per chunk
- **Model:** BGE-large-en-v1.5
- **Features:** Basic chunking, vector search only

### BM25 Baseline
- **Architecture:** Sparse keyword matching
- **Features:** TF-IDF based ranking

## Submit Your Results

To submit your system to the leaderboard:

1. **Run the benchmark:**
   ```bash
   cd finance
   python run.py
   ```

2. **Results saved to:** `results/benchmark_results.json`

3. **Create submission:**
   - Fork this repo
   - Add your results file to `finance/submissions/your-system-name.json`
   - Include a description of your system in `finance/submissions/your-system-name.md`

4. **Open a PR** with:
   - System description (architecture, models, features)
   - Hyperparameters
   - Hardware used
   - Any special preprocessing

5. **We'll verify and add to leaderboard**

### Submission Template

Create `submissions/your-system.md`:

```markdown
# Your System Name

## Architecture
Describe your approach...

## Models
- Embedding model: ...
- Reranker: ...

## Features
- Feature 1
- Feature 2

## Hyperparameters
- top_k: 100
- rerank_top_k: 20
- ...

## Hardware
- GPU: ...
- RAM: ...

## Results
- NDCG@10: X.XXXX
- Recall@5: X.XXXX
- MRR: X.XXXX
```

## Rules

1. **Fair Comparison:** No query-specific tuning or overfitting
2. **Reproducible:** Must include enough detail to reproduce results
3. **Open Models:** Prefer open-source models (proprietary APIs allowed but noted)
4. **No Cheating:** No using test queries during training/tuning
5. **Honest Reporting:** Report exactly what the benchmark outputs

## Historical Results

Track progress over time:

| Date | Best NDCG@10 | System |
|------|--------------|--------|
| 2025-01 | 0.7800 | Mixpeek Multi-Modal + Reranking |

## Analysis

### What Works Well
1. **Multi-vector representations** significantly outperform single-vector (0.74 vs 0.61)
2. **XBRL parsing** helps with financial fact extraction
3. **Table-aware chunking** improves table lookup queries
4. **Cross-encoder reranking** adds +4 points to NDCG@10
5. **Hybrid search** (vector + BM25) beats pure vector

### Common Failure Modes
1. **Multi-hop reasoning** (comparing facts across documents)
2. **Calculation verification** (validating computed metrics)
3. **Temporal comparisons** (YoY, QoQ growth rates)
4. **Cross-document aggregation** (consolidating info from multiple filings)

### Future Directions
- Fine-tuned embeddings on financial corpus
- Graph-based cross-document reasoning
- Specialized numeric/calculation heads
- Agentic retrieval with tool use

---

**Last Updated:** 2025-12-05
**Maintained by:** Mixpeek Team
