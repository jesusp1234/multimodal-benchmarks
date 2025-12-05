# Medical Device Documentation Leaderboard

Official leaderboard for the medical device documentation retrieval benchmark.

**Learn more:** [mxp.co/device](https://mxp.co/device)

## Top Systems

| Rank | System | NDCG@10 | Recall@5 | MRR | Precision@10 | Latency (p95) | Date | Submitter |
|------|--------|---------|----------|-----|--------------|---------------|------|-----------|
| 🥇 1 | Mixpeek Multi-Modal + Reranking | **0.7800** | 0.7400 | 0.8100 | 0.7000 | 3.2s | 2025-01 | Mixpeek Team |
| 🥈 2 | Mixpeek Multi-Modal (Text+Table+Diagram) | **0.7200** | 0.6800 | 0.7500 | 0.6500 | 2.8s | 2025-01 | Mixpeek Team |
| 🥉 3 | Dense Retrieval (BGE-large) | 0.5800 | 0.5200 | 0.6200 | 0.5200 | 0.4s | 2025-01 | Baseline |
| 4 | BM25 Baseline | 0.4100 | 0.3500 | 0.4500 | 0.3800 | 0.1s | 2025-01 | Baseline |

## Evaluation Details

### Dataset
- **Documents:** IFUs (Instructions for Use), 510(k) summaries, recall notices, MAUDE reports, clinical trial data
- **Queries:** 10 diverse queries across safety, procedures, specifications, and regulatory domains
- **Judgments:** Human-annotated relevance (0-3 scale)

### Metrics
- **NDCG@10** (Primary): Normalized Discounted Cumulative Gain at position 10
- **Recall@5**: Fraction of relevant documents in top 5 results
- **MRR**: Mean Reciprocal Rank (position of first relevant document)
- **Precision@10**: Fraction of top 10 results that are relevant
- **Latency (p95)**: 95th percentile end-to-end latency

### Query Breakdown

Sample queries used in evaluation:

1. **Safety & Warnings** (30%)
   - "What are the contraindications for this device?"
   - "Find adverse events related to catheter blockage"
   - "Find information about MRI compatibility"

2. **Procedures** (30%)
   - "Show me the sterilization and cleaning instructions"
   - "How should the device be stored?"
   - "What training is required to use this device?"

3. **Specifications** (20%)
   - "What are the device specifications and dimensions?"
   - "What materials is the device made of?"

4. **Regulatory** (20%)
   - "What are the indications for use?"
   - "Show me the clinical trial results"

## System Descriptions

### Mixpeek Multi-Modal + Reranking
- **Architecture:** Multi-modal extraction (text + tables + diagrams) + cross-encoder reranking
- **Models:** TrOCR (OCR), LayoutLMv3 (layout), TATR + TableFormer (tables), ColPali + BLIP-2 (diagrams), BGE-reranker-v2-m3 (reranking)
- **Features:** Native PDF + OCR pipeline, table extraction with validation, diagram caption generation, bounding box tracking
- **Source:** [/Users/ethan/Dev/mixpeek/extractors/medical-device](../../../extractors/medical-device)

### Mixpeek Multi-Modal
- **Architecture:** Multi-modal extraction without reranking
- **Models:** TrOCR, LayoutLMv3, TATR, TableFormer, ColPali, BLIP-2
- **Features:** Text + table + diagram embeddings, smart chunking by content type
- **Source:** [/Users/ethan/Dev/mixpeek/extractors/medical-device](../../../extractors/medical-device)

### Dense Retrieval (BGE-large)
- **Architecture:** Single dense vector per chunk
- **Model:** BGE-large-en-v1.5
- **Features:** Text-only, basic chunking

### BM25 Baseline
- **Architecture:** Sparse keyword matching
- **Features:** TF-IDF based ranking

## Submit Your Results

To submit your system to the leaderboard:

1. **Run the benchmark:**
   ```bash
   cd device
   python run.py
   ```

2. **Results saved to:** `results/benchmark_results.json`

3. **Create submission:**
   - Fork this repo
   - Add your results file to `device/submissions/your-system-name.json`
   - Include a description of your system in `device/submissions/your-system-name.md`

4. **Open a PR** with:
   - System description (architecture, models, features)
   - Hyperparameters
   - Hardware used
   - OCR and extraction settings

5. **We'll verify and add to leaderboard**

### Submission Template

Create `submissions/your-system.md`:

```markdown
# Your System Name

## Architecture
Describe your approach...

## Models
- OCR: ...
- Layout detection: ...
- Table extraction: ...
- Embeddings: ...

## Features
- Feature 1
- Feature 2

## Hyperparameters
- OCR DPI: 300
- Chunk size: 512
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
4. **Document Quality:** Test on mix of native PDFs and scanned documents
5. **Honest Reporting:** Report exactly what the benchmark outputs

## Historical Results

Track progress over time:

| Date | Best NDCG@10 | System |
|------|--------------|--------|
| 2025-01 | 0.7800 | Mixpeek Multi-Modal + Reranking |

## Analysis

### What Works Well
1. **Multi-modal extraction** (text + tables + diagrams) significantly outperforms text-only
2. **OCR pipeline** (TrOCR + Tesseract) handles scanned documents well
3. **Table extraction** critical for specifications and technical data
4. **Layout detection** helps preserve document structure
5. **Cross-encoder reranking** adds +6 points to NDCG@10

### Common Failure Modes
1. **Complex nested tables** with merged cells
2. **Low-quality scans** with poor resolution
3. **Diagram interpretation** (technical schematics, anatomical illustrations)
4. **Multi-language labels** and symbol legends
5. **Cross-section references** within documents

### Document Challenges
- **Scanned PDFs:** OCR accuracy varies with quality
- **Complex layouts:** Multi-column, nested elements
- **Technical diagrams:** Assembly instructions, anatomical illustrations
- **Regulatory language:** Specific terminology ("contraindication" vs "warning")
- **Multi-page tables:** Maintaining context across pages

### Future Directions
- Fine-tuned OCR models on medical device corpus
- Vision-language models (Qwen2-VL) for diagram understanding
- Graph-based document structure modeling
- Specialized embeddings for regulatory terminology
- Multi-language support for international IFUs

---

**Last Updated:** 2025-12-05
**Maintained by:** Mixpeek Team
