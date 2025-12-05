# Medical Device Documentation Benchmark

**Learn more: https://mxp.co/device**

Benchmark for evaluating retrieval performance on medical device regulatory documents including IFUs (Instructions for Use), recall notices, MAUDE reports, 510(k) summaries, and clinical trial data.

## Features

### Multimodal Extraction
- **PDF parsing** with PyMuPDF for native text
- **OCR pipeline** with TrOCR + Tesseract for scanned documents
- **Layout detection** with LayoutLMv3 + DocFormer ensemble
- **Table extraction** with TATR + TableFormer validation
- **Diagram extraction** with ColPali encoding + BLIP-2 captions

### Advanced Processing
- **Smart chunking** with content-type specific strategies
- **Bounding box tracking** for precise source citation
- **Multi-vector embeddings** for text, tables, and diagrams
- **Domain-specific fine-tuning** on medical device corpus

### Query Types
- **Safety queries**: "What are the contraindications?"
- **Procedures**: "Show me sterilization instructions"
- **Specifications**: "What are the device dimensions?"
- **Adverse events**: "Find catheter blockage reports"
- **Regulatory**: "What are the indications for use?"
- **Compatibility**: "Is this device MRI compatible?"

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/ethan/Dev/mixpeek/benchmarks/device
pip install -r ../shared/requirements.txt

# System dependencies (macOS)
brew install tesseract poppler

# Or Ubuntu/Debian
# sudo apt-get install tesseract-ocr poppler-utils
```

### 2. Run Benchmark

```bash
# Run with demo data
python run.py

# Run quick test (3 queries, 5 documents)
python run.py --quick

# Run with your own device documentation
python run.py --data-dir /path/to/device/pdfs
```

### 3. View Results

Results are saved to `results/benchmark_results.json`

```bash
cat results/benchmark_results.json | jq '.aggregate_metrics'
```

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `ndcg@10` | Ranking quality (primary metric) | >0.72 |
| `recall@5` | Coverage in top 5 results | >0.68 |
| `mrr` | Mean reciprocal rank | >0.60 |
| `precision@10` | Accuracy in top 10 | >0.55 |
| `latency_p95` | 95th percentile latency | <3s |

## Architecture

```
Medical Device PDF → Extraction Pipeline → Chunking → Multi-Vector Embedding → Vector DB
                     ├── Native Text (PyMuPDF)
                     ├── OCR (TrOCR + Tesseract)
                     ├── Layout (LayoutLMv3)
                     ├── Tables (TATR + TableFormer)
                     └── Diagrams (ColPali + BLIP-2)

Query → Query Embedding → Multi-Modal Search → Reranking → Results
        (Text + Visual)   (Text, Table, Diagram)  (Cross-encoder)
```

## Sample Queries

The benchmark includes queries across different regulatory domains:

### 1. Safety & Warnings
- "What are the contraindications for this device?"
- "Find adverse events related to catheter blockage"
- "What are the potential complications?"

### 2. Procedures & Instructions
- "Show me the sterilization and cleaning instructions"
- "How should the device be stored?"
- "What training is required to use this device?"

### 3. Technical Specifications
- "What are the device specifications and dimensions?"
- "What materials is the device made of?"
- "Find information about MRI compatibility"

### 4. Regulatory & Clinical
- "What are the indications for use?"
- "Show me the clinical trial results"
- "What is the device classification?"

## Data Format

### Documents
Place medical device PDFs in your data directory:
```
data/
  ├── device_ifu_manual.pdf
  ├── 510k_summary.pdf
  ├── clinical_trials.pdf
  ├── recall_notice.pdf
  └── maude_report.pdf
```

### Document Types
The system handles various document types:
- **IFU** (Instructions for Use)
- **510(k)** summaries
- **Recall** notices
- **MAUDE** reports (adverse events)
- **Clinical** trial data
- **Labeling** documents

### Queries
Queries are defined in code but can be loaded from JSON:
```json
{
  "id": "dev_001",
  "text": "What are the contraindications for this device?",
  "intent": "safety",
  "domain": "warnings_precautions"
}
```

### Relevance Judgments
Ground truth judgments (0-3 scale):
```json
{
  "query_id": "dev_001",
  "doc_id": "ifu_section_4_contraindications",
  "relevance": 3
}
```

Relevance scale:
- **0**: Not relevant
- **1**: Somewhat relevant (mentions topic)
- **2**: Highly relevant (answers query)
- **3**: Perfect match (exact answer)

## Integration with Medical Device Extractor

This benchmark integrates with the medical device extractor at:
`/Users/ethan/Dev/mixpeek/extractors/medical-device`

The extractor provides:
- PDF parsing with PyMuPDF
- OCR with TrOCR + Tesseract
- Layout detection with LayoutLMv3
- Table extraction with TATR + TableFormer
- Diagram extraction with ColPali
- Multi-vector embeddings

## Extending the Benchmark

### Add Custom Queries

Edit `run.py` and add to `get_sample_queries()`:

```python
Query(
    id="dev_custom_001",
    text="Your custom query here",
    intent="safety",  # or procedure, specification, regulatory
    domain="warnings_precautions"
)
```

### Add Ground Truth Judgments

Add to `get_sample_judgments()`:

```python
RelevanceJudgment(
    query_id="dev_custom_001",
    doc_id="your_section_id",
    relevance=3  # 0-3 scale
)
```

### Process Your Documents

```bash
# Using the medical device extractor directly
cd /Users/ethan/Dev/mixpeek/extractors/medical-device

python scripts/extract_document.py \
    /path/to/device.pdf \
    --manufacturer "Acme Medical" \
    --device-name "CardioDevice Pro" \
    --doc-type ifu \
    --output extracted.json

# Then run benchmark
cd /Users/ethan/Dev/mixpeek/benchmarks/device
python run.py --data-dir /path/to/processed/docs
```

## Performance Targets

Based on analysis of medical device documentation:

| System | NDCG@10 | Recall@5 | Notes |
|--------|---------|----------|-------|
| BM25 Baseline | 0.41 | 0.35 | Keyword only |
| Dense Retrieval (BGE) | 0.58 | 0.52 | Single vector |
| **Mixpeek Multi-Modal** | **0.72** | **0.68** | Text + Tables + Diagrams |
| + Cross-encoder | 0.78 | 0.74 | With reranking |

## Regulatory Compliance

This benchmark is designed to support:
- **FDA 21 CFR Part 820** (Quality System Regulation)
- **ISO 13485** (Medical devices quality management)
- **IEC 62304** (Medical device software lifecycle)
- **EU MDR 2017/745** (Medical Device Regulation)

The system maintains:
- Document provenance (source citations with page numbers)
- Extraction audit trails
- Versioning of document changes
- Quality metrics for extraction confidence

## Common Document Challenges

Medical device documents present unique challenges:

### 1. Complex Tables
- Nested headers
- Merged cells
- Vertical text
- Multi-page tables

### 2. Diagrams & Schematics
- Anatomical illustrations
- Device assembly diagrams
- Workflow flowcharts
- Technical schematics

### 3. Regulatory Language
- Specific terminology (e.g., "contraindication" vs "warning")
- Cross-references between sections
- Symbol legends
- Multi-language labels

### 4. Document Quality
- Scanned PDFs (low resolution)
- Handwritten annotations
- Redacted sections
- Watermarks

## Citation

```bibtex
@misc{mixpeek-device-benchmark,
  title={Medical Device Documentation Retrieval Benchmark},
  author={Mixpeek},
  year={2025},
  url={https://mxp.co/device}
}
```

## Learn More

- **Full Documentation**: https://mxp.co/device
- **Source Code**: `/Users/ethan/Dev/mixpeek/extractors/medical-device`
- **Benchmark Suite**: https://github.com/mixpeek/benchmarks

---

Built by [Mixpeek](https://mixpeek.com) — Multimodal AI for regulated industries.
