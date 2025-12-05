# Curriculum Search Benchmark

**Learn more: https://mxp.co/learning**

Benchmark for evaluating retrieval performance on educational video content including lecture transcripts, presentation slides, code examples, and visual demonstrations.

## Features

### Multi-Modal Content Processing
- **Video transcription** with Whisper ASR (word-level timestamps)
- **Scene detection** with PySceneDetect for temporal segmentation
- **Slide extraction** from PDF with OCR text extraction
- **Code analysis** with multi-language support and AST parsing
- **Keyframe extraction** for visual content

### State-of-the-Art Retrieval
- **BGE-M3 embeddings** (Dense + Sparse + ColBERT in one model)
- **Multi-vector representation** (transcript, code, visual, contextual)
- **HyDE** (Hypothetical Document Embeddings) for query enhancement
- **Reciprocal Rank Fusion** across modalities
- **Scene-transcript binding** for temporal alignment

### Query Types
- **Concept explanation**: "How do pointers work in C?"
- **Code examples**: "Show me malloc examples"
- **Comparisons**: "Difference between stack and heap?"
- **Troubleshooting**: "How to prevent memory leaks?"
- **Tool usage**: "How to debug with valgrind?"

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/ethan/Dev/mixpeek/benchmarks/learning
pip install -r ../shared/requirements.txt

# System dependencies (macOS)
brew install ffmpeg poppler

# Or Ubuntu/Debian
# sudo apt-get install ffmpeg poppler-utils
```

### 2. Run Benchmark

```bash
# Run with demo data
python run.py

# Run quick test (3 queries)
python run.py --quick

# Run with your own course content
python run.py --data-dir /path/to/course/content
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
| `recall@50` | Coverage in top 50 results | >0.90 |
| `mrr` | Mean reciprocal rank | >0.65 |
| `precision@10` | Accuracy in top 10 | >0.60 |
| `latency_p95` | 95th percentile latency | <200ms |

## Architecture

```
Course Content (Video + Slides + Code)
           ↓
    Content Extraction
    ├── Video: Whisper ASR + Scene Detection
    ├── Slides: PDF Processing + OCR
    └── Code: Multi-language Analysis
           ↓
    Multi-Vector Embedding
    ├── Transcript Embedding (BGE-M3)
    ├── Code Embedding (StarCoder/SFR)
    ├── Visual Embedding (BGE-M3)
    └── Bound Embedding (Scene + Transcript)
           ↓
    Vector Store (In-Memory / Qdrant)
           ↓
    Multi-Modal Retrieval
    ├── HyDE Query Enhancement
    ├── Multi-Vector Search
    └── Reciprocal Rank Fusion
```

## Sample Queries

The benchmark includes queries across different learning intents:

### 1. Concept Explanation
- "How do pointers work in C?"
- "What is pointer arithmetic?"
- "Explain how recursion works"

### 2. Code Examples
- "Show me examples of memory allocation with malloc"
- "Show me how to use structs in C"
- "Demonstrate linked list implementation"

### 3. Comparisons
- "What is the difference between stack and heap memory?"
- "Explain the difference between malloc and calloc"
- "Compare arrays vs linked lists"

### 4. Troubleshooting
- "How do I prevent memory leaks?"
- "What are common segmentation fault causes?"
- "How to fix buffer overflow errors?"

### 5. Tool Usage
- "How do I debug memory issues with valgrind?"
- "How to use gdb for debugging?"
- "What compiler flags should I use?"

## Data Format

### Course Content Structure
Organize your course content:
```
course-content/
  ├── video.mp4              # Lecture video
  ├── slides.pdf             # Presentation slides
  └── code.zip               # Code examples (or code/ directory)
      ├── example1.c
      ├── example2.c
      └── example3.c
```

### Supported Formats
- **Video**: MP4, AVI, MOV
- **Slides**: PDF
- **Code**: ZIP archive or directory with source files

### Queries
Queries are defined in code but can be loaded from JSON:
```json
{
  "id": "learn_001",
  "text": "How do pointers work in C?",
  "intent": "concept_explanation",
  "domain": "systems_programming"
}
```

### Relevance Judgments
Ground truth judgments (0-3 scale):
```json
{
  "query_id": "learn_001",
  "doc_id": "segment_45_pointers_intro",
  "relevance": 3
}
```

Relevance scale:
- **0**: Not relevant
- **1**: Somewhat relevant (mentions topic)
- **2**: Highly relevant (explains concept)
- **3**: Perfect match (best explanation with examples)

## Integration with Curriculum Extractor

This benchmark integrates with the curriculum extractor at:
`/Users/ethan/Dev/mixpeek/extractors/curriculum`

The extractor provides:
- Video transcription with Whisper
- Scene detection with PySceneDetect
- Slide extraction from PDF
- Code analysis and extraction
- Multi-vector embeddings (BGE-M3)
- HyDE query enhancement
- Reciprocal Rank Fusion

## Extending the Benchmark

### Add Custom Queries

Edit `run.py` and add to `get_sample_queries()`:

```python
Query(
    id="learn_custom_001",
    text="Your custom query here",
    intent="concept_explanation",  # or code_example, comparison, etc.
    domain="systems_programming"
)
```

### Add Ground Truth Judgments

Add to `get_sample_judgments()`:

```python
RelevanceJudgment(
    query_id="learn_custom_001",
    doc_id="segment_id_from_extraction",
    relevance=3  # 0-3 scale
)
```

### Process Your Course Content

```bash
# Using the curriculum extractor directly
cd /Users/ethan/Dev/mixpeek/extractors/curriculum

python main.py  # Or use the API

# Then run benchmark
cd /Users/ethan/Dev/mixpeek/benchmarks/learning
python run.py --data-dir /path/to/your/course
```

## Performance Targets

Based on analysis of educational content retrieval:

| System | NDCG@10 | Recall@50 | Notes |
|--------|---------|-----------|-------|
| BM25 Baseline | 0.45 | 0.62 | Text-only keyword |
| Dense Retrieval (BGE) | 0.68 | 0.82 | Single vector |
| **Mixpeek Multi-Modal** | **0.79** | **0.91** | Multi-vector + HyDE |
| + LLM Reranking | 0.84 | 0.93 | With Claude reranking |

## Multi-Vector Representation

Each content segment gets multiple independent embeddings:

```python
segment.transcript_embedding    # Instructor explanation
segment.code_embedding         # Code semantics
segment.visual_embedding       # Slide content
segment.bound_embedding        # Scene + transcript binding
segment.concept_embedding      # LLM-extracted concepts
```

This allows retrieval to match queries against different modalities:
- **Concept queries** → Transcript embeddings
- **Code queries** → Code embeddings
- **Visual queries** → Slide embeddings
- **Multi-modal queries** → Fusion across all vectors

## HyDE (Hypothetical Document Embeddings)

Instead of embedding the query directly, HyDE generates a hypothetical answer:

```
Query: "How do pointers work in C?"
         ↓
HyDE: "Pointers in C store memory addresses. You declare them using *,
       dereference with *, and get addresses with &. For example:
       int x = 42;
       int *ptr = &x;  // ptr stores address of x
       printf("%d", *ptr);  // prints 42"
         ↓
Embed HyDE text → Better retrieval
```

This significantly improves retrieval quality for concept-based queries.

## Reciprocal Rank Fusion

Combine rankings from multiple vector types:

```
Transcript results: [seg_A, seg_B, seg_C]
Code results:       [seg_C, seg_A, seg_D]
Visual results:     [seg_B, seg_C, seg_A]
         ↓
RRF Fusion: [seg_C, seg_A, seg_B, seg_D]
```

RRF gives higher scores to segments that appear in multiple result lists.

## Educational Use Cases

This benchmark supports various educational scenarios:

### 1. Student Q&A
Students asking conceptual questions about lecture material

### 2. Code Example Search
Finding specific code patterns or implementations

### 3. Review & Study
Locating explanations of specific topics for exam prep

### 4. Prerequisite Learning
Finding foundational concepts before advanced topics

### 5. Staleness Detection
Identifying outdated content when libraries/languages change

## Citation

```bibtex
@misc{mixpeek-curriculum-benchmark,
  title={Curriculum Search Benchmark for Educational Video Content},
  author={Mixpeek},
  year={2025},
  url={https://mxp.co/learning}
}
```

## Learn More

- **Full Documentation**: https://mxp.co/learning
- **Source Code**: `/Users/ethan/Dev/mixpeek/extractors/curriculum`
- **Benchmark Suite**: https://github.com/mixpeek/benchmarks
- **Research**: [HyDE Paper](https://arxiv.org/abs/2212.10496), [BGE-M3 Paper](https://arxiv.org/abs/2402.03216)

---

Built by [Mixpeek](https://mixpeek.com) — Multimodal AI for regulated industries.
