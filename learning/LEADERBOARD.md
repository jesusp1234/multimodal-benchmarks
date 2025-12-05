# Curriculum Search Leaderboard

Official leaderboard for the curriculum search benchmark (educational video content).

**Learn more:** [mxp.co/learning](https://mxp.co/learning)

## Top Systems

| Rank | System | NDCG@10 | Recall@50 | MRR | Precision@10 | Latency (p95) | Date | Submitter |
|------|--------|---------|-----------|-----|--------------|---------------|------|-----------|
| 🥇 1 | Mixpeek Multi-Modal + LLM Reranking | **0.8400** | 0.9300 | 0.8900 | 0.7800 | 350ms | 2025-01 | Mixpeek Team |
| 🥈 2 | Mixpeek Multi-Modal (HyDE + RRF) | **0.7900** | 0.9100 | 0.8500 | 0.7400 | 180ms | 2025-01 | Mixpeek Team |
| 🥉 3 | Dense Retrieval (BGE-M3) | 0.6800 | 0.8200 | 0.7200 | 0.6500 | 120ms | 2025-01 | Baseline |
| 4 | BM25 Baseline | 0.4500 | 0.6200 | 0.5100 | 0.4200 | 50ms | 2025-01 | Baseline |

## Evaluation Details

### Dataset
- **Content:** Educational videos (lectures), presentation slides (PDF), code examples
- **Queries:** 10 diverse queries across concept explanation, code examples, comparisons, troubleshooting
- **Judgments:** Human-annotated relevance (0-3 scale)

### Metrics
- **NDCG@10** (Primary): Normalized Discounted Cumulative Gain at position 10
- **Recall@50**: Fraction of relevant segments in top 50 results (important for educational content)
- **MRR**: Mean Reciprocal Rank (position of first relevant segment)
- **Precision@10**: Fraction of top 10 results that are relevant
- **Latency (p95)**: 95th percentile end-to-end latency

### Query Breakdown

Sample queries used in evaluation:

1. **Concept Explanation** (30%)
   - "How do pointers work in C?"
   - "What is pointer arithmetic?"
   - "Explain how recursion works"

2. **Code Examples** (20%)
   - "Show me examples of memory allocation with malloc"
   - "Show me how to use structs in C"

3. **Comparisons** (20%)
   - "What is the difference between stack and heap memory?"
   - "Explain the difference between malloc and calloc"

4. **Troubleshooting** (20%)
   - "How do I prevent memory leaks?"
   - "What are common segmentation fault causes?"

5. **Tool Usage** (10%)
   - "How do I debug memory issues with valgrind?"

## System Descriptions

### Mixpeek Multi-Modal + LLM Reranking
- **Architecture:** Multi-vector (transcript + code + visual + contextual) + HyDE + RRF + Claude listwise reranking
- **Models:** Whisper (ASR), BGE-M3 (text embeddings), SFR-Embedding-Code (code), Claude 3.5 Sonnet (reranking)
- **Features:** Word-level timestamps, scene detection, code analysis, HyDE query enhancement, multi-vector fusion
- **Source:** [/Users/ethan/Dev/mixpeek/extractors/curriculum](../../../extractors/curriculum)

### Mixpeek Multi-Modal (HyDE + RRF)
- **Architecture:** Multi-vector representation with HyDE and Reciprocal Rank Fusion
- **Models:** Whisper (base), BGE-M3, SFR-Embedding-Code, PySceneDetect
- **Features:** 4-5 embeddings per segment (transcript, code, visual, bound, concept), HyDE, RRF
- **Source:** [/Users/ethan/Dev/mixpeek/extractors/curriculum](../../../extractors/curriculum)

### Dense Retrieval (BGE-M3)
- **Architecture:** Single dense vector per segment
- **Model:** BGE-M3 (multi-functionality: dense + sparse + ColBERT)
- **Features:** Transcript-only embeddings

### BM25 Baseline
- **Architecture:** Sparse keyword matching on transcripts
- **Features:** TF-IDF based ranking

## Submit Your Results

To submit your system to the leaderboard:

1. **Run the benchmark:**
   ```bash
   cd learning
   python run.py
   ```

2. **Results saved to:** `results/benchmark_results.json`

3. **Create submission:**
   - Fork this repo
   - Add your results file to `learning/submissions/your-system-name.json`
   - Include a description of your system in `learning/submissions/your-system-name.md`

4. **Open a PR** with:
   - System description (architecture, models, features)
   - Hyperparameters
   - ASR and scene detection settings
   - Hardware used

5. **We'll verify and add to leaderboard**

### Submission Template

Create `submissions/your-system.md`:

```markdown
# Your System Name

## Architecture
Describe your approach...

## Models
- ASR: Whisper (which variant?)
- Scene detection: ...
- Text embeddings: ...
- Code embeddings: ...
- Reranker: ...

## Features
- Multi-vector: [transcript, code, visual, ...]
- Query enhancement: HyDE? Query expansion?
- Fusion: RRF? Weighted sum?

## Hyperparameters
- Whisper model: base/small/medium
- Scene threshold: ...
- top_k: ...
- ...

## Hardware
- GPU: ...
- RAM: ...

## Results
- NDCG@10: X.XXXX
- Recall@50: X.XXXX
- MRR: X.XXXX
```

## Rules

1. **Fair Comparison:** No query-specific tuning or overfitting
2. **Reproducible:** Must include enough detail to reproduce results
3. **Open Models:** Prefer open-source models (proprietary APIs allowed but noted)
4. **Content Type:** Test on educational video content (lectures, tutorials, courses)
5. **Honest Reporting:** Report exactly what the benchmark outputs

## Historical Results

Track progress over time:

| Date | Best NDCG@10 | System |
|------|--------------|--------|
| 2025-01 | 0.8400 | Mixpeek Multi-Modal + LLM Reranking |

## Analysis

### What Works Well
1. **Multi-vector representation** crucial for educational content (0.79 vs 0.68 for single vector)
2. **HyDE (Hypothetical Document Embeddings)** significantly improves concept queries
3. **Code embeddings** essential for programming tutorials
4. **Scene-transcript binding** helps with temporal alignment
5. **LLM listwise reranking** adds +5 points to NDCG@10
6. **Reciprocal Rank Fusion** effectively combines modalities

### Common Failure Modes
1. **Abstract concepts** without visual examples
2. **Multi-step procedures** spanning multiple scenes
3. **Code variations** (different implementations of same concept)
4. **Prerequisite dependencies** (assuming prior knowledge)
5. **Temporal reasoning** (finding specific point in explanation)

### Content Challenges
- **ASR errors:** Technical terminology, accents
- **Scene boundaries:** Gradual transitions vs hard cuts
- **Code in slides:** OCR accuracy on code snippets
- **Handwriting:** Annotations, diagrams drawn during lecture
- **Multi-modal alignment:** Syncing visual, audio, and code

### What Makes Educational Retrieval Different
1. **Pedagogical intent matters:** Not just keyword matching
2. **Examples are crucial:** Students want to see implementations
3. **Prerequisites matter:** Need context of what came before
4. **Multiple modalities:** Code, slides, and speech all contribute
5. **Temporal context:** Position in lecture affects understanding

### Future Directions
- Vision-language models (Qwen2-VL, GPT-4V) for slide understanding
- Knowledge graph of concepts and prerequisites
- Fine-tuned embeddings on educational corpus
- Staleness detection (detecting when libraries/languages change)
- Personalized retrieval based on student level
- Interactive retrieval with follow-up questions

---

**Last Updated:** 2025-12-05
**Maintained by:** Mixpeek Team
