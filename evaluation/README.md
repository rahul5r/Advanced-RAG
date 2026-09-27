# Evaluation Pipeline — RAGAS

Automated, grounded evaluation of all **5 advanced RAG retrieval strategies** using [RAGAS](https://docs.ragas.io) metrics powered by Google Gemini.

---

## Why grounded evaluation?

Using a generic public dataset (SQuAD, TriviaQA, etc.) would measure how well your retrievers answer questions that have **nothing to do with your indexed PDF**.  Faithfulness and context-recall scores would reflect topic mismatch — not retriever quality.

The correct approach:

> **Generate the test-set directly from `paper.pdf`** → every question is answerable from the exact content indexed in Qdrant → faithfulness and recall scores are meaningful.

RAGAS `TestsetGenerator` does this automatically by building a knowledge graph from the document, then synthesising diverse question types (direct retrieval, multi-hop abstract, multi-hop specific).

---

## Folder Structure

```
evaluation/
├── __init__.py
├── README.md                    ← you are here
├── config.py                    ← EvaluationConfig, thresholds, retriever registry
├── dataset_generator.py         ← RAGAS TestsetGenerator → testset.json
├── dataset.py                   ← Load + validate testset.json
├── ragas_evaluator.py           ← RAGAS + Gemini LLM/embeddings integration
├── pipeline.py                  ← Core orchestrator (retrieve → generate → score)
├── reporter.py                  ← Console, JSON, and Markdown output
├── run_evaluation.py            ← CLI entry point
├── data/
│   └── testset.json             ← Auto-generated; commit to version control after review
└── reports/                     ← Evaluation outputs (auto-created)
    ├── ragas_report_<ts>.json
    └── ragas_report_<ts>.md
```

| Metric                | Meaning                               |
| --------------------- | ------------------------------------- |
| **faithfulness**      | Is answer grounded in context?        |
| **context_recall**    | Did retriever fetch relevant info?    |
| **answer_relevancy**  | Is answer relevant to question?       |
| **context_precision** | How much retrieved context is useful? |


---

## Prerequisites

### 1. Install additional dependencies

```bash
# Using uv (recommended for this project)
uv add ragas langchain-google-genai langchain-community pypdf datasets

# Or pip
pip install ragas langchain-google-genai langchain-community pypdf datasets
```

### 2. Environment variables (`.env`)

```dotenv
GOOGLE_API_KEY=your_key_here
MODEL_NAME=gemini-2.0-flash
EMBEDDING_MODEL=models/text-embedding-004

QDRANT_URL=http://localhost:6333   # or your Qdrant Cloud URL
QDRANT_API_KEY=                    # leave blank for local

TOP_K=5
RAGAS_FAITHFULNESS_THRESHOLD=0.80
RAGAS_CONTEXT_RECALL_THRESHOLD=0.70
```

### 3. Index your document first

The evaluation reads from an **existing Qdrant collection** (`rag_collection`).
Index `paper.pdf` through the Streamlit app before running evaluation:

```bash
streamlit run src/app/main.py
# Upload PDF → click "Index Document"
```

---

## Running the Evaluation

### Recommended: full pipeline (generate + evaluate)

```bash
python -m evaluation.run_evaluation --generate
```

This runs both steps in sequence:
1. Generate a 20-question test-set from `data/pdfs/paper.pdf`
2. Evaluate all 5 retrievers and print the results table

---

### Step 1 only — generate the test-set

```bash
python -m evaluation.dataset_generator
```

Or via the main CLI:

```bash
python -m evaluation.run_evaluation --generate --gen-only
```

Customize:

```bash
python -m evaluation.dataset_generator \
    --pdf  data/pdfs/paper.pdf \
    --size 30 \
    --out  evaluation/data/testset.json
```

---

### Step 2 only — evaluate (test-set already exists)

```bash
python -m evaluation.run_evaluation
```

Evaluate specific retrievers:

```bash
python -m evaluation.run_evaluation --strategies hybrid rerank multi_hop
```

Custom top-k:

```bash
python -m evaluation.run_evaluation --top-k 5
```

Print results only (no report files):

```bash
python -m evaluation.run_evaluation --no-save
```

Full options:

```bash
python -m evaluation.run_evaluation \
    --strategies hybrid expansion rerank self_query multi_hop \
    --top-k 5 \
    --test-file evaluation/data/testset.json \
    --report-dir evaluation/reports \
    --verbose
```

---

## Retriever Strategies Evaluated

| Key | Class | Description |
|-----|-------|-------------|
| `hybrid` | `HybridSearchRetriever` | α·Qdrant cosine + (1-α)·BM25 keyword fusion |
| `expansion` | `SemanticQueryExpansionRetriever` | Gemini generates 3-4 paraphrases → merged, deduplicated results |
| `rerank` | `ContextReRankingRetriever` | 5× candidate pool → Gemini re-ranks by relevance |
| `self_query` | `SelfQueryRetriever` | Gemini extracts metadata filters → filtered Qdrant search |
| `multi_hop` | `MultiHopRetriever` | Decomposes complex questions → per-sub-query retrieval → union |

---

## RAGAS Metrics

| Metric | What it measures | Threshold |
|--------|-----------------|-----------|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? | ≥ 0.80 ✓ |
| **Context Recall** | Is the ground-truth answer recoverable from the retrieved chunks? | ≥ 0.70 ✓ |
| **Answer Relevancy** | Is the generated answer relevant to the question? | — (reported) |
| **Context Precision** | Are the retrieved chunks relevant to the question? | — (reported) |

A strategy **passes** when both Faithfulness **and** Context Recall meet their thresholds.

---

## Question Types Generated

RAGAS `TestsetGenerator` creates three types of questions from `paper.pdf`:

| Type | Weight | Description |
|------|--------|-------------|
| `SingleHopSpecific` | 50% | Direct factual questions answered by a single chunk |
| `MultiHopAbstract` | 25% | Abstract reasoning questions requiring multiple chunks |
| `MultiHopSpecific` | 25% | Specific facts that require connecting multiple parts of the document |

This distribution ensures the evaluation tests both simple retrieval and complex multi-step reasoning — which is precisely what distinguishes `multi_hop` and `rerank` from simpler strategies.

---

## Expected Console Output

```
────────────────────────────────────────────────────────────────────────────────────────────────────
Strategy        Faithfulness  CtxRecall  AnsRelevancy  CtxPrecision  Latency(ms)  Status
────────────────────────────────────────────────────────────────────────────────────────────────────
hybrid             0.8561     0.7890       0.8300        0.8100        912.4    ✓ PASS
expansion          0.8740     0.8010       0.8500        0.8200       1234.1    ✓ PASS
rerank             0.8920     0.8230       0.8700        0.8400       1890.5    ✓ PASS
self_query         0.8450     0.7700       0.8200        0.7600        930.0    ✓ PASS
multi_hop          0.8810     0.8150       0.8600        0.8300       3240.8    ✓ PASS
────────────────────────────────────────────────────────────────────────────────────────────────────
Thresholds:  Faithfulness ≥ 0.80  |  Context Recall ≥ 0.70
────────────────────────────────────────────────────────────────────────────────────────────────────

Summary: 5/5 strategies passed

Reports written to: evaluation/reports/
```

---

## Report Files

After each run, two report files are saved:

| File | Format | Contents |
|------|--------|----------|
| `ragas_report_<ts>.json` | JSON | Full machine-readable results — aggregate scores, per-sample scores, latencies, answers, and retrieved contexts for every question |
| `ragas_report_<ts>.md`   | Markdown | Human-readable summary table + per-strategy detail with per-sample breakdown |

---

## Interpreting Results

| Score range | Interpretation |
|-------------|----------------|
| **Faithfulness ≥ 0.85** | Excellent — answers are well-grounded in context |
| **Faithfulness 0.80–0.85** | Good — meets threshold |
| **Faithfulness < 0.80** | Poor — the model is hallucinating beyond the retrieved context |
| **Context Recall ≥ 0.80** | Excellent — retriever is surfacing the right information |
| **Context Recall 0.70–0.80** | Good — meets threshold |
| **Context Recall < 0.70** | Poor — relevant information is not being retrieved |

**Expected pattern across strategies:**
- `rerank` typically has the highest faithfulness (precision-first)
- `multi_hop` typically has the highest context recall (coverage-first)
- `hybrid` strikes the best latency/quality balance
- `expansion` helps with ambiguous or short queries

---

## Exit Code

| Code | Meaning |
|------|---------|
| `0` | All evaluated strategies passed thresholds |
| `1` | One or more strategies failed, or a critical error occurred |

Useful in CI:
```bash
python -m evaluation.run_evaluation || echo "Evaluation failed — check reports/"
```
