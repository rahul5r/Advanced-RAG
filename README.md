# RAG Q&A — Advanced Patterns (Local, Gemini + FAISS)

> **Prototype 1** — Local RAG over PDF documents using Google Gemini for generation & embeddings, FAISS for dense search, and BM25 for keyword search. Five advanced retrieval strategies included.

---

## Table of Contents
1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Ingest PDFs](#ingest-pdfs)
5. [Run the API](#run-the-api)
6. [RAG Strategies](#rag-strategies)
7. [Evaluation](#evaluation)
8. [Docker](#docker)
9. [Environment Variables](#environment-variables)

---

## Architecture

```
PDF Files
    │
    ▼
DocumentProcessor ──► chunks (text + metadata)
    │
    ▼
EmbeddingsStore (Gemini text-embedding-004 → FAISS)
BM25 Index (rank-bm25)
    │
    ▼
Retriever  ◄── selects strategy from .env / request param
    │
    ├── dense        →  FAISS cosine search
    ├── hybrid       →  α·FAISS + (1-α)·BM25
    ├── expansion    →  Gemini query paraphrases → merge results
    ├── rerank       →  dense ×2 candidates → Gemini relevance scorer
    ├── self_query   →  Gemini filter extractor → filtered FAISS search
    └── multi_hop    →  Gemini decompose → per-sub-q retrieval → synthesis
    │
    ▼
Generator (Gemini flash, cited answer)
    │
    ▼
FastAPI  POST /ask  →  JSON response
```

---

## Project Structure

```
project_root/
├── .env
├── requirements.txt
├── src/
│   ├── ai/
│   │   ├── agents/
│   │   │   └── base_agent.py (Your provided code)
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── document_processor.py
│   │       ├── chunker.py
│   │       ├── retriever.py
│   │       └── generator.py
│   └── app/
│       └── main.py (Streamlit app)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd rag-qa-local

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY at minimum
```

### 3. Add PDFs

```bash
cp your-documents/*.pdf data/pdfs/
```

---

## Ingest PDFs

Build the FAISS and BM25 indexes from your PDFs:

```bash
python ingest.py --pdf-dir data/pdfs
```

Force a rebuild if indexes already exist:

```bash
python ingest.py --pdf-dir data/pdfs --rebuild
```

---

## Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger UI: http://localhost:8000/docs

### Example request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings of the report?",
    "strategy": "rerank",
    "top_k": 5
  }'
```

### Response schema

```json
{
  "question": "What are the main findings?",
  "answer": "The report finds that… [Source: report.pdf, Page: 4]",
  "strategy": "rerank",
  "top_k": 5,
  "sources": [{"source": "report.pdf", "page": 4}],
  "latency_ms": 1240.5
}
```

---

## RAG Strategies

| Strategy | Description | Best for |
|---|---|---|
| `dense` | Cosine similarity over Gemini embeddings (FAISS) | Baseline; fast |
| `hybrid` | `α·FAISS + (1-α)·BM25` (configurable via `HYBRID_ALPHA`) | Mixed keyword + semantic queries |
| `expansion` | Gemini generates 4 paraphrases; results merged & deduplicated | Ambiguous or short queries |
| `rerank` | Fetch 2×top_k candidates, Gemini scores relevance 0–10 | Precision-critical use cases |
| `self_query` | Gemini extracts metadata filters (source, page, type); filtered FAISS search | Queries with explicit document constraints |
| `multi_hop` | Gemini decomposes question → retrieve+answer per sub-question → synthesise | Complex, multi-part questions |

Switch strategy per-request (API param) or globally via `RETRIEVAL_STRATEGY` in `.env`.

---

## Evaluation

Edit `tests/test_samples.json` with your question/ground-truth pairs, then run:

```bash
python evaluate.py --test-file tests/test_samples.json
```

Evaluate specific strategies only:

```bash
python evaluate.py --strategies dense hybrid rerank --top-k 5
```

Output (console summary + `reports/ragas_report.json`):

```
============================================================
Strategy        Faithfulness  Ctx Recall   Latency(ms)
------------------------------------------------------------
dense               0.8312      0.7640        843.2  ✓
hybrid              0.8561      0.7890        912.4  ✓
expansion           0.8740      0.8010       1234.1  ✓
rerank              0.8920      0.8230       1890.5  ✓
self_query          0.8450      0.7700        930.0  ✓
multi_hop           0.8810      0.8150       3240.8  ✓
============================================================
```

**Pass criteria:** Faithfulness ≥ 0.80 | Context Recall ≥ 0.70

---

## Docker

```bash
# Build
docker build -t rag-qa-local .

# Run (mount data dir so indexes persist)
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  rag-qa-local
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required.** Google AI Studio API key |
| `MODEL_NAME` | `gemini-2.0-flash` | Gemini model for generation |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |
| `CHUNK_SIZE` | `512` | Token size per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `RETRIEVAL_STRATEGY` | `dense` | Default retrieval strategy |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `HYBRID_ALPHA` | `0.6` | Semantic weight in hybrid search (0–1) |
| `FAISS_INDEX_PATH` | `data/indexes/faiss.index` | FAISS index location |
| `BM25_INDEX_PATH` | `data/indexes/bm25.pkl` | BM25 index location |
| `RAGAS_FAITHFULNESS_THRESHOLD` | `0.80` | Minimum faithfulness score |
| `RAGAS_CONTEXT_RECALL_THRESHOLD` | `0.70` | Minimum context recall score |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Running Tests

```bash
pytest tests/ -v
```
