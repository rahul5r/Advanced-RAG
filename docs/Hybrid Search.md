# HybridSearchRetriever Documentation

## Overview

`HybridSearchRetriever` combines:

* **Dense semantic search** using Qdrant (vector embeddings)
* **Sparse keyword search** using BM25
* **Weighted score fusion** to merge both result sets

Key features:

* Persists BM25 index and chunks to local disk (JSON + pickle) for fast reload
* Rebuilds sparse index from Qdrant payloads if cache is missing
* Min-max normalization for cross-score compatibility
* Configurable fusion weights (default: 0.6 dense, 0.4 sparse)

---

## Architecture

```text
                  Query
                    │
          ┌─────────┴─────────┐
          │                   │
       Qdrant              BM25
    (dense vector)      (keyword)
          │                   │
       Scores              Scores
          │                   │
          └─────────┬─────────┘
                    │
           Min-Max Normalize
                    │
         Weighted Fusion (α/β)
                    │
             Top-K Results
```

---

## How It Works

### 1. Index Building (`build_index`)

#### Sparse Index (BM25)

* Tokenizes chunks using regex: `\b\w+\b` (lowercase word boundaries)
* Saves chunks to `data/retriever_cache/{collection}_chunks.json`
* Saves BM25 object to `data/retriever_cache/{collection}_bm25.pkl`

```python
tokenized_corpus = [_tokenize(doc) for doc in chunks]
self.bm25 = BM25Okapi(tokenized_corpus)
```

#### Dense Index (Qdrant)

* Generates embeddings via `BaseAgent.get_embedding()`
* Creates Qdrant collection with COSINE distance
* Stores points with payload: `{"text": chunk, "chunk_id": i, "document_type": "pdf"}`

```python
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": chunk, "chunk_id": i, "document_type": "pdf"},
    )
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]
```

---

### 2. Retrieval (`retrieve`)

#### Step 1: Ensure Index Ready

* Checks Qdrant collection exists
* Loads BM25 from disk or rebuilds from Qdrant payloads via `scroll()`

#### Step 2: Dense Retrieval

```python
query_vec = self.agent.get_embedding(query)
semantic_hits = self.client.query_points(
    collection_name=self.collection_name,
    query=query_vec,
    limit=top_k * 3,
)
```

#### Step 3: Sparse Retrieval

```python
bm25_scores = self.bm25.get_scores(_tokenize(query))
```

#### Step 4: Min-Max Normalization

```python
def normalize(scores: dict[str, float]) -> dict[str, float]:
    min_v, max_v = min(scores.values()), max(scores.values())
    if max_v == min_v:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
```

#### Step 5: Weighted Fusion

```python
alpha, beta = 0.6, 0.4
final_scores = {
    doc: alpha * semantic_norm.get(doc, 0.0) + beta * bm25_norm.get(doc, 0.0)
    for doc in all_docs
}
```

#### Step 6: Return Top-K

Returns documents sorted by fused score.

---

## Key Parameters

| Parameter         | Description                                      |
| ----------------- | ------------------------------------------------ |
| `top_k`           | Number of results to return                      |
| `collection_name` | Qdrant collection name                           |
| `artifacts_dir`   | Local directory for BM25/chunks cache (optional) |
| `alpha`           | Weight for dense search (default: 0.6)           |
| `beta`            | Weight for sparse search (default: 0.4)          |

---

## Why This Approach Works

* **Dense search** captures semantic meaning and paraphrases
* **BM25** captures exact keyword matches and terminology
* **Fusion** leverages both signals for higher recall
* **Disk caching** avoids re-computation across restarts
* **Auto-rebuild** from Qdrant ensures resilience if cache is lost

---

## Limitations

* Basic tokenization (no stemming, stopwords, or lemmatization)
* Fixed fusion weights may not suit all domains
* Pickle dependency for BM25 persistence (Python version coupling)
* Duplicate texts across chunks may affect scoring

---

## Possible Improvements

* Advanced tokenization (lowercase, stemming, stopword removal)
* Dynamic weight adjustment based on query characteristics
* Reciprocal Rank Fusion (RRF) as alternative to min-max
* Hybrid batch retrieval for reduced latency

---

## Summary

`HybridSearchRetriever`:

* Maintains both dense (Qdrant) and sparse (BM25) indexes
* Persists sparse artifacts to disk for fast reload
* Rebuilds BM25 from Qdrant if cache is missing
* Normalizes and fuses scores with configurable weights
* Returns top-K documents balancing semantics and keywords

Ideal for RAG systems where both **meaning** and **terminology** matter.

---
