# ContextReRankingRetriever Documentation

## Overview

`ContextReRankingRetriever` combines:

* **Broad semantic retrieval** from Qdrant (top_k * 3 candidates)
* **LLM-based re-ranking** to reorder candidates by relevance

Key features:

* Fetches more candidates than needed for effective re-ranking
* Robust regex-based index parsing with bounds checking
* Deduplication and backfill logic for reliable output
* Falls back to vector order if LLM parsing fails

---

## Architecture

```text
                  Query
                    │
          ┌─────────┴────────────┐
          │                      │
    Qdrant Embedding    Retrieve top_k*3 Candidates
        (semantic)               │
          │                      │
          └──── Candidate Contexts ──┘
                       │
                 LLM Re-ranking
                       │
              Top-K Ordered Contexts
                       │
                     Return
```

---

## How It Works

### 1. Index Building (`build_index`)

* Generates embeddings via `BaseAgent.get_embedding()`
* Creates Qdrant collection with COSINE distance
* Stores points with payload: `{"text": chunk, "chunk_id": i, "document_type": "pdf"}`
* Uses `wait=True` to ensure persistence before retrieval

```python
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=v,
        payload={"text": c, "chunk_id": i, "document_type": "pdf"},
    )
    for i, (c, v) in enumerate(zip(chunks, embeddings))
]
```

---

### 2. Retrieval (`retrieve`)

#### Step 1: Broad Retrieval

Fetches 3x candidates (reduced from 5x to prevent LLM confusion):

```python
query_vec = self.agent.get_embedding(query)
search_result = self.client.query_points(
    collection_name=self.collection_name,
    query=query_vec,
    limit=top_k * 3,
)
candidates = [hit.payload["text"] for hit in search_result.points]
```

#### Step 2: LLM Re-Ranking Prompt

```python
rerank_prompt = f"""Given the following query and a list of context chunks, re-order the chunks based on their relevance to the query.
Return the indices of the top {top_k} chunks in order of relevance, separated by commas.

Rules:
- Return ONLY the indices as integers between 0 and {len(candidates) - 1} (e.g., 2, 0, 5).
- Provide exactly {top_k} indices if possible.
- Do not repeat an index.
- Do not include any explanation or extra text.

Query: {query}

Chunks:
{chr(10).join([f"[{i}] {chunk}" for i, chunk in enumerate(candidates)])}

Top {top_k} Indices:"""
```

#### Step 3: Parse LLM Response

```python
# Extract all integers from response
extracted_indices = [
    int(idx)
    for idx in re.findall(r'\d+', response)
    if int(idx) < len(candidates)  # Bounds check during parsing
]

# Deduplicate while preserving order
unique_indices = []
for idx in extracted_indices:
    if idx not in unique_indices:
        unique_indices.append(idx)

final_indices = unique_indices[:top_k]

# Backfill if LLM returned fewer than top_k
if len(final_indices) < top_k:
    for idx in range(len(candidates)):
        if idx not in final_indices:
            final_indices.append(idx)
        if len(final_indices) >= min(top_k, len(candidates)):
            break
```

#### Step 4: Return Re-Ranked Contexts

```python
return [candidates[i] for i in final_indices]
```

If parsing fails, falls back to `candidates[:top_k]` (vector order).

---

## Key Parameters

| Parameter         | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `top_k`           | Number of final context chunks to return             |
| `collection_name` | Qdrant collection name                               |
| `agent`           | BaseAgent for embeddings and LLM re-ranking          |

---

## Why This Approach Works

* **Broad retrieval** captures diverse candidate contexts
* **LLM re-ranking** leverages language understanding for relevance
* **Robust parsing** handles varied LLM output formats
* **Fallback behavior** ensures graceful degradation

---

## Limitations

* LLM re-ranking adds latency and cost
* Token limits if candidate chunks are large
* Parsing relies on LLM following index format
* No hybrid signals (purely semantic + LLM judgment)

---

## Possible Improvements

* **Chunk truncation** to avoid token overflow
* **Structured JSON response** for safer parsing
* **Configurable candidate multiplier** based on query complexity
* **Hybrid scoring**: Combine embedding scores with LLM rankings

---

## Summary

`ContextReRankingRetriever`:

* Retrieves top_k*3 candidates via semantic search
* Uses LLM to re-order by relevance
* Returns top-K contexts optimized for answer quality
* Falls back to vector order if re-ranking fails

Ideal for RAG pipelines where **context relevance** is critical and latency budget allows LLM re-ranking.

---
