# MultiHopRetriever Documentation

## Overview

`MultiHopRetriever` handles complex queries by:

* **Decomposing questions** into 2-3 sub-queries using an LLM
* **Retrieving documents** for each sub-query independently
* **Re-ranking merged results** against the original query
* **Weighted fusion** of sub-query scores and re-ranking scores

This approach enables **multi-hop reasoning** where answers span multiple documents.

---

## Architecture

```text
               User Query
                    │
          ┌─────────┴─────────┐
          │                   │
   LLM Decomposition    Original Query Vec
   (2-3 sub-queries)           │
          │                    │
    ┌─────┴─────┐              │
    │           │              │
Sub-query 1  Sub-query 2       │
    │           │              │
    ▼           ▼              │
 Qdrant      Qdrant            │
    │           │              │
    └─────┬─────┘              │
          │                    │
     Merge Scores              │
     (keep best)               │
          │                    │
          └─────────┬──────────┘
                    │
              Re-Rank Against
              Original Query
                    │
         Weighted Fusion (0.7/0.3)
                    │
              Top-K Results
```

---

## How It Works

### 1. Index Building (`build_index`)

* Generates embeddings via `BaseAgent.get_embedding()`
* Creates Qdrant collection with COSINE distance
* Stores points with payload: `{"text": chunk, "chunk_id": i, "document_type": "pdf"}`
* Uses `wait=True` to ensure persistence

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

#### Step 1: Decompose Complex Query

```python
decomposition_prompt = f"""
Decompose the following question into 2 or 3 smaller retrieval sub-queries.
Keep them short and concrete.
Return ONLY a bulleted list.

Question: {query}
"""
response = self.agent.generate_response(decomposition_prompt)
sub_queries = [
    line.lstrip("- ").strip()
    for line in response.splitlines()
    if line.strip().startswith("-")
]
```

**Original query always included:**

```python
all_sub_queries = [query] + [sq for sq in sub_queries if sq and sq != query]
```

#### Step 2: Multi-Hop Retrieval

```python
# Embed original query once (reused for first sub-query and re-ranking)
original_query_vec = self.agent.get_embedding(query)

merged_scores: dict[str, float] = {}
for sq in all_sub_queries:
    query_vec = original_query_vec if sq == query else self.agent.get_embedding(sq)
    search_result = self.client.query_points(
        collection_name=self.collection_name,
        query=query_vec,
        limit=top_k,
    )
    for hit in search_result.points:
        text = hit.payload["text"]
        score = float(hit.score)
        if text not in merged_scores or score > merged_scores[text]:
            merged_scores[text] = score  # Keep best score
```

#### Step 3: Re-Rank Against Original Query

```python
# Re-rank all merged candidates against the original query
reranked_hits = self.client.query_points(
    collection_name=self.collection_name,
    query=original_query_vec,
    limit=max(top_k * 3, top_k),
).points
rerank_map = {hit.payload["text"]: float(hit.score) for hit in reranked_hits}
```

#### Step 4: Weighted Fusion

```python
final_scores = {
    text: (0.7 * merged_scores.get(text, 0.0)) + (0.3 * rerank_map.get(text, 0.0))
    for text in merged_scores
}
```

* **0.7 weight**: Sub-query retrieval scores (captures multi-hop coverage)
* **0.3 weight**: Re-ranking scores (anchors to original query relevance)

#### Step 5: Return Top-K

```python
ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
return [text for text, _ in ranked[:top_k]]
```

---

## Key Parameters

| Parameter         | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `query`           | Complex user question to decompose                    |
| `top_k`           | Number of final results to return                     |
| `collection_name` | Qdrant collection containing document vectors         |
| `agent`           | BaseAgent for decomposition and embedding generation  |

---

## Example

| Original Question                                          | Sub-Queries                                                    |
| ---------------------------------------------------------- | -------------------------------------------------------------- |
| `"Who won the Nobel Prize in 2020 and wrote about Paris?"` | `["Who won the Nobel Prize in 2020?", "Which works mention Paris?"]` |

1. Retrieves documents about "Nobel Prize 2020 winner"
2. Retrieves documents about "works mentioning Paris"
3. Merges results, keeping best scores
4. Re-ranks against original question
5. Returns top-K documents for answer generation

---

## Why This Approach Works

* **Decomposition** breaks complex questions into answerable parts
* **Independent retrieval** ensures coverage across information needs
* **Re-ranking** anchors results to the original question
* **Weighted fusion** balances multi-hop coverage with query relevance

---

## Limitations

* Requires LLM to generate meaningful sub-queries
* Decomposition may drift from core question
* Additional embeddings increase latency
* Limited to 2-3 sub-queries (may miss deeper reasoning chains)

---

## Possible Improvements

* **Adaptive sub-query count** based on question complexity
* **Sub-query validation** against original intent
* **Iterative hopping**: Use previous results to generate next sub-query
* **Frequency boosting**: Increase score for documents retrieved by multiple sub-queries

---

## Summary

`MultiHopRetriever`:

* Decomposes complex questions into 2-3 sub-queries via LLM
* Retrieves top-K for each sub-query independently
* Merges results, keeping best score per document
* Re-ranks merged candidates against original query
* Applies weighted fusion (0.7 sub-query + 0.3 re-ranking)
* Returns top-K documents for RAG generation

Ideal for **multi-hop QA** where answers require chaining information across multiple documents.

---
