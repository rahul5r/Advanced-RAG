# SemanticQueryExpansionRetriever Documentation

## Overview

`SemanticQueryExpansionRetriever` improves retrieval recall by:

* Generating **3-4 paraphrased query variants** using an LLM
* Performing **vector retrieval** for each variant (including the original)
* **Merging results** by keeping the highest similarity score per document
* **Ranking by score** and returning top-K results

This approach captures documents that are semantically relevant but may not match the original query wording.

---

## Architecture

```text
               Original Query
                    │
          ┌─────────┴─────────┐
          │                   │
   LLM Paraphrasing    Keep Original
   (3-4 variants)           │
          │                 │
          └─────────┬───────┘
                    │
      Retrieve top-K for each query
                    │
       Merge & Keep Best Score
                    │
         Sort by Score (desc)
                    │
              Top-K Results
```

---

## How It Works

### 1. Index Building (`build_index`)

* Generates embeddings for all chunks via `BaseAgent.get_embedding()`
* Creates Qdrant collection with COSINE distance
* Stores points with payload: `{"text": chunk, "chunk_id": i, "document_type": "pdf"}`
* Uses `wait=True` to ensure upsert completes before retrieval

```python
embeddings = [self.agent.get_embedding(c) for c in chunks]
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

#### Step 1: Query Expansion

LLM generates 3-4 paraphrases:

```python
expansion_prompt = f"""
You are an expert search query optimizer.
Given the original query, generate 3-4 different paraphrased versions
that cover the same intent but use different wording or synonyms.
Original Query: {query}

Provide the paraphrases as a bulleted list.
"""
```

Original query is always included: `all_queries = [query] + expanded_queries`

#### Step 2: Multi-Query Retrieval

For each query variant:

```python
for q in all_queries:
    query_vec = self.agent.get_embedding(q)
    search_result = self.client.query_points(
        collection_name=self.collection_name,
        query=query_vec,
        limit=top_k,
    )
    for hit in search_result.points:
        text = hit.payload["text"]
        score = hit.score
        if text not in all_contexts or score > all_contexts[text]:
            all_contexts[text] = score  # Keep best score
```

#### Step 3: Merge and Rank

```python
sorted_texts = sorted(all_contexts.items(), key=lambda x: x[1], reverse=True)
return [text for text, _ in sorted_texts[:top_k]]
```

---

## Key Parameters

| Parameter         | Description                                        |
| ----------------- | -------------------------------------------------- |
| `top_k`           | Number of final results to return                  |
| `collection_name` | Qdrant collection name                             |
| `agent`           | BaseAgent for embeddings and LLM paraphrasing      |

---

## Why This Approach Works

* **Increases recall**: Finds documents that don't match exact wording
* **Vocabulary mismatch reduction**: Captures synonyms and alternate phrasings
* **Score-based ranking**: Best match across all variants wins
* **Original query preserved**: Ensures direct matches aren't missed

---

## Limitations

* Paraphrase quality depends on LLM capabilities
* Multiple embeddings per query increases latency
* No frequency weighting (documents appearing across multiple paraphrases)
* May introduce noise from low-quality paraphrases

---

## Possible Improvements

* **Batch retrieval** for parallel query variant processing
* **Frequency boosting**: Increase score for documents retrieved by multiple variants
* **Paraphrase filtering**: Remove low-quality or off-topic expansions
* **Re-ranking step**: Apply cross-encoder or LLM re-ranking on merged results

---

## Summary

`SemanticQueryExpansionRetriever`:

* Expands queries into 3-4 LLM-generated paraphrases
* Retrieves top-K for each variant independently
* Merges results, keeping highest score per document
* Returns top-K documents sorted by similarity score

Best suited for **semantic-rich queries** where users may phrase questions in multiple ways.

---
