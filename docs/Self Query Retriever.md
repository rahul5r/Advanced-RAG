# SelfQueryRetriever Documentation

## Overview

`SelfQueryRetriever` extracts **metadata filters** from natural language queries using an LLM, then retrieves documents from Qdrant with both **semantic similarity** and **metadata constraints**.

Key features:

* **JSON-based filter extraction** with robust parsing
* Supports filters: `document_type`, `date`, `author`
* **Caches filterable-metadata check** to avoid repeated Qdrant scrolls
* Graceful fallback when no filter is detected

---

## Architecture

```text
            User Query
                 │
                 ▼
         ┌────────────────┐
         │ LLM Filterizer │
         └────────────────┘
                 │
          ┌──────┴──────┐
          │             │
   JSON Filter      Query Embedding
   (or null)            │
          │             │
          │      ┌────────────────┐
          │      │ Qdrant Search  │
          │      │ (semantic +    │
          │      │  metadata)     │
          │      └────────────────┘
          │             │
          └──────┬──────┘
                 ▼
          Top-K Document Texts
                 │
                 ▼
           LLM Generation
```

---

## How It Works

### 1. Index Building (`build_index`)

* Generates embeddings via `BaseAgent.get_embedding()`
* Creates Qdrant collection with COSINE distance
* Stores points with payload: `{"text": chunk, "chunk_id": i, "document_type": "pdf"}`
* Uses `wait=True` to ensure persistence
* Resets filterable-metadata cache for fresh check

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

#### Step 1: Check Filterable Metadata (Cached)

```python
def _collection_has_filterable_metadata(self) -> bool:
    if self._has_filterable_metadata is not None:
        return self._has_filterable_metadata  # Cached

    records, _ = self.client.scroll(
        collection_name=self.collection_name,
        with_payload=True,
        with_vectors=False,
        limit=5,
    )
    self._has_filterable_metadata = any(
        field in (point.payload or {})
        for point in records
        for field in self.SUPPORTED_FIELDS
    )
    return self._has_filterable_metadata
```

* Result cached after first call (schema is static after `build_index`)
* Avoids repeated `scroll()` on every `retrieve()` call

#### Step 2: LLM Extracts Metadata Filter

```python
filter_prompt = f"""Extract a metadata filter from the query only if it is explicit.

Supported fields:
- document_type
- date
- author

Return JSON only:
{{"key": "<field_name>", "value": "<field_value>"}}

If no metadata filter is clearly requested, return exactly:
null

Query: {query}
"""
raw = self.agent.generate_response(filter_prompt, temperature=0.0).strip()
```

**Example outputs:**

| User Query                         | LLM Output                        |
| ---------------------------------- | --------------------------------- |
| `"Find emails from Alice in 2023"` | `{"key": "author", "value": "Alice"}` |
| `"Show me reports by Bob"`         | `{"key": "document_type", "value": "report"}` |
| `"Recent documents"`               | `null`                            |

#### Step 3: Parse LLM Response

```python
# Strip markdown code fences
fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
if fenced:
    raw = fenced.group(1).strip()

# Handle null responses
if raw.lower() == "null":
    return None

# Extract first JSON object
first_json = re.search(r"\{.*\}", raw, flags=re.DOTALL)
if first_json:
    raw = first_json.group(0).strip()

try:
    parsed = json.loads(raw)
    key = str(parsed.get("key", "")).strip()
    value = str(parsed.get("value", "")).strip()
    if key in self.SUPPORTED_FIELDS and value:
        return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])
except Exception:
    logger.warning("SelfQueryRetriever: failed to parse filter: %r", raw)

return None
```

#### Step 4: Query Qdrant with Filter

```python
query_vec = self.agent.get_embedding(query)
search_result = self.client.query_points(
    collection_name=self.collection_name,
    query=query_vec,
    query_filter=query_filter,  # None if no filter extracted
    limit=top_k,
)
return [hit.payload["text"] for hit in search_result.points]
```

---

## Key Parameters

| Parameter         | Description                                       |
| ----------------- | ------------------------------------------------- |
| `query`           | User query to search                              |
| `top_k`           | Number of results to return (default: 3)          |
| `collection_name` | Qdrant collection containing document vectors     |
| `agent`           | BaseAgent for embeddings and filter extraction    |

---

## Supported Metadata Fields

| Field           | Example Values                    |
| --------------- | --------------------------------- |
| `document_type` | `"pdf"`, `"email"`, `"report"`    |
| `date`          | `"2023"`, `"2024-01-15"`          |
| `author`        | `"Alice"`, `"Bob"`                |

---

## Why This Approach Works

* **LLM understanding** dynamically generates filters from natural language
* **Cached metadata check** avoids unnecessary Qdrant operations
* **Robust JSON parsing** handles varied LLM output formats
* **Combines semantics + constraints** for precise retrieval

---

## Limitations

* Only handles **single filter** per query (no AND/OR combinations)
* Assumes metadata exists in document payloads
* LLM filter extraction is not 100% reliable
* No range filter support (e.g., `date >= 2023-01-01`)

---

## Possible Improvements

* Support **multiple filters** with boolean logic
* Add **range filter** parsing for dates and numbers
* Validate filter values against known payload values
* Cache filter results for repeated queries

---

## Summary

`SelfQueryRetriever`:

* Uses LLM to extract JSON metadata filters from queries
* Caches filterable-metadata check for efficiency
* Retrieves from Qdrant with semantic + metadata filtering
* Returns top-K documents for RAG generation

Ideal for RAG pipelines with **structured document corpora** where metadata-aware retrieval improves precision.

---
