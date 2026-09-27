import json
import logging
import re
from typing import Optional

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from src.ai.agents.base_agent import BaseAgent
from src.ai.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class SelfQueryRetriever(Retriever):
    """
    Parses the user query to extract metadata filters (e.g. for page number)
    and applies them to the dense search in Qdrant.
    """
    SUPPORTED_FIELDS = {"page_no"}

    def __init__(self, agent: BaseAgent, collection_name: str = "rag_collection"):
        super().__init__(agent, collection_name=collection_name)
        self._has_filterable_metadata: bool | None = None

    def _collection_has_filterable_metadata(self) -> bool:
        """Cache-backed check for filterable payloads in the current collection."""
        if self._has_filterable_metadata is not None:
            return self._has_filterable_metadata

        if not self.has_index:
            return False

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=5,
        )
        self._has_filterable_metadata = any(
            field in (point.payload or {})
            for point in records
            for field in self.SUPPORTED_FIELDS
        )
        return self._has_filterable_metadata

    def _extract_filter(self, query: str) -> Filter | None:
        """Uses LLM to extract structured metadata filters from natural language."""
        filter_prompt = f"""Extract a page number filter from the query only if it is explicit (e.g. "on page 5", "from p. 2").

Supported fields:
- page_no (Integer)

Return JSON only:
{{"key": "page_no", "value": <integer_page_number>}}

If no page number is clearly requested, return exactly:
null

Query: {query}
"""
        raw = self.agent.generate_response(filter_prompt, temperature=0.0).strip()

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()

        if raw.lower().startswith("null") or raw.lower() == "null":
            return None

        first_json = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if first_json:
            raw = first_json.group(0).strip()

        try:
            parsed = json.loads(raw)
            key = str(parsed.get("key", "")).strip()
            value = parsed.get("value")
            
            if key == "page_no" and value is not None:
                # Ensure value is an integer
                val_int = int(float(value)) 
                return Filter(must=[FieldCondition(key=key, match=MatchValue(value=val_int))])
        except Exception:
            logger.warning("SelfQueryRetriever: failed to parse filter: %s", raw)

        return None

    def retrieve(self, query: str, top_k: int = 3, query_filter: Optional[Filter] = None) -> list[str]:
        if not self.has_index:
            logger.warning("SelfQueryRetriever.retrieve called before indexing.")
            return []

        # Start with the optional filter passed in
        final_filter = query_filter
        print(final_filter)

        # Extract and merge/override with self-query filter
        if self._collection_has_filterable_metadata():
            extracted = self._extract_filter(query)
            if extracted:
                if final_filter:
                    # Merge filters (simplified: extracted takes precedence or merges)
                    final_filter.must.extend(extracted.must)
                else:
                    final_filter = extracted

        return super().retrieve(query, top_k=top_k, query_filter=final_filter)
