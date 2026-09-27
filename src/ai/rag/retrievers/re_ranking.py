import logging
import re
from typing import Optional
from qdrant_client.models import Filter

from src.ai.agents.base_agent import BaseAgent
from src.ai.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class ContextReRankingRetriever(Retriever):
    """
    Retrieves a larger set of candidates using dense search, then uses
    the LLM to re-rank them based on explicit relevance.
    """

    def __init__(self, agent: BaseAgent, collection_name: str = "rag_collection"):
        super().__init__(agent, collection_name=collection_name)

    def retrieve(self, query: str, top_k: int = 3, query_filter: Optional[Filter] = None) -> list[str]:
        if not self.has_index:
            logger.warning("ContextReRankingRetriever.retrieve called before indexing.")
            return []

        # 1. Broad retrieval — fetch 3x candidates
        query_vec = self.agent.get_embedding(query)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k * 3,
        )
        candidates = [hit.payload["text"] for hit in search_result.points]

        if not candidates:
            return []

        # 2. LLM Re-ranking
        rerank_prompt = f"""Given the following query and a list of context chunks, re-order the chunks based on their relevance to the query.
Return the indices of the top {top_k} chunks in order of relevance, separated by commas.

Rules:
- Return ONLY the indices (e.g., 2, 0, 5).
- Provide indices as integers between 0 and {len(candidates) - 1}.
- Do not repeat an index or include any extra text.

Query: {query}

Chunks:
{chr(10).join([f"[{i}] {chunk}" for i, chunk in enumerate(candidates)])}

Top {top_k} Indices:"""

        response = self.agent.generate_response(rerank_prompt)

        try:
            # Extract valid indices
            extracted_indices = [
                int(idx)
                for idx in re.findall(r'\d+', response)
                if int(idx) < len(candidates)
            ]

            # Deduplicate
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

            return [candidates[i] for i in final_indices]

        except Exception:
            logger.warning("ContextReRankingRetriever: re-ranking parse failed, falling back to vector order.")
            return candidates[:top_k]