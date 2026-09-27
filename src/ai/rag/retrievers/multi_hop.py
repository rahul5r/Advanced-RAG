import logging
from typing import Optional
from qdrant_client.models import Filter

from src.ai.agents.base_agent import BaseAgent
from src.ai.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class MultiHopRetriever(Retriever):
    """
    Decomposes complex questions into sub-queries and retrieves context
    for each sub-query, then re-ranks globally against the original query.
    """

    def __init__(self, agent: BaseAgent, collection_name: str = "rag_collection"):
        super().__init__(agent, collection_name=collection_name)

    def retrieve(self, query: str, top_k: int = 3, query_filter: Optional[Filter] = None) -> list[str]:
        if not self.has_index:
            logger.warning("MultiHopRetriever.retrieve called before indexing.")
            return []

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

        # Always include the original query first so chunks that directly answer
        # it are never missed when decomposition drifts away from the core question.
        all_sub_queries = [query] + [sq for sq in sub_queries if sq and sq != query]

        # Embed the original query once and reuse
        original_query_vec = self.agent.get_embedding(query)

        merged_scores: dict[str, float] = {}
        for sq in all_sub_queries:
            query_vec = original_query_vec if sq == query else self.agent.get_embedding(
                sq)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                query_filter=query_filter,
                limit=top_k,
            )
            for hit in search_result.points:
                text = hit.payload["text"]
                score = float(hit.score)
                if text not in merged_scores or score > merged_scores[text]:
                    merged_scores[text] = score

        if not merged_scores:
            return []

        # Re-rank all merged candidates against the original query.
        # Uses top_k*3 to widen coverage beyond what each sub-query individually returned.
        reranked_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=original_query_vec,
            query_filter=query_filter,
            limit=max(top_k * 3, len(merged_scores)),
        ).points
        rerank_map = {hit.payload["text"]: float(hit.score) for hit in reranked_hits}

        final_scores = {
            text: (0.7 * merged_scores.get(text, 0.0)) + (0.3 * rerank_map.get(text, 0.0))
            for text in merged_scores
        }

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [text for text, _ in ranked[:top_k]]