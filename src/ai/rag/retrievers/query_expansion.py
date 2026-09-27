import logging
from typing import Optional
from qdrant_client.models import Filter

from src.ai.agents.base_agent import BaseAgent
from src.ai.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class SemanticQueryExpansionRetriever(Retriever):
    """
    Expands the original query into several paraphrases to increase
    recall, then retrieves and merges results using a max-score strategy.
    """

    def __init__(self, agent: BaseAgent, collection_name: str = "rag_collection"):
        super().__init__(agent, collection_name=collection_name)

    def retrieve(self, query: str, top_k: int = 3, query_filter: Optional[Filter] = None) -> list[str]:
        if not self.has_index:
            logger.warning("SemanticQueryExpansionRetriever.retrieve called before indexing.")
            return []

        # 1. Expand query into paraphrases
        expansion_prompt = f"""
You are an expert search query optimizer.
Given the original query, generate 3-4 different paraphrased versions that cover the same intent but use different wording or synonyms.
Original Query: {query}

Provide the paraphrases as a bulleted list.
- paraphrase 1
- paraphrase 2
"""
        response = self.agent.generate_response(expansion_prompt)
        expanded_queries = [
            line.lstrip("- ").strip()
            for line in response.splitlines()
            if line.strip().startswith("-")
        ]

        # Always include original query
        all_queries = [query] + expanded_queries

        # 2. Retrieve for all queries, keeping highest score per chunk
        all_contexts: dict[str, float] = {}
        for q in all_queries:
            query_vec = self.agent.get_embedding(q)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                query_filter=query_filter,
                limit=top_k,
            )
            for hit in search_result.points:
                text = hit.payload["text"]
                score = float(hit.score)
                if text not in all_contexts or score > all_contexts[text]:
                    all_contexts[text] = score

        # 3. Sort by best score and return top_k
        sorted_texts = sorted(all_contexts.items(), key=lambda x: x[1], reverse=True)
        return [text for text, _ in sorted_texts[:top_k]]