import logging
import json
import pickle
import re
from pathlib import Path
from typing import Optional
from qdrant_client.models import Filter

from src.ai.agents.base_agent import BaseAgent
from src.ai.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class HybridSearchRetriever(Retriever):
    """
    Combines dense semantic search (Qdrant) with sparse keyword search (BM25)
    using local artifacts and weighted fusion.
    """

    def __init__(self, agent: BaseAgent, collection_name: str = "rag_collection"):
        super().__init__(agent, collection_name=collection_name)
        self._bm25 = None
        self._chunks = None

    def _load_artifacts(self) -> bool:
        """Loads local BM25 and chunks if available. Uses lazy loading."""
        if self._bm25 is not None and self._chunks is not None:
            return True

        if not self._bm25_path.exists() or not self._chunks_path.exists():
            logger.warning(
                "HybridSearchRetriever: Artifacts missing at: %s or %s",
                self._bm25_path.absolute(),
                self._chunks_path.absolute()
            )
            return False
        
        try:
            with open(self._bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
            self._chunks = json.loads(self._chunks_path.read_text(encoding="utf-8"))
            logger.info("HybridSearchRetriever: Successfully loaded BM25 artifacts.")
            return True
        except Exception as e:
            logger.error("Failed to load BM25 artifacts from %s: %s", self._bm25_path, e)
            return False

    def retrieve(self, query: str, top_k: int = 3, query_filter: Optional[Filter] = None) -> list[str]:
        if not self.has_index:
            logger.warning("HybridSearchRetriever: Index not found.")
            return []

        if not self._load_artifacts():
            logger.warning("HybridSearchRetriever: Falling back to dense-only retrieval.")
            return super().retrieve(query, top_k=top_k, query_filter=query_filter)

        # 1. Semantic Search (Dense)
        query_vec = self.agent.get_embedding(query)
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k * 3,
        ).points
        
        dense_scores = {hit.payload["text"]: float(hit.score) for hit in dense_results}

        # 2. Keyword Search (BM25)
        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        bm25_scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k*3 from BM25
        # chunks can be list[str] or list[dict]
        bm25_results = sorted(
            zip(self._chunks, bm25_scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k * 3]
        
        bm25_map = {}
        for item, score in bm25_results:
            if score > 0:
                text = item["text"] if isinstance(item, dict) else item
                bm25_map[text] = score

        # 3. Normalized Fusion
        all_candidates = set(dense_scores.keys()) | set(bm25_map.keys())
        
        def normalize(scores):
            if not scores: return {}
            min_s = min(scores.values())
            max_s = max(scores.values())
            if max_s == min_s: return {k: 1.0 for k in scores}
            return {k: (s - min_s) / (max_s - min_s) for k, s in scores.items()}

        n_dense = normalize(dense_scores)
        n_bm25 = normalize(bm25_map)

        fused = {}
        for doc in all_candidates:
            # Weighted average: 60% semantic, 40% keyword
            fused[doc] = 0.6 * n_dense.get(doc, 0) + 0.4 * n_bm25.get(doc, 0)
        
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]