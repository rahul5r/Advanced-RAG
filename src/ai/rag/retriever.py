import os
import uuid
import logging
import json
import pickle
import re
from pathlib import Path
from typing import Callable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from src.ai.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, agent: BaseAgent, collection_name: str = "rag_collection"):
        self.agent = agent
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        # Local artifacts for BM25
        self.artifacts_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "retriever_cache"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_path = self.artifacts_dir / f"{self.collection_name}_chunks.json"
        self._bm25_path = self.artifacts_dir / f"{self.collection_name}_bm25.pkl"

    @property
    def has_index(self) -> bool:
        try:
            return self.client.collection_exists(collection_name=self.collection_name)
        except Exception:
            return False

    def build_index(
        self, 
        chunks: list[dict] | list[str], 
        on_progress: Optional[Callable[[str], None]] = None,
        source: str = "unknown",
        date: str = "unknown"
    ) -> None:
        """
        Unified indexing process:
        1. Generates dense embeddings (Cloud).
        2. Generates BM25 index (Local).
        3. Saves BM25 artifacts locally.
        4. Upserts dense vectors to Qdrant.
        """
        if not chunks:
            return

        # Canonicalize chunks to list[dict]
        processed_chunks: list[dict] = []
        for i, c in enumerate(chunks):
            if isinstance(c, str):
                processed_chunks.append({
                    "text": c, 
                    "page_no": 1, 
                    "source": source, 
                    "date": date,
                    "chunk_id": i
                })
            else:
                c["chunk_id"] = i
                c.setdefault("source", source)
                c.setdefault("date", date)
                c.setdefault("page_no", 1)
                processed_chunks.append(c)

        # 1. Local BM25 Generation
        if on_progress:
            on_progress("Generating BM25 index (Local)...")
        
        tokenized_corpus = [re.findall(r"\b\w+\b", c["text"].lower()) for c in processed_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        if on_progress:
            on_progress("Saving BM25 artifacts locally...")
        
        self._chunks_path.write_text(json.dumps(processed_chunks, ensure_ascii=False), encoding="utf-8")
        with open(self._bm25_path, "wb") as f:
            pickle.dump(bm25, f)

        # 2. Dense Embedding Generation
        if on_progress:
            on_progress("Generating Dense Embeddings (Cloud)...")
        
        embeddings = [self.agent.get_embedding(c["text"]) for c in processed_chunks]
        vector_size = len(embeddings[0])

        # 3. Qdrant Configuration
        if on_progress:
            on_progress("Configuring Qdrant Collection (Cloud)...")

        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        # 4. Upserting
        if on_progress:
            on_progress("Upserting Dense Vectors to Cloud...")

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=c,
            )
            for c, vector in zip(processed_chunks, embeddings)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )
        
        if on_progress:
            on_progress("Indexing complete!")

    def retrieve(self, query: str, top_k: int = 3, query_filter: Optional[Filter] = None) -> list[str]:
        """Base retrieval uses dense semantic search."""
        if not self.has_index:
            return []

        query_vec = self.agent.get_embedding(query)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k,
        )
        return [hit.payload["text"] for hit in search_result.points]