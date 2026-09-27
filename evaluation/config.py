"""
evaluation/config.py
--------------------
Central configuration for the RAGAS evaluation pipeline.
All thresholds, retriever names, paths, and runtime flags live here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Retriever registry  (key used in CLI + reports, value is the display label)
# ---------------------------------------------------------------------------

RETRIEVER_REGISTRY: dict[str, str] = {
    "hybrid":     "Hybrid Search (Qdrant + BM25)",
    "expansion":  "Semantic Query Expansion",
    "rerank":     "Context Re-Ranking",
    "self_query": "Self-Query Retriever",
    "multi_hop":  "Multi-Hop Retriever",
}

ALL_STRATEGIES = list(RETRIEVER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """
    Holds every knob for a single evaluation run.
    Fields resolve from .env first, then fall back to the defaults here.
    CLI arguments in run_evaluation.py override everything.
    """

    # ── Which retrievers to evaluate ─────────────────────────────────────
    strategies: List[str] = field(
        default_factory=lambda: list(RETRIEVER_REGISTRY.keys())
    )

    # ── Retrieval params ─────────────────────────────────────────────────
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", 5))
    )

    # ── Bedrock model / HF embedding ─────────────────────────────────────
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "amazon.nova-lite-v1:0")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_collection: str = "rag_collection"

    # ── RAGAS pass/fail thresholds ────────────────────────────────────────
    faithfulness_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", 0.80)
        )
    )
    context_recall_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("RAGAS_CONTEXT_RECALL_THRESHOLD", 0.70)
        )
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    pdf_path: str = "data/pdfs/Company-Policy.pdf"
    test_file: str = "evaluation/data/testset.json"
    report_dir: str = "evaluation/reports"

    # ── TestsetGenerator params ───────────────────────────────────────────
    testset_size: int = 20

    # ── Runtime flags ─────────────────────────────────────────────────────
    verbose: bool = False
    save_per_sample: bool = True

    def validate(self) -> None:
        """Raise on obviously bad inputs."""
        unknown = set(self.strategies) - set(ALL_STRATEGIES)
        if unknown:
            raise ValueError(
                f"Unknown strategies: {unknown}. "
                f"Valid options: {ALL_STRATEGIES}"
            )
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not os.getenv("AWS_API_KEY"):
            raise EnvironmentError(
                "AWS_API_KEY is not set. Add it to your .env file."
            )
        if not os.getenv("HF_TOKEN"):
            raise EnvironmentError(
                "HF_TOKEN is not set. Add it to your .env file."
            )
