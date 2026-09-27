"""
evaluation/pipeline.py
-----------------------
Core evaluation orchestrator.

What changed:
- auto-builds indexes from source_path when missing
- avoids NaN propagation
- keeps per-sample errors isolated
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    latency_ms: float
    error: Optional[str] = None
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None


@dataclass
class StrategyResult:
    strategy_key: str
    strategy_label: str

    faithfulness: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0

    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    avg_generation_latency_ms: float = 0.0

    samples: List[SampleResult] = field(default_factory=list)
    passed: bool = False
    error: Optional[str] = None

    def apply_thresholds(
        self,
        faithfulness_threshold: float,
        context_recall_threshold: float,
    ) -> None:
        self.passed = (
            self.faithfulness >= faithfulness_threshold
            and self.context_recall >= context_recall_threshold
        )


def _build_retriever(strategy_key: str, agent):
    from src.ai.rag.retrievers import (
        ContextReRankingRetriever,
        HybridSearchRetriever,
        MultiHopRetriever,
        SemanticQueryExpansionRetriever,
        SelfQueryRetriever,
    )

    mapping = {
        "hybrid": lambda: HybridSearchRetriever(agent),
        "expansion": lambda: SemanticQueryExpansionRetriever(agent),
        "rerank": lambda: ContextReRankingRetriever(agent),
        "self_query": lambda: SelfQueryRetriever(agent),
        "multi_hop": lambda: MultiHopRetriever(agent),
    }

    if strategy_key not in mapping:
        raise ValueError(
            f"Unknown strategy: {strategy_key}. Valid: {list(mapping.keys())}")

    return mapping[strategy_key]()


class EvaluationPipeline:
    def __init__(self, config):
        self.config = config
        self._agent = None
        self._generator = None
        self._ragas_llm = None
        self._ragas_embeddings = None

    def _get_agent(self):
        if self._agent is None:
            from src.ai.agents.base_agent import BaseAgent

            self._agent = BaseAgent(
                model_name=self.config.model_name,
                embedding_model=self.config.embedding_model,
            )
            logger.info(
                "Shared BaseAgent ready — model=%s | embeddings=%s",
                self.config.model_name,
                self.config.embedding_model,
            )
        return self._agent

    def _get_generator(self):
        if self._generator is None:
            from src.ai.rag.generator import Generator

            self._generator = Generator(self._get_agent())
        return self._generator

    def _get_ragas_components(self):
        if self._ragas_llm is None:
            from evaluation.ragas_evaluator import build_ragas_llm_and_embeddings

            self._ragas_llm, self._ragas_embeddings = build_ragas_llm_and_embeddings(
                self._get_agent())
        return self._ragas_llm, self._ragas_embeddings

    def _resolve_source_pdf(self, test_samples) -> Optional[str]:
        if not test_samples:
            return None

        for sample in test_samples:
            source_path = getattr(sample, "source_path", None) or getattr(
                sample, "source_file", None)
            if not source_path:
                continue

            path = Path(source_path)
            if path.exists():
                return str(path)

            guessed = Path("data/pdfs") / Path(source_path).name
            if guessed.exists():
                return str(guessed)

        return None

    def _build_index_from_source(self, retriever, source_pdf: str) -> None:
        from src.ai.rag.chunker import Chunker
        from src.ai.rag.document_processor import DocumentProcessor

        logger.info("Building missing index for %s from %s",
                    retriever.__class__.__name__, source_pdf)
        processor = DocumentProcessor()
        raw_text = processor.process(source_pdf)
        chunks = Chunker().split_text(raw_text)
        chunks = [c.strip() for c in chunks if c and c.strip()]
        retriever.build_index(chunks)
        logger.info("Index built with %d chunk(s).", len(chunks))

    def _evaluate_strategy(self, strategy_key: str, test_samples) -> StrategyResult:
        from evaluation.config import RETRIEVER_REGISTRY
        from evaluation.ragas_evaluator import run_ragas_evaluation

        label = RETRIEVER_REGISTRY.get(strategy_key, strategy_key)
        logger.info("")
        logger.info("=== Evaluating: %-12s (%s) ===", strategy_key, label)

        result = StrategyResult(
            strategy_key=strategy_key, strategy_label=label)
        agent = self._get_agent()
        generator = self._get_generator()

        try:
            retriever = _build_retriever(strategy_key, agent)
        except Exception as exc:
            result.error = f"Retriever init failed: {exc}"
            logger.error(result.error)
            return result

        if not retriever.has_index:
            source_pdf = self._resolve_source_pdf(test_samples)
            if source_pdf:
                try:
                    self._build_index_from_source(retriever, source_pdf)
                except Exception as exc:
                    result.error = f"Index build failed: {exc}"
                    logger.error(result.error)
                    return result

        if not retriever.has_index:
            result.error = "Index not found for this retriever and auto-build was not possible."
            logger.error(result.error)
            return result

        raw_results: List[Dict[str, Any]] = []
        sample_results: List[SampleResult] = []

        for i, sample in enumerate(test_samples, 1):
            logger.info("  [%d/%d] %s", i, len(test_samples),
                        sample.question[:100])

            contexts: List[str] = []
            answer = ""
            retrieval_latency_ms = 0.0
            generation_latency_ms = 0.0
            error = None

            try:
                t0 = time.perf_counter()
                contexts = retriever.retrieve(
                    sample.question, top_k=self.config.top_k)
                retrieval_latency_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                answer = generator.generate_answer(sample.question, contexts)
                generation_latency_ms = (time.perf_counter() - t1) * 1000.0

            except Exception as exc:
                error = str(exc)
                logger.warning("    x Error on sample %d: %s", i, exc)

            total_latency_ms = retrieval_latency_ms + generation_latency_ms

            sample_result = SampleResult(
                question=sample.question,
                answer=answer,
                contexts=contexts,
                ground_truth=sample.ground_truth,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                latency_ms=total_latency_ms,
                error=error,
            )
            sample_results.append(sample_result)

            if not error and contexts and answer.strip():
                raw_results.append(
                    {
                        "question": sample.question,
                        "answer": answer,
                        "contexts": contexts,
                        "ground_truth": sample.ground_truth,
                    }
                )

        result.samples = sample_results

        latencies = [s.latency_ms for s in sample_results]
        retrieval_latencies = [s.retrieval_latency_ms for s in sample_results]
        generation_latencies = [
            s.generation_latency_ms for s in sample_results]

        result.total_latency_ms = sum(latencies)
        result.avg_latency_ms = result.total_latency_ms / \
            max(len(latencies), 1)
        result.avg_retrieval_latency_ms = sum(
            retrieval_latencies) / max(len(retrieval_latencies), 1)
        result.avg_generation_latency_ms = sum(
            generation_latencies) / max(len(generation_latencies), 1)

        if not raw_results:
            result.error = "No valid samples were available for scoring."
            logger.error(result.error)
            return result

        try:
            ragas_llm, ragas_emb = self._get_ragas_components()
            aggregate, per_sample_scores = run_ragas_evaluation(
                raw_results, ragas_llm, ragas_emb)
        except Exception as exc:
            result.error = f"Evaluation failed: {exc}"
            logger.error(result.error)
            return result

        result.faithfulness = float(aggregate.get("faithfulness", 0.0) or 0.0)
        result.context_recall = float(
            aggregate.get("context_recall", 0.0) or 0.0)
        result.answer_relevancy = float(
            aggregate.get("answer_relevancy", 0.0) or 0.0)
        result.context_precision = float(
            aggregate.get("context_precision", 0.0) or 0.0)

        successful_samples = [
            s for s in sample_results if not s.error and s.contexts and s.answer.strip()]
        for sr, ps in zip(successful_samples, per_sample_scores):
            sr.faithfulness = float(ps.get("faithfulness", 0.0) or 0.0)
            sr.context_recall = float(ps.get("context_recall", 0.0) or 0.0)
            sr.answer_relevancy = float(ps.get("answer_relevancy", 0.0) or 0.0)
            sr.context_precision = float(
                ps.get("context_precision", 0.0) or 0.0)

        result.apply_thresholds(
            self.config.faithfulness_threshold,
            self.config.context_recall_threshold,
        )

        logger.info(
            "  -> Faithfulness=%.4f | CtxRecall=%.4f | AnsRelevancy=%.4f | "
            "CtxPrecision=%.4f | AvgRetrieve=%.0f ms | AvgGenerate=%.0f ms | AvgTotal=%.0f ms | %s",
            result.faithfulness,
            result.context_recall,
            result.answer_relevancy,
            result.context_precision,
            result.avg_retrieval_latency_ms,
            result.avg_generation_latency_ms,
            result.avg_latency_ms,
            "PASS" if result.passed else "FAIL",
        )
        return result

    def run(self, test_samples) -> List[StrategyResult]:
        logger.info(
            "Starting evaluation | strategies=%s | samples=%d | top_k=%d",
            self.config.strategies,
            len(test_samples),
            self.config.top_k,
        )

        all_results: List[StrategyResult] = []
        for strategy_key in self.config.strategies:
            all_results.append(self._evaluate_strategy(
                strategy_key, test_samples))
        return all_results
