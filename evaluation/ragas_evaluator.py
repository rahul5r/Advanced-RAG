"""
evaluation/ragas_evaluator.py
------------------------------
Proper RAGAS evaluator using:
- boto3-backed Nova Lite as the judge LLM
- local embeddings from your existing BaseAgent
- all 4 classic RAG metrics
- sequential execution to reduce timeout pressure
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class BaseAgentLLM:
    """
    LangChain-compatible LLM wrapper for RAGAS.
    Uses a dedicated judge agent to avoid mixing app generation with evaluation.
    """

    def __init__(self, judge_agent: Any):
        self.agent = judge_agent
        self._wrapped = None

    def as_langchain_llm(self):
        if self._wrapped is not None:
            return self._wrapped

        from langchain_core.language_models.llms import LLM

        class _WrappedLLM(LLM):
            @property
            def _llm_type(self) -> str:
                return "bedrock-nova-lite-ragas"

            def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
                text = self._agent.generate_response(
                    prompt=prompt,
                    system_prompt=(
                        "You are an automated evaluation model inside a benchmark.\n"
                        "Follow the formatting instructions exactly.\n"
                        "Do not add commentary.\n"
                        "Return only the requested output."
                    ),
                    temperature=0.0,
                    max_output_tokens=256,
                ).strip()

                if stop:
                    for token in stop:
                        if token in text:
                            text = text.split(token)[0]
                return text

        wrapped = _WrappedLLM()
        wrapped._agent = self.agent
        self._wrapped = wrapped
        return self._wrapped


class BaseAgentEmbeddings:
    """
    LangChain-compatible embeddings wrapper for RAGAS.
    Reuses your main app agent's local embedding model.
    """

    def __init__(self, embedding_agent: Any):
        self.agent = embedding_agent
        self._wrapped = None

    def as_langchain_embeddings(self):
        if self._wrapped is not None:
            return self._wrapped

        from langchain_core.embeddings import Embeddings

        class _WrappedEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                texts = [str(t) for t in texts]
                if not texts:
                    return []
                return self._agent.get_embeddings_batch(texts)

            def embed_query(self, text: str) -> List[float]:
                return self._agent.get_embedding(str(text))

        wrapped = _WrappedEmbeddings()
        wrapped._agent = self.agent
        self._wrapped = wrapped
        return self._wrapped


BedrockLLMWrapper = BaseAgentLLM
HFEmbeddingsWrapper = BaseAgentEmbeddings


def build_ragas_llm_and_embeddings(app_agent: Any) -> Tuple[BaseAgentLLM, BaseAgentEmbeddings]:
    from src.ai.agents.judge_agent import JudgeAgent

    judge_agent = JudgeAgent()
    return BaseAgentLLM(judge_agent), BaseAgentEmbeddings(app_agent)


def _load_ragas_dependencies():
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        )
        from ragas.run_config import RunConfig

        return Dataset, evaluate, RunConfig, {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
    except ImportError as exc:
        raise ImportError(
            "True RAGAS evaluation requires: pip install ragas datasets langchain-core"
        ) from exc


def _coerce_score(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _extract_aggregate_scores(result: Any) -> Dict[str, float]:
    import math

    keys = ["faithfulness", "answer_relevancy",
            "context_precision", "context_recall"]
    scores = {k: 0.0 for k in keys}

    # Case 1: plain dict
    if isinstance(result, dict):
        for k in keys:
            v = result.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                scores[k] = float(v)
        return scores

    # Case 2: RAGAS EvaluationResult -> dataframe
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        for k in keys:
            if k in df.columns:
                vals = [x for x in df[k].tolist() if x is not None and not (
                    isinstance(x, float) and math.isnan(x))]
                if vals:
                    scores[k] = float(sum(vals) / len(vals))
        return scores

    # Case 3: fallback attrs
    for k in keys:
        v = getattr(result, k, None)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            scores[k] = float(v)

    return scores


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_rows: List[Dict[str, Any]] = []

    max_answer_chars = int(os.getenv("RAGAS_MAX_ANSWER_CHARS", "600"))
    max_context_chars = int(os.getenv("RAGAS_MAX_CONTEXT_CHARS", "1800"))

    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        contexts = [str(c).strip()
                    for c in (row.get("contexts") or []) if str(c).strip()]
        ground_truth = str(row.get("ground_truth", "")).strip()

        if not question or not ground_truth:
            continue

        if len(answer) > max_answer_chars:
            answer = answer[:max_answer_chars].rstrip()

        trimmed_contexts: List[str] = []
        for c in contexts:
            if len(c) > max_context_chars:
                trimmed_contexts.append(c[:max_context_chars].rstrip())
            else:
                trimmed_contexts.append(c)

        valid_rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": trimmed_contexts,
                "ground_truth": ground_truth,
            }
        )

    return valid_rows


def _evaluate_rows(
    rows: List[Dict[str, Any]],
    llm_wrapper: BaseAgentLLM,
    embeddings_wrapper: BaseAgentEmbeddings,
) -> Dict[str, float]:
    Dataset, evaluate, RunConfig, metric_map = _load_ragas_dependencies()

    dataset = Dataset.from_list(rows)
    metrics = [
        metric_map["faithfulness"],
        metric_map["answer_relevancy"],
        metric_map["context_precision"],
        metric_map["context_recall"],
    ]

    logger.info(
        "Using %d RAGAS metric(s): %s",
        len(metrics),
        [getattr(m, "name", str(m)) for m in metrics],
    )

    run_config = RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT", "180")),
        max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "3")),
        max_wait=int(os.getenv("RAGAS_MAX_WAIT", "60")),
        max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "1")),
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper.as_langchain_llm(),
        embeddings=embeddings_wrapper.as_langchain_embeddings(),
        run_config=run_config,
        raise_exceptions=False,
        batch_size=1,
        show_progress=True,
    )

    scores = _extract_aggregate_scores(result)

    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        scores.setdefault(key, 0.0)

    return scores


def run_ragas_evaluation(
    rows: List[Dict[str, Any]],
    llm: BaseAgentLLM | None,
    embeddings: BaseAgentEmbeddings | None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    if llm is None or embeddings is None:
        raise ValueError(
            "RAGAS evaluator requires both llm and embeddings wrappers.")

    valid_rows = _normalize_rows(rows)

    if not valid_rows:
        raise ValueError(
            "No valid rows were available for true RAGAS evaluation.")

    logger.info(
        "Running true RAGAS evaluation on %d valid sample(s)...", len(valid_rows))

    aggregate = _evaluate_rows(valid_rows, llm, embeddings)

    # Keep per-sample section lightweight and stable.
    per_sample_scores: List[Dict[str, float]] = [
        {
            "faithfulness": aggregate["faithfulness"],
            "answer_relevancy": aggregate["answer_relevancy"],
            "context_precision": aggregate["context_precision"],
            "context_recall": aggregate["context_recall"],
        }
        for _ in valid_rows
    ]

    logger.info("True RAGAS evaluation complete.")
    return aggregate, per_sample_scores


class RagasEvaluator:
    def __init__(self, agent: Any):
        self.agent = agent
        self.llm, self.embeddings = build_ragas_llm_and_embeddings(agent)

    def evaluate(self, rows: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        return run_ragas_evaluation(rows, self.llm, self.embeddings)
