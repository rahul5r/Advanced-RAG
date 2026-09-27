"""
evaluation/dataset_generator.py
---------------------------------
testset generation using the shared BaseAgent.

What changed:
- generates grounded QA pairs directly from the same processed chunks used by RAG
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _looks_bad_question(question: str) -> bool:
    q = question.strip()
    if len(q) < 12:
        return True
    if "?" not in q:
        return True
    lowered = q.lower()
    bad_patterns = [
        "how decoder work",
        "what is this document about exactly exactly",
        "can you provide information based on the provided context",
    ]
    return any(p in lowered for p in bad_patterns)


def _prepare_chunks(pdf_path: str) -> list[dict[str, Any]]:
    from src.ai.rag.chunker import Chunker
    from src.ai.rag.document_processor import DocumentProcessor

    processor = DocumentProcessor()
    raw_text = processor.process(pdf_path)
    chunker = Chunker()
    chunks = chunker.split_text(raw_text)

    prepared: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        cleaned = chunk.strip()
        if cleaned:
            prepared.append(
                {
                    "chunk_id": idx,
                    "text": cleaned,
                }
            )
    return prepared


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)

    if text.startswith("[") and text.endswith("]"):
        return json.loads(text)

    match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(
        f"Could not parse JSON array from model output: {text[:500]}")


def _generate_qas_for_chunk(agent, chunk_text: str, n_qas: int) -> list[dict[str, str]]:
    prompt = f"""
You are creating evaluation data for a RAG system.

Generate exactly {n_qas} high-quality question-answer pairs ONLY from the passage below.

Rules:
- Return ONLY a valid JSON array.
- Each item must have exactly these keys:
  - "question"
  - "ground_truth"
- Questions must be fully answerable from the passage alone.
- Questions must be specific, grammatically correct, and non-trivial.
- Do not use conversational slang.
- Do not ask about information not present in the passage.
- Do not mention "provided context" or "passage" in the question.
- Answers must be concise but complete.
- If the passage is too weak to create {n_qas} good pairs, still return fewer valid pairs in the same JSON array.

Passage:
\"\"\"
{chunk_text}
\"\"\"

Return only JSON.
""".strip()

    raw = agent.generate_response(prompt=prompt, temperature=0.2)
    rows = _extract_json_array(raw)

    cleaned_rows: list[dict[str, str]] = []
    for row in rows:
        q = _clean_text(str(row.get("question", "")))
        a = _clean_text(str(row.get("ground_truth", "")))
        if q and a:
            cleaned_rows.append({"question": q, "ground_truth": a})
    return cleaned_rows


def generate_testset(
    pdf_path: str,
    output_path: str,
    testset_size: int = 20,
    model_name: str = "amazon.nova-lite-v1:0",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict]:
    if not Path(pdf_path).exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Place your document at data/pdfs/Company-Policy.pdf or pass --pdf <path>."
        )

    logger.info("Loading PDF: %s", pdf_path)
    prepared_chunks = _prepare_chunks(pdf_path)
    logger.info("Prepared %d chunk(s) from %s", len(prepared_chunks), pdf_path)

    if len(prepared_chunks) < 3:
        logger.warning(
            "Only %d chunk(s) prepared. Testset quality may be limited.",
            len(prepared_chunks),
        )

    from src.ai.agents.base_agent import BaseAgent

    agent = BaseAgent(model_name=model_name, embedding_model=embedding_model)

    logger.info(
        "Using shared BaseAgent for dataset generation — model=%s | embeddings=%s",
        model_name,
        embedding_model,
    )

    rng = random.Random(42)
    candidate_chunks = prepared_chunks[:]
    rng.shuffle(candidate_chunks)

    source_hash = _file_sha256(pdf_path)
    final_samples: list[dict] = []
    seen_questions: set[str] = set()

    # Try enough chunks to fill the requested testset.
    for chunk in candidate_chunks:
        if len(final_samples) >= testset_size:
            break

        chunk_text = chunk["text"]
        if len(chunk_text) < 250:
            continue

        needed = min(2, testset_size - len(final_samples))

        try:
            qa_rows = _generate_qas_for_chunk(agent, chunk_text, needed)
        except Exception as exc:
            logger.warning(
                "Skipping chunk %s due to QA generation error: %s",
                chunk["chunk_id"],
                exc,
            )
            continue

        for row in qa_rows:
            question = row["question"]
            ground_truth = row["ground_truth"]

            if _looks_bad_question(question):
                continue

            dedupe_key = question.lower()
            if dedupe_key in seen_questions:
                continue
            seen_questions.add(dedupe_key)

            final_samples.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "source_file": Path(pdf_path).name,
                    "source_path": str(Path(pdf_path).as_posix()),
                    "source_sha256": source_hash,
                    "chunk_id": chunk["chunk_id"],
                }
            )

            if len(final_samples) >= testset_size:
                break

    if not final_samples:
        raise RuntimeError("No usable QA pairs were generated.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(final_samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Saved %d QA pairs to %s", len(final_samples), output_path)
    return final_samples


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m evaluation.dataset_generator",
        description="Generate a grounded evaluation test-set from a PDF.",
    )
    p.add_argument("--pdf", default="data/pdfs/Company-Policy.pdf",
                   help="Source PDF path.")
    p.add_argument("--size", type=int, default=20,
                   help="Number of QA pairs to generate.")
    p.add_argument("--out", default="evaluation/data/testset.json",
                   help="Output JSON path.")
    p.add_argument("--model-name", default="amazon.nova-lite-v1:0",
                   help="BaseAgent generation model.")
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="BaseAgent embedding model.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args(argv)
    try:
        samples = generate_testset(
            pdf_path=args.pdf,
            output_path=args.out,
            testset_size=args.size,
            model_name=args.model_name,
            embedding_model=args.embedding_model,
        )
        print(f"\n[OK] Generated {len(samples)} QA pair(s) -> {args.out}")
        return 0
    except Exception as exc:
        logger.exception("Dataset generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
