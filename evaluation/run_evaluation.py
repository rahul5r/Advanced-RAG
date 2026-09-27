#!/usr/bin/env python3
"""
evaluation/run_evaluation.py
-----------------------------
End-to-end CLI for the RAGAS evaluation pipeline.

Two-step workflow
-----------------
Step 1  — Generate a grounded test-set from your PDF  (run once, reuse many times)
Step 2  — Evaluate all 5 retrievers against the test-set

Both steps can be run together or separately.

Usage examples
--------------
# Full pipeline: generate test-set THEN evaluate all 5 retrievers
python -m evaluation.run_evaluation --generate

# Only generate the test-set (skip evaluation)
python -m evaluation.run_evaluation --generate --gen-only

# Only evaluate (test-set already exists)
python -m evaluation.run_evaluation

# Evaluate specific retrievers only
python -m evaluation.run_evaluation --strategies hybrid rerank multi_hop

# Use a larger test-set and custom PDF
python -m evaluation.run_evaluation \\
    --generate \\
    --pdf data/pdfs/my_paper.pdf \\
    --testset-size 30 \\
    --top-k 5 \\
    --verbose

Available retriever strategies
--------------------------------
  hybrid      — Hybrid Search (Qdrant cosine + BM25 keyword fusion)
  expansion   — Semantic Query Expansion (paraphrases -> merged results)
  rerank      — Context Re-Ranking (broad retrieval -> LLM re-scorer)
  self_query  — Self-Query Retriever (LLM metadata filter extraction)
  multi_hop   — Multi-Hop Retriever (query decomposition -> per-sub-query retrieval)

Required environment variables (.env)
--------------------------------------
  AWS_API_KEY   — Bedrock Nova API key
  HF_TOKEN      — HuggingFace token for embedding API
  QDRANT_URL    — Qdrant cluster URL
  QDRANT_API_KEY — Qdrant API key
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ── Make project root importable ────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for noisy in ("httpx", "httpcore", "urllib3", "faiss"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m evaluation.run_evaluation",
        description="RAGAS evaluation pipeline for all 5 RAG retriever strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Generation flags ──────────────────────────────────────────────────
    gen = p.add_argument_group("Test-set generation (--generate enables these)")
    gen.add_argument(
        "--generate", action="store_true",
        help="Generate a grounded test-set from the PDF before evaluating."
    )
    gen.add_argument(
        "--gen-only", action="store_true",
        help="Only generate the test-set; skip evaluation."
    )
    gen.add_argument(
        "--pdf", default=None, metavar="PATH",
        help="PDF to generate from (default: data/pdfs/Company-Policy.pdf)."
    )
    gen.add_argument(
        "--testset-size", type=int, default=None, metavar="N",
        help="Number of QA pairs to generate (default: 20)."
    )

    # ── Evaluation flags ──────────────────────────────────────────────────
    ev = p.add_argument_group("Evaluation")
    ev.add_argument(
        "--strategies", nargs="+", metavar="STRATEGY", default=None,
        help=(
            "Strategies to evaluate. "
            "Choices: hybrid expansion rerank self_query multi_hop. "
            "Default: all five."
        ),
    )
    ev.add_argument(
        "--top-k", type=int, default=None, metavar="N",
        help="Chunks retrieved per query (default: from .env or 5).",
    )
    ev.add_argument(
        "--test-file", default=None, metavar="PATH",
        help="Path to testset JSON (default: evaluation/data/testset.json).",
    )
    ev.add_argument(
        "--report-dir", default=None, metavar="DIR",
        help="Directory for report output (default: evaluation/reports).",
    )
    ev.add_argument(
        "--no-save", action="store_true",
        help="Print results only; do not write report files.",
    )
    ev.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    logger = logging.getLogger("evaluation")

    # ── Config ────────────────────────────────────────────────────────────
    from evaluation.config import EvaluationConfig
    config = EvaluationConfig()

    if args.strategies:
        config.strategies = args.strategies
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.test_file:
        config.test_file = args.test_file
    if args.report_dir:
        config.report_dir = args.report_dir
    if args.pdf:
        config.pdf_path = args.pdf
    if args.testset_size:
        config.testset_size = args.testset_size
    config.verbose = args.verbose

    try:
        config.validate()
    except (ValueError, EnvironmentError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    # ── Step 1: Generate test-set (optional) ─────────────────────────────
    if args.generate:
        logger.info("===  Step 1: Generating test-set from PDF  ===")
        logger.info("PDF      : %s", config.pdf_path)
        logger.info("Size     : %d QA pairs", config.testset_size)
        logger.info("Output   : %s", config.test_file)

        from evaluation.dataset_generator import generate_testset
        try:
            samples = generate_testset(
                pdf_path=config.pdf_path,
                output_path=config.test_file,
                testset_size=config.testset_size,
                model_name=config.model_name,
                embedding_model=config.embedding_model,
            )
            print(f"\n[OK] Test-set generated: {len(samples)} QA pairs -> {config.test_file}\n")
        except Exception as exc:
            logger.error("Test-set generation failed: %s", exc)
            return 1

        if args.gen_only:
            return 0

    # ── Step 2: Load test samples ─────────────────────────────────────────
    logger.info("===  Step 2: Loading test samples  ===")
    from evaluation.dataset import load_test_samples
    try:
        test_samples = load_test_samples(config.test_file)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(
            "\nTip: Run with --generate to auto-create the test-set:\n"
            "   python -m evaluation.run_evaluation --generate\n"
        )
        return 1
    except ValueError as exc:
        logger.error("Dataset error: %s", exc)
        return 1

    logger.info("Loaded %d test samples.", len(test_samples))

    # ── Step 3: Run evaluation pipeline ──────────────────────────────────
    logger.info("===  Step 3: Evaluating %d strategies  ===", len(config.strategies))
    logger.info("Strategies : %s", config.strategies)
    logger.info("Top-K      : %d", config.top_k)

    from evaluation.pipeline import EvaluationPipeline
    pipeline = EvaluationPipeline(config)
    results  = pipeline.run(test_samples)

    # ── Step 4: Report ────────────────────────────────────────────────────
    from evaluation.reporter import print_console_report, save_all_reports

    print_console_report(
        results,
        faithfulness_threshold=config.faithfulness_threshold,
        context_recall_threshold=config.context_recall_threshold,
    )

    if not args.no_save:
        saved = save_all_reports(results, config)
        for fmt, path in saved.items():
            logger.info("Saved %s report -> %s", fmt.upper(), path)
        print(f"Reports written to: {config.report_dir}/")

    # Exit 1 if any strategy failed thresholds
    any_failed = any(not r.passed and not r.error for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
