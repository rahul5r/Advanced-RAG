"""
evaluation/reporter.py
-----------------------
Three output formats from the same list of StrategyResult objects:

  1. Console  — rich ASCII alignment table printed to stdout
  2. JSON     — machine-readable report  (reports/ragas_report_<ts>.json)
  3. Markdown — human-readable report    (reports/ragas_report_<ts>.md)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _fmt(v, prec: int = 4) -> str:
    return f"{v:.{prec}f}" if v is not None else "  N/A "


def _pass(passed: bool) -> str:
    return "✓ PASS" if passed else "✗ FAIL"


# ---------------------------------------------------------------------------
# 1. Console report
# ---------------------------------------------------------------------------

def print_console_report(results, faithfulness_threshold=0.80, context_recall_threshold=0.70):
    """
    Example output:
    ──────────────────────────────────────────────────────────────────────────────────
    Strategy        Faithfulness  CtxRecall  AnsRelevancy  CtxPrecision  Latency(ms)  Status
    ──────────────────────────────────────────────────────────────────────────────────
    hybrid             0.8561     0.7890       0.8300        0.8100        912.4    ✓ PASS
    expansion          0.8740     0.8010       0.8500        0.8200       1234.1    ✓ PASS
    ...
    """
    sep = "─" * 100

    header = (
        f"{'Strategy':<16}"
        f"{'Faithfulness':>14}"
        f"{'CtxRecall':>11}"
        f"{'AnsRelevancy':>14}"
        f"{'CtxPrecision':>14}"
        f"{'Latency(ms)':>13}"
        f"  Status"
    )

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in results:
        if r.error:
            print(
                f"{r.strategy_key:<16}"
                f"{'ERROR':>14}"
                f"{'':>11}"
                f"{'':>14}"
                f"{'':>14}"
                f"{'':>13}"
                f"  {r.error[:40]}"
            )
        else:
            print(
                f"{r.strategy_key:<16}"
                f"{_fmt(r.faithfulness):>14}"
                f"{_fmt(r.context_recall):>11}"
                f"{_fmt(r.answer_relevancy):>14}"
                f"{_fmt(r.context_precision):>14}"
                f"{r.avg_latency_ms:>13.1f}"
                f"  {_pass(r.passed)}"
            )

    print(sep)
    print(
        f"Thresholds:  Faithfulness ≥ {faithfulness_threshold:.2f}"
        f"  |  Context Recall ≥ {context_recall_threshold:.2f}"
    )
    print(sep)

    passed  = sum(1 for r in results if r.passed and not r.error)
    errored = sum(1 for r in results if r.error)
    total   = len(results)
    print(
        f"\nSummary: {passed}/{total} strategies passed"
        + (f"  |  {errored} errored" if errored else "")
        + "\n"
    )


# ---------------------------------------------------------------------------
# 2. JSON report
# ---------------------------------------------------------------------------

def save_json_report(results, config, report_dir: str = "evaluation/reports") -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(report_dir, f"ragas_report_{_ts()}.json")

    report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "strategies":               config.strategies,
            "top_k":                    config.top_k,
            "model_name":               config.model_name,
            "faithfulness_threshold":   config.faithfulness_threshold,
            "context_recall_threshold": config.context_recall_threshold,
        },
        "strategies": [],
    }

    for r in results:
        entry = {
            "strategy_key":   r.strategy_key,
            "strategy_label": r.strategy_label,
            "passed":         r.passed,
            "error":          r.error,
            "metrics": {
                "faithfulness":      r.faithfulness,
                "context_recall":    r.context_recall,
                "answer_relevancy":  r.answer_relevancy,
                "context_precision": r.context_precision,
            },
            "latency": {
                "avg_ms":   round(r.avg_latency_ms, 2),
                "total_ms": round(r.total_latency_ms, 2),
            },
            "per_sample": [
                {
                    "question":          s.question,
                    "answer":            s.answer,
                    "contexts":          s.contexts,
                    "ground_truth":      s.ground_truth,
                    "latency_ms":        round(s.latency_ms, 2),
                    "faithfulness":      s.faithfulness,
                    "context_recall":    s.context_recall,
                    "answer_relevancy":  s.answer_relevancy,
                    "context_precision": s.context_precision,
                }
                for s in r.samples
            ],
        }
        report["strategies"].append(entry)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("JSON report → %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# 3. Markdown report
# ---------------------------------------------------------------------------

def save_markdown_report(results, config, report_dir: str = "evaluation/reports") -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(report_dir, f"ragas_report_{_ts()}.md")

    lines: List[str] = []
    a = lines.append

    a("# RAGAS Evaluation Report")
    a("")
    a(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**Model:** `{config.model_name}`")
    a(f"**Top-K:** {config.top_k}  |  **Strategies:** {len(results)}")
    a("")

    # ── Thresholds ────────────────────────────────────────────────────────
    a("## Pass/Fail Thresholds")
    a("")
    a("| Metric | Threshold |")
    a("|--------|-----------|")
    a(f"| Faithfulness | ≥ {config.faithfulness_threshold:.2f} |")
    a(f"| Context Recall | ≥ {config.context_recall_threshold:.2f} |")
    a("")

    # ── Summary table ─────────────────────────────────────────────────────
    a("## Summary")
    a("")
    a("| Strategy | Faithfulness | Ctx Recall | Ans Relevancy | Ctx Precision | Avg Latency (ms) | Status |")
    a("|----------|:------------:|:----------:|:-------------:|:-------------:|:----------------:|:------:|")

    for r in results:
        if r.error:
            a(f"| `{r.strategy_key}` | — | — | — | — | — | ❌ ERROR |")
        else:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            a(
                f"| `{r.strategy_key}` "
                f"| {_fmt(r.faithfulness)} "
                f"| {_fmt(r.context_recall)} "
                f"| {_fmt(r.answer_relevancy)} "
                f"| {_fmt(r.context_precision)} "
                f"| {r.avg_latency_ms:.1f} "
                f"| {status} |"
            )
    a("")

    # ── Per-strategy sections ─────────────────────────────────────────────
    a("## Per-Strategy Details")
    a("")
    for r in results:
        a(f"### {r.strategy_label} (`{r.strategy_key}`)")
        a("")
        if r.error:
            a(f"> ⚠️ **Error:** {r.error}")
            a("")
            continue

        a(f"| Metric | Score |")
        a(f"|--------|-------|")
        a(f"| Faithfulness | {_fmt(r.faithfulness)} |")
        a(f"| Context Recall | {_fmt(r.context_recall)} |")
        a(f"| Answer Relevancy | {_fmt(r.answer_relevancy)} |")
        a(f"| Context Precision | {_fmt(r.context_precision)} |")
        a(f"| Avg Latency | {r.avg_latency_ms:.1f} ms |")
        a(f"| Result | {'✅ PASS' if r.passed else '❌ FAIL'} |")
        a("")

        if r.samples:
            a("#### Per-Sample Scores")
            a("")
            a("| # | Question | Faithfulness | Ctx Recall | Latency (ms) |")
            a("|---|----------|:------------:|:----------:|:------------:|")
            for idx, s in enumerate(r.samples, 1):
                q = s.question[:60] + ("…" if len(s.question) > 60 else "")
                a(
                    f"| {idx} | {q} "
                    f"| {_fmt(s.faithfulness, 3)} "
                    f"| {_fmt(s.context_recall, 3)} "
                    f"| {s.latency_ms:.0f} |"
                )
            a("")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Markdown report → %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Convenience: save both formats at once
# ---------------------------------------------------------------------------

def save_all_reports(results, config) -> dict:
    return {
        "json":     save_json_report(results, config, config.report_dir),
        "markdown": save_markdown_report(results, config, config.report_dir),
    }
