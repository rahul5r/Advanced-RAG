"""
evaluation/dataset.py
---------------------
Loads and validates the test-set JSON produced by dataset_generator.py.

Expected schema
---------------
[
  {
    "question":     "What method did the paper propose?",
    "ground_truth": "The paper proposed ..."
  },
  ...
]
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TestSample:
    question: str
    ground_truth: str
    source_file: Optional[str] = None
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    chunk_id: Optional[int] = None

    def __post_init__(self):
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.ground_truth.strip():
            raise ValueError("ground_truth must not be empty")


def load_test_samples(filepath: str) -> List[TestSample]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Test-set not found: {filepath}\n\n"
            "Generate it first by running:\n"
            "  python -m evaluation.dataset_generator\n\n"
            "Or specify a custom path:\n"
            "  python -m evaluation.run_evaluation --test-file <path>"
        )

    with path.open("r", encoding="utf-8") as fh:
        raw: list = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError("testset.json must be a JSON array.")

    samples: List[TestSample] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dict: {item}")

        missing = {"question", "ground_truth"} - item.keys()
        if missing:
            raise ValueError(f"Item {i} is missing fields: {missing}")

        chunk_id = item.get("chunk_id")
        if chunk_id is not None:
            try:
                chunk_id = int(chunk_id)
            except Exception:
                chunk_id = None

        samples.append(
            TestSample(
                question=str(item["question"]).strip(),
                ground_truth=str(item["ground_truth"]).strip(),
                source_file=item.get("source_file"),
                source_path=item.get("source_path"),
                source_sha256=item.get("source_sha256"),
                chunk_id=chunk_id,
            )
        )

    if not samples:
        raise ValueError("testset.json contains no samples.")

    logger.info("Loaded %d test samples from %s", len(samples), filepath)
    return samples
