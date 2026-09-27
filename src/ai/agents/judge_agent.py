"""
Dedicated Nova Lite judge agent for RAGAS.

Keeps your main app generation agent separate from evaluator behavior.
"""

from __future__ import annotations

import os
from src.ai.agents.base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model_name=os.getenv("RAGAS_MODEL_NAME", os.getenv(
                "MODEL_NAME", "amazon.nova-lite-v1:0")),
            temperature=0.0,
            request_timeout=int(os.getenv("RAGAS_REQUEST_TIMEOUT", "180")),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
