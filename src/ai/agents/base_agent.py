"""
Base Agent — Amazon Bedrock (Nova) wrapper + Local Embeddings.
All RAG sub-modules import from here; no direct HTTP calls elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseAgent:
    """
    Thin wrapper around Amazon Bedrock Nova that provides:
      - Stateless single-turn generation  (`generate_response`)
      - Strict JSON generation            (`generate_json_response`)
      - Stateful multi-turn chat          (`chat` / `reset_chat`)
      - Local embeddings via SentenceTransformer
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        request_timeout: Optional[int] = None,
        region_name: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("MODEL_NAME", "amazon.nova-lite-v1:0")
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.default_temperature = temperature
        self.request_timeout = int(request_timeout or os.getenv("BEDROCK_REQUEST_TIMEOUT", "180"))
        self.chat_history: list[dict[str, Any]] = []

        # boto3 Bedrock Runtime client
        boto_config = Config(
            read_timeout=self.request_timeout,
            connect_timeout=30,
            retries={"max_attempts": 5, "mode": "standard"},
        )
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
            config=boto_config,
        )

        # Local embeddings
        self.embedding_model_name = embedding_model
        self.embedding_model_id = os.getenv("EMBEDDING_MODEL_ID")

        # FIX: Removed duplicate log line — "Configured embedding model" and
        # "Embedding model" were logging the exact same value twice.
        logger.info("BaseAgent initialised — model: %s", self.model_name)
        logger.info("Bedrock region: %s", self.region_name)
        logger.info("Embedding model: %s", self.embedding_model_name)

    # Internal helpers
    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = {
            "messages": messages,
            "inferenceConfig": {
                "temperature": self.default_temperature if temperature is None else temperature,
                "maxTokens": max_tokens or 2048,
            },
        }
        if system_prompt:
            payload["system"] = [{"text": system_prompt}]
        return payload

    def _invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())

    @staticmethod
    def _extract_text(result: dict[str, Any]) -> str:
        try:
            output = result.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            texts: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))

            return "".join(texts).strip()
        except Exception:
            logger.exception("Unexpected Bedrock response format: %s", result)
            return ""

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @staticmethod
    def _extract_first_json_candidate(text: str) -> Optional[str]:
        text = BaseAgent._strip_code_fences(text)
        if not text:
            return None

        try:
            json.loads(text)
            return text
        except Exception:
            pass

        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            if start == -1:
                continue

            depth = 0
            in_string = False
            escape = False

            for i in range(start, len(text)):
                ch = text[i]

                if escape:
                    escape = False
                    continue

                if ch == "\\":
                    escape = True
                    continue

                if ch == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            break

        return None

    # Public API
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
            system_prompt=system_prompt,
        )
        result = self._invoke(payload)
        return self._extract_text(result)

    def generate_json_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        fallback: str = "{}",
    ) -> str:
        json_system = (
            "You are a strict JSON generator.\n"
            "Return ONLY valid JSON.\n"
            "Do not use markdown fences.\n"
            "Do not explain.\n"
            "Do not answer conversationally.\n"
            "The output must be parseable by json.loads()."
        )
        if system_prompt:
            json_system = f"{json_system}\n\n{system_prompt}"

        raw = self.generate_response(
            prompt=prompt,
            system_prompt=json_system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        candidate = self._extract_first_json_candidate(raw)
        if candidate is not None:
            return candidate

        logger.warning("generate_json_response: non-JSON output from model: %r", raw[:500])
        return fallback

    def chat(self, message: str) -> str:
        self.chat_history.append(
            {"role": "user", "content": [{"text": message}]}
        )

        payload = self._build_payload(self.chat_history)
        result = self._invoke(payload)
        response_text = self._extract_text(result)

        self.chat_history.append(
            {"role": "assistant", "content": [{"text": response_text}]}
        )
        return response_text

    def reset_chat(self) -> None:
        self.chat_history = []
        logger.debug("Chat session reset.")

    def get_embedding(self, text: str) -> list[float]:
        body = {"inputText": text}

        response = self.client.invoke_model(
            modelId=self.embedding_model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        return result["embedding"]

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.get_embedding(text) for text in texts]


if __name__ == "__main__":
    agent = BaseAgent()
    print("Model:", agent.model_name)
    print(
        agent.generate_response(
            "Explain recursion in one sentence.",
            system_prompt="You are a concise technical assistant.",
            temperature=0.2,
            max_output_tokens=100,
        )
    )