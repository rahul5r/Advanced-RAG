import re
from src.ai.agents.base_agent import BaseAgent


class Generator:
    """
    Generator for RAG that produces grounded, context-based answers.

    Design principles:
    - Answers must be grounded in the provided context
    - Response length adapts to question complexity (factoid vs explanatory)
    - Clear distinction between "not in context" vs "no context provided"
    """

    FALLBACK = "I don't know based on the provided context."

    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def generate_answer(self, query: str, context_chunks: list[str], temperature: float = 0.1) -> str:
        """
        Generate a grounded answer based on retrieved context.

        Args:
            query: The user's question
            context_chunks: List of retrieved text chunks
            temperature: Generation temperature (default 0.1 for more deterministic output)

        Returns:
            Grounded answer or fallback message
        """
        if not context_chunks:
            return self.FALLBACK

        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a precise RAG answering assistant. Your task is to answer questions "
            "using ONLY the provided context.\n\n"
            "GUIDELINES:\n"
            "1. Answer using ONLY the information in the provided context.\n"
            "2. Do NOT use outside knowledge or make assumptions.\n"
            "3. For simple factoid questions: Answer concisely in 1-2 sentences.\n"
            "4. For complex 'how' or 'why' questions: Provide a complete explanation using all relevant details from the context.\n"
            "5. If the context contains the answer: Extract and synthesize it clearly.\n"
            "6. If the context does NOT contain the answer, say EXACTLY:\n 'I don't know based on the provided context.'\n"
            "7. Do NOT add phrases like 'Based on the context...' or 'The context says...'\n"
            "8. Do NOT add commentary, disclaimers, or explanations about your process.\n\n"
            "9. Add citations in the format [Chunk <number>] when referencing specific information.\n"
            "10. Add More number of citations if the chunks contain relevant information like this: [Chunk <number1>, Chunk <number2>].\n\n"
            "Remember: Being concise is good, but being complete is better when the question requires it."
        )

        user_prompt = (
            f"=== CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
            f"=== QUESTION ===\n{query}\n=== END QUESTION ===\n\n"
            "Provide your answer:"
        )

        answer = self.agent.generate_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=2048,
        ).strip()

        if not answer:
            return self.FALLBACK

        # Clean up common LLM response patterns
        answer = self._clean_answer(answer)

        # Validate the answer is substantive
        if self._is_valid_answer(answer):
            return answer
        else:
            return self.FALLBACK

    def _clean_answer(self, answer: str) -> str:
        """Remove unwanted prefixes and formatting from the answer."""
        # Strip code fences
        answer = re.sub(r"^```(?:\w+)?\n?", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"\n```$", "", answer)

        # Strip common lead-in phrases
        patterns = [
            r"^(answer\s*:\s*)",
            r"^(based on the (context|evidence|provided information)[,:\s]*)",
            r"^(according to the (context|evidence|provided information)[,:\s]*)",
            r"^(the context (says|states|mentions|indicates)[,:\s]*)",
            r"^(from the context[,:\s]*)",
            r"^(here is the answer[,:\s]*)",
        ]
        for pattern in patterns:
            answer = re.sub(pattern, "", answer, flags=re.IGNORECASE).strip()

        # Remove trailing whitespace and normalize
        answer = answer.strip()
        answer = re.sub(r"\s+", " ", answer)

        return answer

    def _is_valid_answer(self, answer: str) -> bool:
        """
        Check if the answer is substantive and not a deflection.

        Returns False for:
        - Very short non-answers (< 10 chars unless it's a clear factoid)
        - Deflections like "I cannot answer"
        - Empty or whitespace-only responses
        """
        if not answer or not answer.strip():
            return False

        # Check for deflection patterns
        deflections = [
            "i cannot",
            "i can't",
            "unable to",
            "not enough information",
            "not mentioned",
            "not provided",
            "doesn't say",
            "does not say",
        ]
        answer_lower = answer.lower()
        if any(d in answer_lower for d in deflections):
            # Exception: if it contains our exact fallback, that's handled separately
            if answer == self.FALLBACK:
                return True
            return False

        # Very short answers might be valid for factoid questions
        if len(answer) < 10:
            # Allow short answers that look like entities/facts
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", answer):
                return True
            if re.match(r"^\d+(?:\s*\%|\s*times|\s*years?)?$", answer):
                return True
            return False

        return True