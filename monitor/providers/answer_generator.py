from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from openai import OpenAI


class AnswerGenerator(Protocol):
    """Interface for services that generate answers to prompts."""

    def generate_answer(self, prompt: str) -> Dict[str, Any]:  # pragma: no cover - interface definition
        """Generate an answer for the provided prompt and return the raw provider payload."""
        ...


class OpenAIAnswerGenerator:
    """OpenAI-backed answer generator using chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini", client: OpenAI | None = None) -> None:
        self._model = model
        self._client = client or OpenAI()

    def generate_answer(self, prompt: str) -> Dict[str, Any]:
        """Generate an answer using OpenAI chat completions."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        answer_text = response.choices[0].message.content or ""
        
        if hasattr(response, "model_dump"):
            raw_response = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw_response = response.to_dict()
        else:
            raw_response = {"raw": str(response)}
        
        return {
            "model": self._model,
            "text": answer_text,
            "raw_response": raw_response,
        }

