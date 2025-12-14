from __future__ import annotations

from typing import Any, Dict, Protocol

from openai import OpenAI


# Legacy interfaces - kept for backward compatibility with old moderation_output.py
class ModerationProvider(Protocol):
    """Legacy interface for services that evaluate text via moderation APIs."""

    def moderate_text(self, text: str) -> Dict[str, Any]:  # pragma: no cover - interface definition
        """Run moderation on the provided text and return the raw provider payload."""


class OpenAIModerationProvider:
    """Legacy OpenAI-backed moderation provider. Use OpenAIInputClassifier or OpenAIOutputClassifier instead."""

    def __init__(self, model: str = "omni-moderation-latest", client: OpenAI | None = None) -> None:
        self._model = model
        self._client = client or OpenAI()

    def moderate_text(self, text: str) -> Dict[str, Any]:
        response = self._client.moderations.create(model=self._model, input=text)

        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()  # type: ignore[return-value]
        return response  # type: ignore[return-value]
