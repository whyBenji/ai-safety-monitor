from __future__ import annotations

from typing import Any, Dict, Optional

from openai import OpenAI


class ModerationProvider:
    """Interface for moderation providers."""

    def moderate_text(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIModerationProvider(ModerationProvider):
    """Thin wrapper around OpenAI's moderation endpoint."""

    def __init__(self, model: str = "omni-moderation-latest", client: Optional[OpenAI] = None) -> None:
        self.model = model
        self._client = client or OpenAI()

    def moderate_text(self, text: str) -> Dict[str, Any]:
        response = self._client.moderations.create(model=self.model, input=text)
        return response.model_dump()
