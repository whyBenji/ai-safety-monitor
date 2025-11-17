from __future__ import annotations

from typing import Any, Dict, Protocol

from openai import OpenAI


class ModerationProvider(Protocol):
    """Interface for services that evaluate text via moderation APIs."""

    def moderate_text(self, text: str) -> Dict[str, Any]:  # pragma: no cover - interface definition
        """Run moderation on the provided text and return the raw provider payload."""


class OpenAIModerationProvider:
    """OpenAI-backed moderation provider that wraps the Moderations API."""

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


_chat_client = OpenAI()


def complete_gpt4o_mini(prompt: str) -> str:
    """Lightweight convenience wrapper around the GPT-4o-mini chat completions API."""

    resp = _chat_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
