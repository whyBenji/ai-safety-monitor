from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from monitor.providers.openai_client import ModerationProvider
from schema import ModerationFlag, ModerationResult, Prompt


class ModerationService:
    """Coordinates prompt evaluation through a moderation provider."""

    def __init__(self, provider: ModerationProvider, logger: Optional[logging.Logger] = None) -> None:
        self._provider = provider
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def moderate_prompts(self, prompts: Iterable[Prompt]) -> List[ModerationResult]:
        results: List[ModerationResult] = []
        for prompt in prompts:
            self._logger.debug("Moderating prompt: %s", prompt.text[:80])
            raw_response = self._provider.moderate_text(prompt.text)
            results.append(self._build_result(prompt, raw_response))
        return results

    @staticmethod
    def _build_result(prompt: Prompt, raw_response: dict) -> ModerationResult:
        result_payload = (raw_response.get("results") or [{}])[0]
        categories = result_payload.get("categories", {}) or {}
        category_scores = result_payload.get("category_scores", {}) or {}

        flags = [
            ModerationFlag(
                category=category,
                score=float(category_scores.get(category, 0.0)),
                violated=bool(is_flagged),
            )
            for category, is_flagged in categories.items()
        ]

        return ModerationResult(
            prompt=prompt,
            flagged=bool(result_payload.get("flagged", False)),
            flags=flags,
            raw_response=raw_response,
        )
