from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

from schema import ClassificationResult, ModerationFlag, ModerationResult, PipelineResult, Prompt


class ModerationService:
    """
    Legacy moderation service for backward compatibility.
    Only performs input classification (no answer generation or output classification).
    For full pipeline, use PipelineService instead.
    """

    def __init__(self, provider: Any, logger: Optional[logging.Logger] = None) -> None:
        self._provider = provider
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def moderate_prompts(self, prompts: Iterable[Prompt]) -> List[ModerationResult]:
        """Moderate prompts and return results (legacy - only input classification)."""
        results: List[ModerationResult] = []
        for prompt in prompts:
            self._logger.debug("Moderating prompt: %s", prompt.text[:80])
            
            # Support both old and new provider interfaces
            if hasattr(self._provider, "classify_input"):
                raw_response = self._provider.classify_input(prompt.text)
            elif hasattr(self._provider, "moderate_text"):
                raw_response = self._provider.moderate_text(prompt.text)
            else:
                raise ValueError("Provider must have either 'classify_input' or 'moderate_text' method")
            
            # Convert to PipelineResult for consistency, then to ModerationResult for backward compatibility
            pipeline_result = self._build_pipeline_result(prompt, raw_response)
            # Convert PipelineResult to ModerationResult (backward compatibility)
            moderation_result = ModerationResult(
                prompt=pipeline_result.prompt,
                flagged=pipeline_result.input_classification.flagged,
                flags=pipeline_result.input_classification.flags,
                raw_response=pipeline_result.input_classification.raw_response,
            )
            results.append(moderation_result)
        return results

    @staticmethod
    def _build_pipeline_result(prompt: Prompt, raw_response: dict) -> PipelineResult:
        """Build a PipelineResult from raw response (only input classification stage)."""
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

        input_classification = ClassificationResult(
            flagged=bool(result_payload.get("flagged", False)),
            flags=flags,
            raw_response=raw_response,
        )

        return PipelineResult(
            prompt=prompt,
            input_classification=input_classification,
            answer=None,
            output_classification=None,
        )
