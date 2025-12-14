from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptMetadata(BaseModel):
    """Structured metadata associated with a prompt."""

    dataset_id: Optional[str] = None
    dataset_split: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class Prompt(BaseModel):
    """Normalized prompt representation used throughout the pipeline."""

    text: str
    metadata: PromptMetadata = Field(default_factory=PromptMetadata)


class ModerationFlag(BaseModel):
    """Represents a single moderation category evaluation."""

    category: str
    score: float
    violated: bool


class ClassificationResult(BaseModel):
    """Result from a classification step (input or output)."""

    flagged: bool
    flags: List[ModerationFlag] = Field(default_factory=list)
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class AnswerGeneration(BaseModel):
    """Generated answer from the LLM."""

    text: str
    model: str
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Complete pipeline result: input classification -> answer generation -> output classification."""

    prompt: Prompt
    input_classification: ClassificationResult
    answer: Optional[AnswerGeneration] = None
    output_classification: Optional[ClassificationResult] = None
    
    # Legacy compatibility - use input_classification for backward compatibility
    @property
    def flagged(self) -> bool:
        """Backward compatibility: returns input classification flagged status."""
        return self.input_classification.flagged
    
    @property
    def flags(self) -> List[ModerationFlag]:
        """Backward compatibility: returns input classification flags."""
        return self.input_classification.flags
    
    @property
    def raw_response(self) -> Dict[str, Any]:
        """Backward compatibility: returns input classification raw response."""
        return self.input_classification.raw_response


class ModerationResult(BaseModel):
    """
    Legacy moderation result for backward compatibility.
    Only includes input classification (no answer generation or output classification).
    For full pipeline results, use PipelineResult.
    """

    prompt: Prompt
    flagged: bool
    flags: List[ModerationFlag] = Field(default_factory=list)
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_pipeline_result(cls, pipeline_result: PipelineResult) -> "ModerationResult":
        """Create a ModerationResult from a PipelineResult (for backward compatibility)."""
        return cls(
            prompt=pipeline_result.prompt,
            flagged=pipeline_result.input_classification.flagged,
            flags=pipeline_result.input_classification.flags,
            raw_response=pipeline_result.input_classification.raw_response,
        )
