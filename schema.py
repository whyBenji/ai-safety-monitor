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


class ModerationResult(BaseModel):
    """Aggregate moderation outcome for a prompt."""

    prompt: Prompt
    flagged: bool
    flags: List[ModerationFlag] = Field(default_factory=list)
    raw_response: Dict[str, Any] = Field(default_factory=dict)
