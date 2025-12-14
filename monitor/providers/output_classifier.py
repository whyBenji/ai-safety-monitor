from __future__ import annotations

from typing import Any, Dict, Protocol

from toxic_gemma_classifier import ClassifierConfig, ToxicLoRAClassifier


class OutputClassifier(Protocol):
    """Interface for services that classify generated outputs."""

    def classify_output(self, text: str) -> Dict[str, Any]:  # pragma: no cover - interface definition
        """Classify the output text and return the raw provider payload."""
        ...


class GemmaOutputClassifier:
    """Runs the Gemma LoRA safety classifier locally for output classification."""

    def __init__(
        self,
        *,
        classifier: ToxicLoRAClassifier | None = None,
        config: ClassifierConfig | None = None,
    ) -> None:
        if classifier is not None:
            self._classifier = classifier
        else:
            if config is None:
                config = ClassifierConfig()
            self._classifier = ToxicLoRAClassifier(config)
        self._model_name = self._classifier.config.base_model

    def classify_output(self, text: str) -> Dict[str, Any]:
        """Classify output text using Gemma classifier."""
        classification = self._classifier.classify_text(text)
        label = classification.get("label", "UNKNOWN").upper()
        flagged = label == "TOXIC"

        return {
            "model": self._model_name,
            "results": [
                {
                    "flagged": flagged,
                    "label": label,
                    "raw_text": classification.get("raw", ""),
                    "categories": {
                        "toxicity": flagged,
                        "safety": not flagged,
                    },
                    "category_scores": {
                        "toxicity": 1.0 if flagged else 0.0,
                        "safety": 1.0 if not flagged else 0.0,
                    },
                }
            ],
        }


class OpenAIOutputClassifier:
    """OpenAI-backed output classifier using the Moderations API."""

    def __init__(self, model: str = "omni-moderation-latest", client: Any = None) -> None:
        from openai import OpenAI
        self._model = model
        self._client = client or OpenAI()

    def classify_output(self, text: str) -> Dict[str, Any]:
        """Classify output text using OpenAI Moderation API."""
        response = self._client.moderations.create(model=self._model, input=text)

        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()
        return response

