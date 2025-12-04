from __future__ import annotations

from typing import Any, Dict, Protocol

from toxic_gemma_classifier import ToxicLoRAClassifier, ClassifierConfig

class ClassifierProvider(Protocol):
    """Interface for services that evaluate text via input classifier APIs."""

    def moderate_text(self, text: str) -> Dict[str, Any]:  # pragma: no cover - interface definition
        """Run the classification on the provided text and return the raw provider payload."""


class OnPermClassifier:
    """Runs the Gemma LoRA safety classifier locally instead of calling an API."""

    def __init__(self, * , classifier: ToxicLoRAClassifier | None = None, config: ClassifierConfig | None = None) -> None:
        if classifier is not None:
            self._classifier = classifier
        else:
            if config is None:
                config = ClassifierConfig()
            self._classifier = ToxicLoRAClassifier(config)


    def moderate_text(self, text: str) -> Dict[str, Any]:
        classification = self._classifier.classify_text(text)[0]
        label = classification.get("label", "UNKNOWN")
        flagged = label == "TOXIC"
        
        return {
            "flagged": flagged,
            "label": label,
        
        }