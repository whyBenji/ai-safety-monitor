from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from schema import AnswerGeneration, ClassificationResult, ModerationFlag, PipelineResult, Prompt

from monitor.providers.answer_generator import AnswerGenerator
from monitor.providers.input_classifier import InputClassifier
from monitor.providers.output_classifier import OutputClassifier


class PipelineService:
    """
    Orchestrates the complete AI safety pipeline:
    1. Input Classification - classify the input prompt
    2. Answer Generation - generate answer if input is safe
    3. Output Classification - classify the generated answer
    4. Return complete pipeline result
    """

    def __init__(
        self,
        input_classifier: InputClassifier,
        answer_generator: Optional[AnswerGenerator] = None,
        output_classifier: Optional[OutputClassifier] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._input_classifier = input_classifier
        self._answer_generator = answer_generator
        self._output_classifier = output_classifier
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def process_prompts(self, prompts: Iterable[Prompt]) -> List[PipelineResult]:
        """
        Process prompts through the complete pipeline:
        input -> input_classifier -> answer_generator -> output_classifier -> output
        """
        results: List[PipelineResult] = []
        
        for prompt in prompts:
            self._logger.debug("Processing prompt: %s", prompt.text[:80])
            
            # Stage 1: Input Classification
            self._logger.debug("Stage 1: Classifying input...")
            input_raw_response = self._input_classifier.classify_input(prompt.text)
            input_classification = self._build_classification_result(input_raw_response)
            
            # Stage 2: Answer Generation (only if input is safe and generator is available)
            answer: Optional[AnswerGeneration] = None
            if self._answer_generator and not input_classification.flagged:
                self._logger.debug("Stage 2: Generating answer...")
                answer_raw = self._answer_generator.generate_answer(prompt.text)
                answer = AnswerGeneration(
                    text=answer_raw.get("text", ""),
                    model=answer_raw.get("model", "unknown"),
                    raw_response=answer_raw,
                )
            
            # Stage 3: Output Classification (only if answer was generated and classifier is available)
            output_classification: Optional[ClassificationResult] = None
            if self._output_classifier and answer:
                self._logger.debug("Stage 3: Classifying output...")
                output_raw_response = self._output_classifier.classify_output(answer.text)
                output_classification = self._build_classification_result(output_raw_response)
            
            result = PipelineResult(
                prompt=prompt,
                input_classification=input_classification,
                answer=answer,
                output_classification=output_classification,
            )
            results.append(result)
            
            self._logger.debug(
                "Pipeline complete - Input flagged: %s, Answer generated: %s, Output flagged: %s",
                input_classification.flagged,
                answer is not None,
                output_classification.flagged if output_classification else None,
            )
        
        return results

    @staticmethod
    def _build_classification_result(raw_response: dict) -> ClassificationResult:
        """Build a ClassificationResult from a raw provider response."""
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

        return ClassificationResult(
            flagged=bool(result_payload.get("flagged", False)),
            flags=flags,
            raw_response=raw_response,
        )

