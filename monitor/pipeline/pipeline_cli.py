from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.pipeline import PipelineService
from monitor.prompts import load_custom_prompts_from_file, load_custom_prompts_from_list, load_prompts
from monitor.providers.answer_generator import OpenAIAnswerGenerator
from monitor.providers.input_classifier import GemmaInputClassifier, OpenAIInputClassifier
from monitor.providers.output_classifier import GemmaOutputClassifier, OpenAIOutputClassifier
from monitor.storage import DatabaseLogHandler, ModerationRepository
from schema import PipelineResult
from toxic_gemma_classifier import ClassifierConfig, ToxicLoRAClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompts through the complete AI safety pipeline: input classification -> answer generation -> output classification."
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of prompts to evaluate.")
    parser.add_argument(
        "--input-classifier",
        type=str,
        choices=("gemma", "openai"),
        default="gemma",
        help="Input classifier backend. Defaults to Gemma.",
    )
    parser.add_argument(
        "--output-classifier",
        type=str,
        choices=("gemma", "openai", "none"),
        default="gemma",
        help="Output classifier backend. Use 'none' to skip output classification.",
    )
    parser.add_argument(
        "--answer-generator",
        type=str,
        choices=("openai", "none"),
        default="openai",
        help="Answer generator. Use 'none' to skip answer generation.",
    )
    parser.add_argument(
        "--openai-moderation-model",
        type=str,
        default="omni-moderation-latest",
        help="OpenAI moderation model name. Used for input/output classification when using OpenAI.",
    )
    parser.add_argument(
        "--openai-answer-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for answer generation.",
    )
    parser.add_argument(
        "--gemma-base-model",
        type=str,
        default="google/gemma-2b-it",
        help="Base Gemma checkpoint to use for the local classifier.",
    )
    parser.add_argument(
        "--gemma-adapter-path",
        type=str,
        default=None,
        help="Optional path to a PEFT adapter checkpoint fine-tuned on safety data.",
    )
    parser.add_argument(
        "--gemma-max-length",
        type=int,
        default=256,
        help="Maximum prompt length (tokens) when tokenizing for Gemma.",
    )
    parser.add_argument("--preview", type=int, default=5, help="How many results to preview in stdout.")
    parser.add_argument(
        "--dataset-id", type=str, default="allenai/real-toxicity-prompts", help="Dataset identifier to load."
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Path to a text file with custom prompts (one per line). If provided, this overrides --dataset-id.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        default=None,
        help="Custom prompt text (can be used multiple times). If provided, this overrides --dataset-id and --prompts-file.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("pipeline_results.json"), help="Path to store JSON results."
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Optional PostgreSQL database URL for persisting runs (e.g. postgresql+psycopg://user:pass@host/db)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def preview_results(results: Iterable[PipelineResult], limit: int) -> None:
    for index, result in enumerate(results):
        if index >= limit:
            break
        input_violated = [flag for flag in result.input_classification.flags if flag.violated]
        input_categories = ", ".join(f"{flag.category} ({flag.score:.2f})" for flag in input_violated) or "None"
        
        output_info = ""
        if result.output_classification:
            output_violated = [flag for flag in result.output_classification.flags if flag.violated]
            output_categories = ", ".join(f"{flag.category} ({flag.score:.2f})" for flag in output_violated) or "None"
            output_info = f" | Output: Flagged={result.output_classification.flagged} ({output_categories})"
        
        answer_preview = result.answer.text[:60] + "..." if result.answer and len(result.answer.text) > 60 else (result.answer.text if result.answer else "N/A")
        
        print(f"{index + 1}. Input: Flagged={result.input_classification.flagged} ({input_categories}){output_info}")
        print(f"   Prompt: {result.prompt.text[:80]}...")
        print(f"   Answer: {answer_preview}\n")


def save_results(results: List[PipelineResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump() for result in results]
    output_path.write_text(json.dumps(payload, indent=2))


def _serialize_args(args: argparse.Namespace) -> dict:
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def build_providers(args: argparse.Namespace):
    """Build input classifier, answer generator, and output classifier based on args."""
    # Build input classifier
    if args.input_classifier == "openai":
        input_classifier = OpenAIInputClassifier(model=args.openai_moderation_model)
        input_model_name = args.openai_moderation_model
    else:  # gemma
        adapter_path = args.gemma_adapter_path or None
        config = ClassifierConfig(
            base_model=args.gemma_base_model,
            adapter_path=adapter_path,
            max_length=args.gemma_max_length,
        )
        classifier = ToxicLoRAClassifier(config)
        input_classifier = GemmaInputClassifier(classifier=classifier)
        input_model_name = config.base_model

    # Build answer generator
    answer_generator: Optional[OpenAIAnswerGenerator] = None
    if args.answer_generator == "openai":
        answer_generator = OpenAIAnswerGenerator(model=args.openai_answer_model)
        answer_model_name = args.openai_answer_model
    else:
        answer_model_name = None

    # Build output classifier
    output_classifier: Optional[GemmaOutputClassifier | OpenAIOutputClassifier] = None
    if args.output_classifier == "openai":
        output_classifier = OpenAIOutputClassifier(model=args.openai_moderation_model)
        output_model_name = args.openai_moderation_model
    elif args.output_classifier == "gemma":
        adapter_path = args.gemma_adapter_path or None
        config = ClassifierConfig(
            base_model=args.gemma_base_model,
            adapter_path=adapter_path,
            max_length=args.gemma_max_length,
        )
        classifier = ToxicLoRAClassifier(config)
        output_classifier = GemmaOutputClassifier(classifier=classifier)
        output_model_name = config.base_model
    else:
        output_model_name = None

    # Create a model name string for the run
    model_name = f"input:{input_model_name}"
    if answer_model_name:
        model_name += f",answer:{answer_model_name}"
    if output_model_name:
        model_name += f",output:{output_model_name}"

    return input_classifier, answer_generator, output_classifier, model_name


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    # Load prompts: custom prompts take precedence over dataset
    if args.prompt:
        # Custom prompts from CLI arguments
        logging.info("Loading %d custom prompt(s) from command line.", len(args.prompt))
        prompts = load_custom_prompts_from_list(args.prompt)
    elif args.prompts_file:
        # Custom prompts from file
        logging.info("Loading custom prompts from file: %s", args.prompts_file)
        prompts = load_custom_prompts_from_file(args.prompts_file)
        if args.limit and args.limit < len(prompts):
            prompts = prompts[: args.limit]
            logging.info("Limited to first %d prompts.", args.limit)
    else:
        # Load from dataset
        logging.info("Loading %s prompts from dataset.", args.limit)
        prompts = load_prompts(n=args.limit, dataset_id=args.dataset_id, split=args.split)

    input_classifier, answer_generator, output_classifier, model_name = build_providers(args)
    service = PipelineService(
        input_classifier=input_classifier,
        answer_generator=answer_generator,
        output_classifier=output_classifier,
    )

    repository: ModerationRepository | None = None
    run_record = None
    if args.database_url:
        repository = ModerationRepository(args.database_url)
        repository.create_schema()
        # Determine dataset info for run record
        if args.prompt:
            dataset_id = "custom_cli"
            dataset_split = "custom"
            prompt_limit = len(prompts)
        elif args.prompts_file:
            dataset_id = "custom_file"
            dataset_split = str(args.prompts_file)
            prompt_limit = len(prompts)
        else:
            dataset_id = args.dataset_id
            dataset_split = args.split
            prompt_limit = args.limit
        
        run_record = repository.start_run(
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            model=model_name,
            prompt_limit=prompt_limit,
            output_path=str(args.output),
            extra_args=_serialize_args(args),
        )
        db_handler = DatabaseLogHandler(repository, run_record.id)
        logging.getLogger().addHandler(db_handler)

    logging.info("Starting pipeline: input classification -> answer generation -> output classification")
    results = service.process_prompts(prompts)
    logging.info("Pipeline complete. Writing results to %s", args.output)

    save_results(results, args.output)
    if repository and run_record:
        repository.save_results(run_record.id, results)
        repository.complete_run(run_record.id)
    preview_results(results, args.preview)


if __name__ == "__main__":
    main()

