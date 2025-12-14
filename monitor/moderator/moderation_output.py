from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.moderator import ModerationService
from monitor.prompts import load_prompts
from monitor.providers.input_classifier import GemmaInputClassifier
from monitor.providers.openai_client import ModerationProvider, OpenAIModerationProvider

# Backward compatibility wrapper
class OnPermClassifier:
    """Backward compatibility wrapper for GemmaInputClassifier."""
    def __init__(self, *, classifier=None, config=None):
        self._classifier = GemmaInputClassifier(classifier=classifier, config=config)
    
    def moderate_text(self, text: str):
        """Legacy interface - calls classify_input internally."""
        return self._classifier.classify_input(text)
from monitor.storage import DatabaseLogHandler, ModerationRepository
from schema import ModerationResult
from toxic_gemma_classifier import ClassifierConfig, ToxicLoRAClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompts through moderation providers and persist the results.")
    parser.add_argument("--limit", type=int, default=50, help="Number of prompts to evaluate.")
    parser.add_argument(
        "--provider",
        type=str,
        choices=("gemma", "openai"),
        default="gemma",
        help="Which moderation backend to use. Defaults to the local Gemma classifier.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="omni-moderation-latest",
        help="OpenAI moderation model name. Only used when --provider=openai.",
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
    parser.add_argument("--dataset-id", type=str, default="allenai/real-toxicity-prompts", help="Dataset identifier to load.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument("--output", type=Path, default=Path("moderation_results.json"), help="Path to store JSON results.")
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


def preview_results(results: Iterable[ModerationResult], limit: int) -> None:
    for index, result in enumerate(results):
        if index >= limit:
            break
        violated = [flag for flag in result.flags if flag.violated]
        categories = ", ".join(f"{flag.category} ({flag.score:.2f})" for flag in violated) or "None"
        print(f"{index + 1}. Flagged={result.flagged} | Categories={categories}")
        print(f"   Prompt: {result.prompt.text}\n")


def save_results(results: List[ModerationResult], output_path: Path) -> None:
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


def build_provider(args: argparse.Namespace):
    if args.provider == "openai":
        model_name = args.model
        provider = OpenAIModerationProvider(model=model_name)
        return provider, model_name

    adapter_path = args.gemma_adapter_path or None
    config = ClassifierConfig(
        base_model=args.gemma_base_model,
        adapter_path=adapter_path,
        max_length=args.gemma_max_length,
    )
    classifier = ToxicLoRAClassifier(config)
    provider = OnPermClassifier(classifier=classifier)
    return provider, config.base_model


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    logging.info("Loading %s prompts from dataset.", args.limit)
    prompts = load_prompts(n=args.limit, dataset_id=args.dataset_id, split=args.split)

    provider, provider_model_name = build_provider(args)
    service = ModerationService(provider)

    repository: ModerationRepository | None = None
    run_record = None
    if args.database_url:
        repository = ModerationRepository(args.database_url)
        repository.create_schema()
        run_record = repository.start_run(
            dataset_id=args.dataset_id,
            dataset_split=args.split,
            model=provider_model_name,
            prompt_limit=args.limit,
            output_path=str(args.output),
            extra_args=_serialize_args(args),
        )
        db_handler = DatabaseLogHandler(repository, run_record.id)
        logging.getLogger().addHandler(db_handler)

    logging.info("Starting moderation run with %s backend.", args.provider)
    results = service.moderate_prompts(prompts)
    logging.info("Moderation complete. Writing results to %s", args.output)

    save_results(results, args.output)
    if repository and run_record:
        repository.save_results(run_record.id, results)
        repository.complete_run(run_record.id)
    preview_results(results, args.preview)


if __name__ == "__main__":
    main()
