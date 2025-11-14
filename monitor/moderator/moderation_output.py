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
from monitor.providers.openai_client import OpenAIModerationProvider
from schema import ModerationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompts through OpenAI moderation and persist the results.")
    parser.add_argument("--limit", type=int, default=50, help="Number of prompts to evaluate.")
    parser.add_argument("--model", type=str, default="omni-moderation-latest", help="OpenAI moderation model to use.")
    parser.add_argument("--preview", type=int, default=5, help="How many results to preview in stdout.")
    parser.add_argument("--dataset-id", type=str, default="allenai/real-toxicity-prompts", help="Dataset identifier to load.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument("--output", type=Path, default=Path("moderation_results.json"), help="Path to store JSON results.")
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


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    logging.info("Loading %s prompts from dataset.", args.limit)
    prompts = load_prompts(n=args.limit, dataset_id=args.dataset_id, split=args.split)

    provider = OpenAIModerationProvider(model=args.model)
    service = ModerationService(provider)

    logging.info("Starting moderation run.")
    results = service.moderate_prompts(prompts)
    logging.info("Moderation complete. Writing results to %s", args.output)

    save_results(results, args.output)
    preview_results(results, args.preview)


if __name__ == "__main__":
    main()
