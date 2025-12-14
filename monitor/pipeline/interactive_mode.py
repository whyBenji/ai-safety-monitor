#!/usr/bin/env python3
"""
Interactive mode for the AI Safety Monitor pipeline.
Allows real-time prompt entry and processing, similar to production systems.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.pipeline import PipelineService
from monitor.prompts import load_custom_prompts_from_list
from monitor.providers.answer_generator import OpenAIAnswerGenerator
from monitor.providers.input_classifier import GemmaInputClassifier, OpenAIInputClassifier
from monitor.providers.output_classifier import GemmaOutputClassifier, OpenAIOutputClassifier
from monitor.storage import DatabaseLogHandler, ModerationRepository
from schema import PipelineResult, Prompt
from toxic_gemma_classifier import ClassifierConfig, ToxicLoRAClassifier


def configure_logging(verbose: bool) -> None:
    """Configure logging for interactive mode."""
    level = logging.DEBUG if verbose else logging.WARNING  # Less verbose for interactive
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],  # Log to stderr so stdout is clean
    )


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


def display_result(result: PipelineResult, index: int) -> None:
    """Display a formatted result in the terminal."""
    print("\n" + "=" * 80)
    print(f"RESULT #{index}")
    print("=" * 80)
    
    print(f"\n📝 PROMPT:")
    print(f"   {result.prompt.text}")
    
    print(f"\n🔍 INPUT CLASSIFICATION:")
    status = "🔴 TOXIC" if result.input_classification.flagged else "🟢 SAFE"
    print(f"   Status: {status}")
    if result.input_classification.flags:
        for flag in result.input_classification.flags:
            if flag.violated:
                print(f"   ⚠️  {flag.category}: {flag.score:.2f}")
    
    if result.answer:
        print(f"\n💬 GENERATED ANSWER:")
        answer_preview = result.answer.text[:200] + "..." if len(result.answer.text) > 200 else result.answer.text
        print(f"   {answer_preview}")
        print(f"   Model: {result.answer.model}")
        
        if result.output_classification:
            print(f"\n🔍 OUTPUT CLASSIFICATION:")
            status = "🔴 TOXIC" if result.output_classification.flagged else "🟢 SAFE"
            print(f"   Status: {status}")
            if result.output_classification.flags:
                for flag in result.output_classification.flags:
                    if flag.violated:
                        print(f"   ⚠️  {flag.category}: {flag.score:.2f}")
    else:
        print(f"\n⏭️  ANSWER GENERATION: Skipped (input was flagged as toxic)")
    
    print("\n" + "=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive mode: Enter prompts manually and process them through the AI safety pipeline."
    )
    parser.add_argument(
        "--input-classifier",
        type=str,
        choices=("gemma", "openai"),
        default="openai",
        help="Input classifier backend. Defaults to OpenAI for faster response.",
    )
    parser.add_argument(
        "--output-classifier",
        type=str,
        choices=("gemma", "openai", "none"),
        default="openai",
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
        help="OpenAI moderation model name.",
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
        help="Optional path to a PEFT adapter checkpoint.",
    )
    parser.add_argument(
        "--gemma-max-length",
        type=int,
        default=256,
        help="Maximum prompt length (tokens) when tokenizing for Gemma.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="sqlite:///./ai_monitor.db",
        help="Database URL for storing results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to save results (in addition to database).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    print("\n" + "=" * 80)
    print("🤖 AI Safety Monitor - Interactive Mode")
    print("=" * 80)
    print("\nEnter prompts one at a time. Each prompt will be processed through:")
    print("  1. Input Classification")
    print("  2. Answer Generation (if input is safe)")
    print("  3. Output Classification (if answer was generated)")
    print("\nCommands:")
    print("  - Type your prompt and press Enter to process it")
    print("  - Type 'exit' or 'quit' to finish")
    print("  - Type 'clear' to clear the screen")
    print("  - Press Ctrl+C to exit")
    print("\n" + "=" * 80 + "\n")

    # Build providers
    print("Initializing pipeline components...")
    input_classifier, answer_generator, output_classifier, model_name = build_providers(args)
    service = PipelineService(
        input_classifier=input_classifier,
        answer_generator=answer_generator,
        output_classifier=output_classifier,
    )
    print("✓ Pipeline ready!\n")

    # Setup database
    repository = ModerationRepository(args.database_url)
    repository.create_schema()
    
    # Start a run for this interactive session
    run_record = repository.start_run(
        dataset_id="interactive",
        dataset_split="manual_entry",
        model=model_name,
        prompt_limit=0,  # Will be updated as prompts are added
        output_path=str(args.output) if args.output else None,
        extra_args={"mode": "interactive"},
    )
    db_handler = DatabaseLogHandler(repository, run_record.id)
    logging.getLogger().addHandler(db_handler)
    
    print(f"📊 Session started. Run ID: {run_record.id}")
    print(f"💾 Results will be saved to: {args.database_url}")
    if args.output:
        print(f"📄 Results will also be saved to: {args.output}")
    print("\n" + "-" * 80 + "\n")

    results: list[PipelineResult] = []
    prompt_count = 0

    try:
        while True:
            try:
                # Get user input
                user_input = input("Enter prompt (or 'exit'/'quit' to finish): ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                
                if user_input.lower() == "clear":
                    import os
                    os.system("clear" if os.name != "nt" else "cls")
                    continue
                
                # Process the prompt
                prompt_count += 1
                print(f"\n🔄 Processing prompt #{prompt_count}...")
                
                prompt = Prompt(text=user_input, metadata={})
                prompt_results = service.process_prompts([prompt])
                
                if prompt_results:
                    result = prompt_results[0]
                    results.append(result)
                    
                    # Display result
                    display_result(result, prompt_count)
                    
                    # Save to database immediately
                    repository.save_results(run_record.id, [result])
                    
                    # Update run limit
                    with repository.session() as session:
                        from monitor.storage.models import ModerationRun
                        run = session.get(ModerationRun, run_record.id)
                        if run:
                            run.prompt_limit = prompt_count
                    
                    print(f"✅ Saved to database (Run ID: {run_record.id})")
                    
                    # Save to JSON file if specified
                    if args.output:
                        output_data = [r.model_dump() for r in results]
                        args.output.write_text(json.dumps(output_data, indent=2))
                        print(f"✅ Saved to {args.output}")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error processing prompt: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
    
    except EOFError:
        print("\n\n⚠️  End of input")
    
    finally:
        # Complete the run
        repository.complete_run(run_record.id)
        
        print("\n" + "=" * 80)
        print(f"📊 Session Summary")
        print("=" * 80)
        print(f"Total prompts processed: {prompt_count}")
        print(f"Run ID: {run_record.id}")
        print(f"Database: {args.database_url}")
        if args.output:
            print(f"JSON file: {args.output}")
        print("\n💡 View results in the dashboard:")
        print(f"   export AI_SAFETY_MONITOR_DB={args.database_url}")
        print("   uvicorn monitor.dashboard.app:app --reload --port 8000")
        print("   Then open: http://127.0.0.1:8000")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

