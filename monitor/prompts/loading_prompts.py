from pathlib import Path
from typing import List, Optional

from datasets import load_dataset

from schema import Prompt, PromptMetadata


def load_prompts(
    n: int = 1000,
    dataset_id: str = "allenai/real-toxicity-prompts",
    split: str = "train",
) -> List[Prompt]:
    """Load a subset of the Real Toxicity Prompts dataset as normalized Prompt objects."""

    dataset = load_dataset(dataset_id, split=split)
    limit = min(n, len(dataset))
    subset = dataset.select(range(limit))

    prompts: List[Prompt] = []
    for row in subset:
        prompt_data = row.get("prompt", {})
        metadata = PromptMetadata(
            dataset_id=dataset_id,
            dataset_split=split,
            attributes={key: value for key, value in prompt_data.items() if key != "text"},
        )
        prompts.append(Prompt(text=prompt_data.get("text", ""), metadata=metadata))

    return prompts


def load_custom_prompts_from_file(file_path: Path) -> List[Prompt]:
    """Load custom prompts from a text file (one prompt per line)."""
    prompts: List[Prompt] = []
    
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            text = line.strip()
            if text:  # Skip empty lines
                metadata = PromptMetadata(
                    dataset_id="custom",
                    dataset_split="custom",
                    attributes={"source_file": str(file_path), "line_number": line_num},
                )
                prompts.append(Prompt(text=text, metadata=metadata))
    
    return prompts


def load_custom_prompts_from_list(prompt_texts: List[str]) -> List[Prompt]:
    """Load custom prompts from a list of strings."""
    prompts: List[Prompt] = []
    
    for idx, text in enumerate(prompt_texts):
        if text.strip():  # Skip empty strings
            metadata = PromptMetadata(
                dataset_id="custom",
                dataset_split="custom",
                attributes={"index": idx},
            )
            prompts.append(Prompt(text=text.strip(), metadata=metadata))
    
    return prompts
