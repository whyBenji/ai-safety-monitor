from typing import List

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
