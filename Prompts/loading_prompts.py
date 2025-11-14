from datasets import load_dataset


def load_prompts(n=1000):
    """Load a subset of the Real Toxicity Prompts dataset."""
    
    ds = load_dataset("allenai/real-toxicity-prompts")
    small_ds = ds["train"].select(range(n))
    return small_ds
