from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore


INSTRUCTION = (
    "You are a safety classifier. "
    "Classify the following text as SAFE or TOXIC. "
    "Respond with only one word."
)


def _extract_label(decoded: str) -> str:
    """Parse the generated string and extract SAFE/TOXIC labels."""
    tail = decoded.split("Answer:")[-1].strip().upper()
    first_token = tail.split()[0] if tail else ""

    if "TOXIC" in first_token:
        return "TOXIC"
    if "SAFE" in first_token:
        return "SAFE"
    return "UNKNOWN"


@dataclass
class ClassifierConfig:
    base_model: str = "google/gemma-2b-it"
    adapter_path: Optional[str] = None
    max_length: int = 256


class ToxicLoRAClassifier:
    """Gemma-based safety classifier that optionally loads LoRA adapters."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.adapter_path or self.config.base_model
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if BitsAndBytesConfig is not None:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except Exception:
                bnb_config = None  # fall back to full precision if bitsandbytes missing

        model_kwargs: Dict[str, Any] = {"device_map": "auto"}
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs,
        )
        if self.config.adapter_path:
            model = PeftModel.from_pretrained(model, self.config.adapter_path)
        self.model = model.eval()

    def classify_text(self, text: str) -> Dict[str, Any]:
        """Generate a safety label for the provided text."""
        prompt = f"{INSTRUCTION}\n\nText: {text}\n\nAnswer:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                temperature=0.0,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = _extract_label(decoded)
        return {"label": label, "raw": decoded}
    
