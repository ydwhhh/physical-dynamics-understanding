"""Lazy Qwen2.5-VL loading and generation."""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenClassifier:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_new_tokens: int = 8) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this Phase 1 inference pipeline")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not report BF16 support")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def classify(self, images: Sequence[Image.Image], prompt: str) -> str:
        if not images:
            raise ValueError("at least one image is required")
        # Placeholder order is matched to the flat PIL image list passed below.
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[rendered],
            images=list(images),
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        continuation = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
